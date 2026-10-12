"""Read-only Train-bank/Val-query audit for selective retrieval policies."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import common
from .memory import (
    add_statistics,
    assert_no_query_target_interfaces,
    build_quality_view,
    device_bank,
    finish_statistics,
    read_banks,
    retrieve,
)
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    OutputLock,
    require_hard,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    encode_regions,
)
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import (
    target_distribution_and_quality,
)


CASES = (
    ('global_hard', 'global_hard', 'normal'),
    ('fixed_balanced', 'fixed_balanced', 'normal'),
    ('quality_adaptive', 'adaptive_cap', 'normal'),
    ('quality_adaptive_random', 'adaptive_cap', 'random'),
    ('quality_adaptive_shuffle_labels', 'adaptive_cap', 'shuffle_labels'),
    ('quality_adaptive_off', 'adaptive_cap', 'off'),
)


def new_row():
    return dict(
        queries=0, supported=0, neighbor_slots=0, valid_neighbors=0,
        valid_neighbors_by_class=[0] * 4,
        queries_with_class_by_class=[0] * 4,
        selected_hard_count_sum=0., selected_labelled_mass_sum=0.,
        selected_purity_sum=0., selected_entropy_sum=0.,
        cosine_sum=0., distinct_event_sum=0.,
        query_labelled_all=0, query_labelled_hard=0,
        knn_supported=0, knn_nll_sum=0., knn_mass_weight=0.,
        knn_mass_nll_sum=0., knn_mass_correct=0.,
        posterior_confidence_sum=0., posterior_entropy_sum=0.,
        knn_confusion=torch.zeros((4, 4), dtype=torch.long))


def retrieval_evidence(query, bank, event_id, image_id, args, policy,
                       intervention, generator):
    support, labels, valid, metadata = retrieve(
        query, bank, event_id, image_id, policy=policy,
        topk=args.topk, preselect=args.preselect,
        per_class_cap=args.per_class_cap, min_cosine=args.min_cosine,
        exclude_same_event=not args.allow_same_event,
        intervention=intervention, generator=generator)
    raw = dict(
        queries=len(query), supported=int(valid.any(-1).sum()),
        neighbor_slots=valid.numel(), valid_neighbors=int(valid.sum()),
        valid_neighbors_by_class=[0] * 4,
        queries_with_class_by_class=[0] * 4)
    for class_index in range(4):
        selected = valid & (metadata['dominant_class'] == class_index)
        raw['valid_neighbors_by_class'][class_index] = int(selected.sum())
        raw['queries_with_class_by_class'][class_index] = int(
            selected.any(-1).sum())
    for key in ('hard_count', 'labelled_mass', 'purity', 'entropy'):
        values = torch.where(
            valid, metadata[key].float(),
            torch.zeros_like(metadata[key], dtype=torch.float32))
        raw['selected_' + key + '_sum'] = float(values.sum())
    cosine = torch.where(
        valid, metadata['cosine'].float(),
        torch.zeros_like(metadata['cosine'], dtype=torch.float32))
    raw['cosine_sum'] = float(cosine.sum())
    raw['selected_cosine_sum'] = raw['cosine_sum']
    event_count = max(1, len(bank['event_names']))
    event_ids = metadata['events'].clamp(0, event_count - 1)
    event_flags = F.one_hot(event_ids, num_classes=event_count).bool()
    raw['distinct_event_sum'] = int(
        (event_flags & valid[..., None]).any(1).sum())
    supported_query = valid.any(-1)
    scores = metadata['cosine'].float() / args.temperature
    scores = scores.masked_fill(~valid, -torch.inf)
    weights = torch.zeros_like(scores)
    if supported_query.any():
        weights[supported_query] = torch.softmax(
            scores[supported_query], dim=-1)
    safe_labels = torch.where(
        valid[..., None], labels.float(), torch.zeros_like(labels.float()))
    posterior = (weights[..., None] * safe_labels).sum(1)
    if supported_query.any():
        selected = posterior[supported_query].clamp_min(1e-12)
        raw['posterior_confidence_sum'] = float(selected.max(-1).values.sum())
        raw['posterior_entropy_sum'] = float(
            (-(selected * selected.log()).sum(-1)).sum())
    else:
        raw['posterior_confidence_sum'] = 0.
        raw['posterior_entropy_sum'] = 0.
    return raw, posterior, supported_query


def finish_row(row):
    row = dict(row)
    confusion = row.pop('knn_confusion')
    finished = finish_statistics(row)
    neighbors = row['valid_neighbors']
    queries = row['queries']
    supported = row['knn_supported']
    finished.update(
        mean_query_neighbor_cosine=(
            row['cosine_sum'] / max(1, neighbors)),
        mean_distinct_support_events=(
            row['distinct_event_sum'] / max(1, queries)),
        mean_posterior_confidence=(
            row['posterior_confidence_sum'] / max(1, row['supported'])),
        mean_posterior_entropy=(
            row['posterior_entropy_sum'] / max(1, row['supported'])),
        knn_supported_queries=supported,
        knn_hard_query_coverage=(
            supported / max(1, row['query_labelled_hard'])),
        knn_all_positive_query_coverage=(
            supported / max(1, row['query_labelled_all'])),
        knn_soft_cross_entropy=(
            row['knn_nll_sum'] / max(1, supported)),
        knn_mass_weighted_soft_cross_entropy=(
            row['knn_mass_nll_sum'] /
            max(1e-12, row['knn_mass_weight'])),
        knn_mass_weighted_accuracy=(
            row['knn_mass_correct'] /
            max(1e-12, row['knn_mass_weight'])),
        knn_confusion=confusion.tolist(),
        knn_dominant_accuracy=float(
            confusion.diag().sum() / max(1, int(confusion.sum()))),
        knn_per_class_recall=[
            float(confusion[i, i] / max(1, int(confusion[i].sum())))
            for i in range(4)])
    return finished


def _validate(args):
    require_hard(args)
    if args.topk < 1 or args.preselect < args.topk:
        raise ValueError('Require topk>=1 and preselect>=topk')
    if args.topk < 4 or args.topk % 4:
        raise ValueError('fixed_balanced audit requires topk divisible by four')
    if args.topk < 4 or args.topk % 4:
        raise ValueError(
            'Audit includes fixed_balanced, so topk must be divisible by four')
    if args.per_class_cap < 1:
        raise ValueError('per_class_cap must be positive')
    if not 0 <= args.quality_quantile < .5:
        raise ValueError('quality_quantile must lie in [0,.5)')
    if args.min_cosine != -1.:
        raise ValueError(
            'Selective V1 fixes min_cosine=-1 so its random control is fair')
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        raise ValueError('temperature must be finite and positive')


def run(args):
    _validate(args)
    assert_no_query_target_interfaces()
    begin = time.monotonic()
    common.seed_everything(args.seed)
    code_hash = common.experiment_hash()
    parent_hash = common.parent_experiment_hash()
    bundle = common.load_source(args)
    identity = common.source_identity(bundle, args)
    banks_cpu, bank_manifest = read_banks(
        args.bank, identity, args.smoke, common.digest, bundle['train'])
    bank_cpu = build_quality_view(
        banks_cpu['hard'], quantile=args.quality_quantile)
    bank = device_bank(bank_cpu, bundle['device'])
    bank_files = {
        str(Path(args.bank) / filename): common.digest(Path(args.bank) / filename)
        for filename in ('bank_all.pt', 'bank_hard.pt', 'manifest.json')
    }
    out = common.new_output(args.output)
    common.save_json(out / 'manifest.json', dict(
        format='ROI_MEMORY_SELECTIVE_AUDIT_V1', complete=False,
        arguments=vars(args), identity=identity, code_hash=code_hash,
        parent_code_hash=parent_hash, bank_format=bank_manifest['format'],
        bank_file_sha256=bank_files, smoke=args.smoke,
        test_accessed=False,
        quality_quantile=bank['quality_quantile'],
        quality_thresholds=[dict(row) for row in bank['quality_thresholds']],
        quality_eligible_regions=int(bank['quality_eligible'].sum().item()),
        ranking=(
            'query-token cosine (or isolated random intervention); '
            'query GT is absent'),
        quality=(
            'classwise Train-bank thresholds only; no Val labels or '
            'query labels enter selection'),
        adaptive=(
            'global similarity order, per-class maximum, no minimum quota, '
            'no backfill when support is insufficient'),
        evidence=(
            'temperature-weighted support-label posterior; Val GT is '
            'consumed only after every neighbor set is fixed')))
    rows = {name: new_row() for name, _, _ in CASES}
    validation = common.loader(
        args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
        workers=args.workers, seed=args.seed + 3)
    generators = {
        name: torch.Generator(device=bundle['device']).manual_seed(
            args.seed + 700 + index)
        for index, (name, _, _) in enumerate(CASES)
    }
    image_count = 0
    with torch.no_grad():
        for batch in validation:
            pre, post, _, gt, image_names = common.parse_batch(
                batch, bundle['device'])
            reference = common.capture(bundle['front'], pre, post)
            pool_route, _ = common.routes(
                reference, bundle['manifest'], 'hard', 'hard')
            encoded = encode_regions(
                bundle['heads']['hard'], reference['features']['roi'],
                pool_route)
            for image_index, image_name in enumerate(image_names):
                active = (encoded['mass'][image_index] > 0).nonzero().flatten()
                query = encoded['tokens'][image_index, active]
                event_name = common.event(image_name)
                event_id = bank['event_lookup'].get(event_name, -1)
                image_id = bank['image_lookup'].get(image_name, -1)
                pending = {}
                for case_name, retrieval_policy, intervention in CASES:
                    pending[case_name] = retrieval_evidence(
                        query, bank, event_id, image_id, args,
                        retrieval_policy, intervention,
                        generators[case_name])
                # Val GT is deliberately computed only after all retrievals.
                single_route = {
                    key: (value[image_index:image_index + 1]
                          if torch.is_tensor(value) and value.ndim > 0 and
                          value.shape[0] == len(gt) else value)
                    for key, value in pool_route.items()
                }
                distribution, labelled_mass, hard_count, _, _ = (
                    target_distribution_and_quality(
                        gt[image_index:image_index + 1], single_route))
                target = distribution[0, active]
                query_mass = labelled_mass[0, active]
                labelled = query_mass > 0
                hard_labelled = labelled & (hard_count[0, active] > 0)
                truth = target.argmax(-1)
                for case_name, (raw, posterior, supported_query) in pending.items():
                    row = rows[case_name]
                    add_statistics(row, raw)
                    row['query_labelled_all'] += int(labelled.sum())
                    row['query_labelled_hard'] += int(hard_labelled.sum())
                    eligible = hard_labelled & supported_query
                    row['knn_supported'] += int(eligible.sum())
                    if eligible.any():
                        nll = -(target[eligible] *
                                posterior[eligible].clamp_min(1e-12).log()).sum(-1)
                        weight = query_mass[eligible]
                        row['knn_nll_sum'] += float(nll.sum())
                        row['knn_mass_weight'] += float(weight.sum())
                        row['knn_mass_nll_sum'] += float((nll * weight).sum())
                        predicted = posterior[eligible].argmax(-1)
                        row['knn_mass_correct'] += float(
                            (weight * (predicted == truth[eligible])).sum())
                        row['knn_confusion'] += torch.bincount(
                            (truth[eligible] * 4 + predicted).cpu(),
                            minlength=16).reshape(4, 4)
            image_count += len(image_names)
            if image_count % 25 == 0:
                print('selective support audit', image_count, flush=True)
    expected = 2 if args.smoke else 450
    if image_count != expected:
        raise RuntimeError(
            f'Expected {expected} Val images, received {image_count}')
    common.assert_selective_immutable(bundle, code_hash, parent_hash)
    for path, expected_hash in bank_files.items():
        if common.digest(path) != expected_hash:
            raise RuntimeError('Bank changed during audit: ' + path)
    result = dict(
        complete=True, smoke=args.smoke, test_accessed=False,
        image_count=image_count, temperature=args.temperature,
        quality_eligible_regions=int(bank['quality_eligible'].sum().item()),
        quality_thresholds=[dict(row) for row in bank['quality_thresholds']],
        cases={name: finish_row(row) for name, row in rows.items()},
        frozen_source_verified=True, bank_immutable_verified=True,
        elapsed_seconds=time.monotonic() - begin,
        caution=(
            'kNN evidence is retrieval screening, not trained pixel mIoU. '
            'Val GT is used only after Train-bank feature-only ranking.'))
    common.save_json(out / 'audit.json', result)
    common.save_json(out / 'summary.json', result)
    manifest = json.loads(
        (out / 'manifest.json').read_text(encoding='utf-8'))
    manifest['complete'] = True
    common.save_json(out / 'manifest.json', manifest)
    print('SELECTIVE SUPPORT AUDIT COMPLETE. No weights updated; no Test.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--preselect', type=int, default=32)
    parser.add_argument('--per_class_cap', type=int, default=4)
    parser.add_argument('--quality_quantile', type=float, default=.25)
    parser.add_argument('--min_cosine', type=float, default=-1.)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--allow_same_event', action='store_true')
    args = parser.parse_args()
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
