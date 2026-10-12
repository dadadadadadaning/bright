"""Read-only support audit before training any residual Memory head."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import common
from .memory import (CLASS_NAMES, add_statistics, assert_no_query_target_interfaces,
                     device_bank, finish_statistics, read_banks, retrieve,
                     target_distribution_and_quality)
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    OutputLock, require_hard)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions


CASES = (
    ('global_all', 'all', False, 'normal'),
    ('global_hard', 'hard', False, 'normal'),
    ('balanced_hard', 'hard', True, 'normal'),
    ('balanced_hard_random', 'hard', True, 'random'),
    ('balanced_hard_shuffle_labels', 'hard', True, 'shuffle_labels'),
)


def new_row():
    return dict(queries=0, supported=0, neighbor_slots=0, valid_neighbors=0,
                valid_neighbors_by_class=[0] * 4,
                queries_with_class_by_class=[0] * 4,
                selected_hard_count_sum=0., selected_labelled_mass_sum=0.,
                selected_purity_sum=0., selected_entropy_sum=0.,
                cosine_sum=0., distinct_event_sum=0.,
                query_labelled_all=0, query_labelled_hard=0,
                knn_supported=0, knn_nll_sum=0., knn_mass_weight=0.,
                knn_mass_nll_sum=0., knn_mass_correct=0.,
                knn_confusion=torch.zeros((4, 4), dtype=torch.long))


def retrieval_evidence(query, bank, event_id, image_id, topk, exclude_same_event,
                       intervention, balanced, temperature, generator):
    support, labels, valid, metadata = retrieve(
        query, bank, event_id, image_id, topk, exclude_same_event,
        intervention, balanced, generator)
    raw = dict(queries=len(query), supported=int(valid.any(-1).sum()),
               neighbor_slots=valid.numel(), valid_neighbors=int(valid.sum()),
               valid_neighbors_by_class=[0] * 4,
               queries_with_class_by_class=[0] * 4)
    for cls in range(4):
        selected = valid & (metadata['dominant_class'] == cls)
        raw['valid_neighbors_by_class'][cls] = int(selected.sum())
        raw['queries_with_class_by_class'][cls] = int(selected.any(-1).sum())
    for key in ('hard_count', 'labelled_mass', 'purity', 'entropy'):
        raw['selected_' + key + '_sum'] = float(metadata[key][valid].float().sum())
    raw['cosine_sum'] = float(metadata['cosine'][valid].sum())
    event_flags = F.one_hot(metadata['events'],
                            num_classes=len(bank['event_names'])).bool()
    raw['distinct_event_sum'] = int(
        (event_flags & valid[..., None]).any(1).sum())
    scores = metadata['cosine'] / temperature
    scores = scores.masked_fill(~valid, -torch.inf)
    supported = valid.any(-1)
    weights = torch.zeros_like(scores)
    if supported.any():
        weights[supported] = torch.softmax(scores[supported], dim=-1)
    probability = (weights[..., None] * labels).sum(1)
    return raw, probability, supported


def finish_row(row):
    cm = row.pop('knn_confusion')
    finished = finish_statistics(row)
    neighbors, queries = row['valid_neighbors'], row['queries']
    finished.update(
        mean_query_neighbor_cosine=row['cosine_sum'] / max(1, neighbors),
        mean_distinct_support_events=row['distinct_event_sum'] / max(1, queries),
        knn_supported_queries=row['knn_supported'],
        knn_hard_query_coverage=(row['knn_supported'] /
                                 max(1, row['query_labelled_hard'])),
        knn_all_positive_query_coverage=(row['knn_supported'] /
                                         max(1, row['query_labelled_all'])),
        knn_soft_cross_entropy=row['knn_nll_sum'] / max(1, row['knn_supported']),
        knn_mass_weighted_soft_cross_entropy=(row['knn_mass_nll_sum'] /
                                              max(1e-12, row['knn_mass_weight'])),
        knn_mass_weighted_accuracy=(row['knn_mass_correct'] /
                                    max(1e-12, row['knn_mass_weight'])),
        knn_confusion=cm.tolist(),
        knn_dominant_accuracy=float(cm.diag().sum() / max(1, int(cm.sum()))),
        knn_per_class_recall=[float(cm[i, i] / max(1, int(cm[i].sum()))) for i in range(4)],
    )
    return finished


def run(args):
    require_hard(args)
    if args.topk < 4 or args.topk % 4 or args.temperature <= 0:
        raise ValueError('topk must be divisible by four and temperature positive')
    assert_no_query_target_interfaces()
    begin = time.monotonic()
    common.seed_everything(args.seed)
    code_hash = common.experiment_hash()
    bundle = common.load_source(args)
    identity = common.source_identity(bundle, args)
    banks_cpu, bank_manifest = read_banks(
        args.bank, identity, args.smoke, common.digest, bundle['train'])
    banks = {name: device_bank(bank, bundle['device'])
             for name, bank in banks_cpu.items()}
    bank_files = {str(Path(args.bank) / filename): common.digest(Path(args.bank) / filename)
                  for filename in ('bank_all.pt', 'bank_hard.pt', 'manifest.json')}
    out = common.new_output(args.output)
    common.save_json(out / 'manifest.json', dict(
        format='ROI_MEMORY_SUPPORT_AUDIT_V2', complete=False,
        arguments=vars(args), identity=identity, code_hash=code_hash,
        bank_format=bank_manifest['format'], bank_file_sha256=bank_files,
        smoke=args.smoke, test_accessed=False,
        ranking='feature cosine only (or random intervention); query GT absent',
        evidence='temperature-weighted support soft labels; primary query eligibility is labelled_mass>0 AND query hard_count>0; GT consumed only after retrieval'))
    rows = {name: new_row() for name, _, _, _ in CASES}
    validation = common.loader(
        args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
        workers=args.workers, seed=args.seed + 3)
    generators = {name: torch.Generator(device=bundle['device']).manual_seed(args.seed + 700 + i)
                  for i, (name, _, _, _) in enumerate(CASES)}
    image_count = 0
    with torch.no_grad():
        for batch in validation:
            pre, post, _, gt, image_names = common.parse_batch(batch, bundle['device'])
            reference = common.capture(bundle['front'], pre, post)
            pool_route, _ = common.routes(reference, bundle['manifest'], 'hard', 'hard')
            encoded = encode_regions(bundle['heads']['hard'],
                                     reference['features']['roi'], pool_route)
            for image_index, image_name in enumerate(image_names):
                active = (encoded['mass'][image_index] > 0).nonzero().flatten()
                query = encoded['tokens'][image_index, active]
                event_name = common.event(image_name)
                pending = {}
                for case_name, bank_name, balanced, intervention in CASES:
                    bank = banks[bank_name]
                    evidence = retrieval_evidence(
                        query, bank, bank['event_lookup'].get(event_name, -1),
                        bank['image_lookup'].get(image_name, -1), args.topk,
                        not args.allow_same_event, intervention, balanced,
                        args.temperature, generators[case_name])
                    pending[case_name] = evidence
                # Query GT is computed only after every neighbor set is fixed.
                distribution, labelled_mass, query_hard_count, _, _ = target_distribution_and_quality(
                    gt[image_index:image_index + 1], {
                        key: (value[image_index:image_index + 1]
                              if torch.is_tensor(value) and value.ndim > 0
                              and value.shape[0] == len(gt) else value)
                        for key, value in pool_route.items()})
                target = distribution[0, active]
                query_mass = labelled_mass[0, active]
                labelled = query_mass > 0
                hard_labelled = labelled & (query_hard_count[0, active] > 0)
                truth = target.argmax(-1)
                for case_name, (raw, probability, supported) in pending.items():
                    row = rows[case_name]
                    add_statistics(row, raw)
                    row['query_labelled_all'] += int(labelled.sum())
                    row['query_labelled_hard'] += int(hard_labelled.sum())
                    eligible = hard_labelled & supported
                    row['knn_supported'] += int(eligible.sum())
                    if eligible.any():
                        nll = -(target[eligible] * probability[eligible].clamp_min(1e-12).log()).sum(-1)
                        weight = query_mass[eligible]
                        row['knn_nll_sum'] += float(nll.sum())
                        row['knn_mass_weight'] += float(weight.sum())
                        row['knn_mass_nll_sum'] += float((nll * weight).sum())
                        predicted = probability[eligible].argmax(-1)
                        row['knn_mass_correct'] += float(
                            (weight * (predicted == truth[eligible])).sum())
                        row['knn_confusion'] += torch.bincount(
                            (truth[eligible] * 4 + predicted).cpu(), minlength=16).reshape(4, 4)
            image_count += len(image_names)
            if image_count % 25 == 0:
                print('support audit', image_count, flush=True)
    expected = 2 if args.smoke else 450
    if image_count != expected:
        raise RuntimeError(f'Expected {expected} Val images, received {image_count}')
    common.assert_v2_immutable(bundle, code_hash)
    for path, expected_hash in bank_files.items():
        if common.digest(path) != expected_hash:
            raise RuntimeError('Bank changed during audit: ' + path)
    result = dict(
        complete=True, smoke=args.smoke, test_accessed=False,
        image_count=image_count, temperature=args.temperature,
        cases={name: finish_row(row) for name, row in rows.items()},
        frozen_source_verified=True, bank_immutable_verified=True,
        elapsed_seconds=time.monotonic() - begin,
        caution=('kNN evidence uses Train support labels and post-retrieval Val GT only for audit; '
                 'it is not final pixel mIoU or a trained classifier'))
    common.save_json(out / 'audit.json', result)
    common.save_json(out / 'summary.json', result)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    manifest['complete'] = True
    common.save_json(out / 'manifest.json', manifest)
    print('SUPPORT AUDIT COMPLETE. Read-only Train bank / Val queries; no Test.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--allow_same_event', action='store_true')
    args = parser.parse_args()
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
