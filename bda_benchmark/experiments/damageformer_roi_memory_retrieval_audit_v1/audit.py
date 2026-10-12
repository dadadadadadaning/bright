"""GT-conditioned, read-only Val audit of frozen Train-Memory retrieval.

Retrieval is always fixed before Val GT is inspected.  GT is used only to
measure whether the selected Train supports are useful for each query class.
No optimizer, backward pass, checkpoint write, or Test split is supported.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import common
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    OutputLock,
    require_hard,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    encode_regions,
)
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import (
    CLASS_NAMES,
    target_distribution_and_quality,
)
from bda_benchmark.experiments.damageformer_roi_memory_selective_v1.memory import (
    build_quality_view,
    device_bank,
    read_banks,
    retrieve,
)


GROUP_NAMES = ('predicted_background', 'predicted_building')
PURITY_THRESHOLD = .95
CASES = (
    dict(name='global_exclude_event', policy='global_hard',
         intervention='normal', exclude_same_event=True, same_group=False),
    dict(name='balanced_exclude_event', policy='fixed_balanced',
         intervention='normal', exclude_same_event=True, same_group=False),
    dict(name='adaptive_exclude_event', policy='adaptive_cap',
         intervention='normal', exclude_same_event=True, same_group=False),
    dict(name='adaptive_random_exclude_event', policy='adaptive_cap',
         intervention='random', exclude_same_event=True, same_group=False),
    dict(name='adaptive_shuffle_labels_exclude_event', policy='adaptive_cap',
         intervention='shuffle_labels', exclude_same_event=True,
         same_group=False),
    dict(name='adaptive_allow_same_event', policy='adaptive_cap',
         intervention='normal', exclude_same_event=False, same_group=False),
    dict(name='adaptive_same_predicted_group', policy='adaptive_cap',
         intervention='normal', exclude_same_event=True, same_group=True),
)


def empty_stats():
    return dict(
        query_by_class=torch.zeros(4, dtype=torch.long),
        query_mass_by_class=torch.zeros(4, dtype=torch.float64),
        supported_by_class=torch.zeros(4, dtype=torch.long),
        supported_mass_by_class=torch.zeros(4, dtype=torch.float64),
        same_class_hit_by_class=torch.zeros(4, dtype=torch.long),
        same_class_hit_mass_by_class=torch.zeros(4, dtype=torch.float64),
        valid_neighbors_by_query_class=torch.zeros(4, dtype=torch.long),
        same_group_neighbors_by_query_class=torch.zeros(4, dtype=torch.long),
        support_class_matrix=torch.zeros((4, 4), dtype=torch.long),
        injected_label_class_matrix=torch.zeros((4, 4), dtype=torch.long),
        top1_support_confusion=torch.zeros((4, 4), dtype=torch.long),
        posterior_confusion=torch.zeros((4, 4), dtype=torch.long),
        posterior_confusion_mass=torch.zeros((4, 4), dtype=torch.float64),
        soft_ce_sum_by_class=torch.zeros(4, dtype=torch.float64),
        mass_soft_ce_sum_by_class=torch.zeros(4, dtype=torch.float64),
        mass_sum_by_class=torch.zeros(4, dtype=torch.float64),
    )


def _clone_stats(row):
    return {key: value.clone() for key, value in row.items()}


def add_stats(total, row):
    for key, value in row.items():
        total[key] += value


def _safe_rows(matrix):
    matrix = matrix.double()
    denominator = matrix.sum(-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).tiny)
    return matrix / denominator


def _ratio(numerator, denominator):
    denominator = float(denominator)
    return None if denominator <= 0 else float(numerator) / denominator


def finish_stats(row):
    query = row['query_by_class']
    supported = row['supported_by_class']
    neighbors = row['valid_neighbors_by_query_class']
    same_group = row['same_group_neighbors_by_query_class']
    confusion = row['posterior_confusion']
    confusion_mass = row['posterior_confusion_mass']
    support_matrix = row['support_class_matrix']
    confusion_rate = _safe_rows(confusion)
    confusion_mass_rate = _safe_rows(confusion_mass)
    support_fraction = _safe_rows(support_matrix)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = dict(
            queries=int(query[index]),
            query_mass=float(row['query_mass_by_class'][index]),
            supported_queries=int(supported[index]),
            support_coverage=_ratio(supported[index], query[index]),
            mass_weighted_support_coverage=_ratio(
                row['supported_mass_by_class'][index],
                row['query_mass_by_class'][index]),
            same_class_hit_at_k=_ratio(
                row['same_class_hit_by_class'][index], query[index]),
            mass_weighted_same_class_hit_at_k=_ratio(
                row['same_class_hit_mass_by_class'][index],
                row['query_mass_by_class'][index]),
            same_class_neighbor_fraction=_ratio(
                support_matrix[index, index], support_matrix[index].sum()),
            posterior_recall=_ratio(
                confusion[index, index], confusion[index].sum()),
            mass_weighted_posterior_recall=_ratio(
                confusion_mass[index, index], confusion_mass[index].sum()),
            posterior_to_class={
                CLASS_NAMES[j]: (
                    None if not int(confusion[index].sum()) else
                    float(confusion_rate[index, j]))
                for j in range(4)},
            unconditional_posterior_to_class={
                CLASS_NAMES[j]: _ratio(confusion[index, j], query[index])
                for j in range(4)},
            mass_weighted_posterior_to_class={
                CLASS_NAMES[j]: (
                    None if float(confusion_mass[index].sum()) <= 0 else
                    float(confusion_mass_rate[index, j]))
                for j in range(4)},
            neighbor_class_fraction={
                CLASS_NAMES[j]: (
                    None if not int(support_matrix[index].sum()) else
                    float(support_fraction[index, j]))
                for j in range(4)},
            mean_valid_neighbors=_ratio(neighbors[index], supported[index]),
            same_predicted_group_neighbor_fraction=_ratio(
                same_group[index], neighbors[index]),
            soft_cross_entropy=_ratio(
                row['soft_ce_sum_by_class'][index], supported[index]),
            mass_weighted_soft_cross_entropy=_ratio(
                row['mass_soft_ce_sum_by_class'][index],
                row['mass_sum_by_class'][index]),
        )
    return dict(
        queries=int(query.sum()),
        query_mass=float(row['query_mass_by_class'].sum()),
        supported_queries=int(supported.sum()),
        support_coverage=_ratio(supported.sum(), query.sum()),
        mass_weighted_support_coverage=_ratio(
            row['supported_mass_by_class'].sum(),
            row['query_mass_by_class'].sum()),
        same_class_hit_at_k=_ratio(
            row['same_class_hit_by_class'].sum(), query.sum()),
        mass_weighted_same_class_hit_at_k=_ratio(
            row['same_class_hit_mass_by_class'].sum(),
            row['query_mass_by_class'].sum()),
        posterior_accuracy=_ratio(confusion.diag().sum(), confusion.sum()),
        mass_weighted_posterior_accuracy=_ratio(
            confusion_mass.diag().sum(), confusion_mass.sum()),
        same_predicted_group_neighbor_fraction=_ratio(
            same_group.sum(), neighbors.sum()),
        posterior_confusion=confusion.tolist(),
        posterior_confusion_mass=confusion_mass.tolist(),
        top1_support_confusion=row['top1_support_confusion'].tolist(),
        support_class_matrix=support_matrix.tolist(),
        support_class_basis=(
            'stored Train-support dominant class before label intervention'),
        injected_label_class_matrix=(
            row['injected_label_class_matrix'].tolist()),
        per_class=per_class,
    )


def _group_filtered_bank(bank, group):
    result = dict(bank)
    result['quality_eligible'] = (
        bank['quality_eligible'] & ((bank['regions'] % 2) == int(group)))
    return result


@torch.no_grad()
def retrieve_case(query, query_group, bank, event_id, image_id, case, args,
                  generator):
    """Retrieve without accepting query GT or labels.

    The same-group case uses only the *predicted* hard-route slot parity.  It
    does not use a building GT mask or damage target.
    """
    kwargs = dict(
        policy=case['policy'], topk=args.topk, preselect=args.preselect,
        per_class_cap=args.per_class_cap, min_cosine=args.min_cosine,
        exclude_same_event=case['exclude_same_event'],
        intervention=case['intervention'], generator=generator)
    if not case['same_group']:
        return retrieve(query, bank, event_id, image_id, **kwargs)
    if case['policy'] != 'adaptive_cap':
        raise ValueError('The isolated same-group audit uses adaptive_cap only')
    n, dim = query.shape
    support = query.new_zeros((n, args.topk, dim))
    labels = query.new_zeros((n, args.topk, 4))
    valid = torch.zeros((n, args.topk), dtype=torch.bool, device=query.device)
    metadata = {}
    for group in (0, 1):
        selected = (query_group == group).nonzero().flatten()
        if not len(selected):
            continue
        part = retrieve(
            query[selected], _group_filtered_bank(bank, group),
            event_id, image_id, **kwargs)
        part_support, part_labels, part_valid, part_meta = part
        support.index_copy_(0, selected, part_support)
        labels.index_copy_(0, selected, part_labels)
        valid.index_copy_(0, selected, part_valid)
        for key, value in part_meta.items():
            if key not in metadata:
                metadata[key] = value.new_zeros((n, *value.shape[1:]))
            metadata[key].index_copy_(0, selected, value)
    return support, labels, valid, metadata


@torch.no_grad()
def evidence_row(target, labelled_mass, hard_count, query_group, labels,
                 valid, metadata, temperature, query_mask=None):
    if query_mask is None:
        query_mask = torch.ones(len(target), dtype=torch.bool,
                                device=target.device)
    eligible = ((labelled_mass > 0) & (hard_count > 0) & query_mask)
    row = empty_stats()
    if not eligible.any():
        return row
    truth = target.argmax(-1)
    for cls in range(4):
        selected = eligible & (truth == cls)
        row['query_by_class'][cls] = int(selected.sum())
        row['query_mass_by_class'][cls] = (
            labelled_mass[selected].double().sum().cpu())

    active = eligible.nonzero().flatten()
    truth_active = truth[active]
    valid_active = valid[active]
    supported = valid_active.any(-1)
    for cls in range(4):
        selected = supported & (truth_active == cls)
        row['supported_by_class'][cls] = int(selected.sum())
        row['supported_mass_by_class'][cls] = (
            labelled_mass[active][selected].double().sum().cpu())

    support_class = metadata['dominant_class'][active]
    injected_class = labels[active].argmax(-1)
    same_class = (support_class == truth_active[:, None]) & valid_active
    for cls in range(4):
        hit = same_class.any(-1) & (truth_active == cls)
        row['same_class_hit_by_class'][cls] = int(hit.sum())
        row['same_class_hit_mass_by_class'][cls] = (
            labelled_mass[active][hit].double().sum().cpu())
        selected = supported & (truth_active == cls)
        if selected.any():
            row['valid_neighbors_by_query_class'][cls] = int(
                valid_active[selected].sum())
            group_match = (
                (metadata['regions'][active][selected] % 2) ==
                query_group[active][selected, None]) & valid_active[selected]
            row['same_group_neighbors_by_query_class'][cls] = int(
                group_match.sum())
            pairs = (
                truth_active[selected, None] * 4 + support_class[selected])
            row['support_class_matrix'] += torch.bincount(
                pairs[valid_active[selected]].cpu(), minlength=16).reshape(4, 4)
            injected_pairs = (
                truth_active[selected, None] * 4 + injected_class[selected])
            row['injected_label_class_matrix'] += torch.bincount(
                injected_pairs[valid_active[selected]].cpu(),
                minlength=16).reshape(4, 4)

    if supported.any():
        scores = metadata['cosine'][active][supported].float() / temperature
        selected_valid = valid_active[supported]
        scores = scores.masked_fill(~selected_valid, -torch.inf)
        supported_truth = truth_active[supported]
        top1 = support_class[supported].gather(
            1, scores.argmax(-1, keepdim=True)).squeeze(1)
        row['top1_support_confusion'] += torch.bincount(
            (supported_truth * 4 + top1).cpu(),
            minlength=16).reshape(4, 4)
        weights = torch.softmax(scores, dim=-1)
        safe_labels = torch.where(
            selected_valid[..., None], labels[active][supported].float(),
            torch.zeros_like(labels[active][supported].float()))
        posterior = (weights[..., None] * safe_labels).sum(1)
        posterior = posterior.clamp_min(1e-12)
        posterior = posterior / posterior.sum(-1, keepdim=True)
        predicted = posterior.argmax(-1)
        row['posterior_confusion'] += torch.bincount(
            (supported_truth * 4 + predicted).cpu(),
            minlength=16).reshape(4, 4)
        mass = labelled_mass[active][supported]
        row['posterior_confusion_mass'] += torch.bincount(
            (supported_truth * 4 + predicted).cpu(),
            weights=mass.double().cpu(), minlength=16).reshape(4, 4)
        soft_target = target[active][supported]
        ce = -(soft_target * posterior.log()).sum(-1)
        for cls in range(4):
            selected = supported_truth == cls
            if selected.any():
                row['soft_ce_sum_by_class'][cls] += ce[selected].double().sum().cpu()
                row['mass_soft_ce_sum_by_class'][cls] += (
                    ce[selected] * mass[selected]).double().sum().cpu()
                row['mass_sum_by_class'][cls] += mass[selected].double().sum().cpu()
    return row


def validate_args(args):
    require_hard(args)
    if args.topk < 4 or args.topk % 4:
        raise ValueError('topk must be positive and divisible by four')
    if args.preselect < args.topk or args.per_class_cap < 1:
        raise ValueError('Require preselect>=topk and positive per_class_cap')
    if not 0 <= args.quality_quantile < .5:
        raise ValueError('quality_quantile must lie in [0,.5)')
    if args.min_cosine != -1.:
        raise ValueError('Use min_cosine=-1 for a fair random control')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if args.limit_images < 0:
        raise ValueError('limit_images must be nonnegative')


def run(args):
    validate_args(args)
    common.seed_everything(args.seed)
    code_hash = common.implementation_hash()
    parent_hash = common.parent_hash()
    bundle = common.load_source(args)
    identity = common.source_identity(bundle, args)
    banks_cpu, bank_manifest = read_banks(
        args.bank, identity, args.smoke, common.digest, bundle['train'])
    bank_cpu = build_quality_view(
        banks_cpu['hard'], quantile=args.quality_quantile)
    bank = device_bank(bank_cpu, bundle['device'])
    bank_files = {
        str(Path(args.bank) / filename): common.digest(Path(args.bank) / filename)
        for filename in ('bank_all.pt', 'bank_hard.pt', 'manifest.json')}
    out = common.new_output(args.output)
    begin = time.monotonic()
    common.save_json(out / 'manifest.json', dict(
        format='ROI_MEMORY_RETRIEVAL_AUDIT_V1', complete=False,
        arguments=vars(args), cases=CASES, identity=identity,
        code_hash=code_hash, parent_code_hash=parent_hash,
        bank_format=bank_manifest['format'], bank_file_sha256=bank_files,
        smoke=args.smoke, pipeline_smoke=bool(args.limit_images),
        limited_images=args.limit_images,
        test_accessed=False,
        gt_boundary=(
            'Val GT is consumed only after every support index is fixed; '
            'GT never enters retrieval, ranking, or same-group masking'),
        group_definition=(
            'slot parity from predicted B>=threshold hard routing; this is '
            'not a GT building instance or GT foreground mask')))
    totals = {case['name']: {
        'all_hard_labelled': empty_stats(),
        'purity_ge_095': empty_stats()} for case in CASES}
    events = {case['name']: {} for case in CASES}
    groups = {
        case['name']: {name: {
            'all_hard_labelled': empty_stats(),
            'purity_ge_095': empty_stats()} for name in GROUP_NAMES}
        for case in CASES}
    generators = {
        case['name']: torch.Generator(device=bundle['device']).manual_seed(
            args.seed + 900 + index)
        for index, case in enumerate(CASES)}
    val_names = (bundle['val'][:args.limit_images]
                 if args.limit_images else bundle['val'])
    validation = common.loader(
        args.dataset, val_names, workers=args.workers, seed=args.seed + 3)
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
                query_group = (active % 2).long()
                event_name = common.event(image_name)
                event_id = bank['event_lookup'].get(event_name, -1)
                image_id = bank['image_lookup'].get(image_name, -1)
                pending = {}
                for case in CASES:
                    pending[case['name']] = retrieve_case(
                        query, query_group, bank, event_id, image_id, case,
                        args, generators[case['name']])

                # GT is deliberately introduced only after all retrievals.
                single_route = {
                    key: (value[image_index:image_index + 1]
                          if torch.is_tensor(value) and value.ndim > 0 and
                          value.shape[0] == len(gt) else value)
                    for key, value in pool_route.items()}
                target, mass, hard_count, _, _ = target_distribution_and_quality(
                    gt[image_index:image_index + 1], single_route)
                target, mass, hard_count = (
                    target[0, active], mass[0, active], hard_count[0, active])
                purity_mask = target.max(-1).values >= PURITY_THRESHOLD
                for case in CASES:
                    name = case['name']
                    _, labels, valid, metadata = pending[name]
                    rows = {
                        'all_hard_labelled': evidence_row(
                            target, mass, hard_count, query_group, labels,
                            valid, metadata, args.temperature),
                        'purity_ge_095': evidence_row(
                            target, mass, hard_count, query_group, labels,
                            valid, metadata, args.temperature,
                            query_mask=purity_mask),
                    }
                    event_rows = events[name].setdefault(event_name, {
                        'all_hard_labelled': empty_stats(),
                        'purity_ge_095': empty_stats()})
                    for stratum, row in rows.items():
                        add_stats(totals[name][stratum], row)
                        add_stats(event_rows[stratum], row)
                    for group_id, group_name in enumerate(GROUP_NAMES):
                        masks = {
                            'all_hard_labelled': query_group == group_id,
                            'purity_ge_095': ((query_group == group_id) &
                                             purity_mask),
                        }
                        for stratum, mask in masks.items():
                            group_row = evidence_row(
                                target, mass, hard_count, query_group, labels,
                                valid, metadata, args.temperature,
                                query_mask=mask)
                            add_stats(groups[name][group_name][stratum],
                                      group_row)
            image_count += len(image_names)
            if image_count % 25 == 0 or image_count == len(val_names):
                print('retrieval audit', image_count, flush=True)
    expected = len(val_names)
    if image_count != expected:
        raise RuntimeError(f'Expected {expected} Val images, got {image_count}')
    common.assert_immutable(bundle, code_hash, parent_hash)
    for path, expected_hash in bank_files.items():
        if common.digest(path) != expected_hash:
            raise RuntimeError('Bank changed during audit: ' + path)
    result = dict(
        complete=True, smoke=args.smoke,
        pipeline_smoke=bool(args.limit_images),
        limited_images=args.limit_images,
        test_accessed=False,
        image_count=image_count, class_names=CLASS_NAMES,
        predicted_group_names=GROUP_NAMES,
        cases={case['name']: dict(
            config=case,
            overall=finish_stats(
                totals[case['name']]['all_hard_labelled']),
            purity_ge_095=finish_stats(
                totals[case['name']]['purity_ge_095']),
            by_predicted_group={
                name: finish_stats(row['all_hard_labelled'])
                for name, row in groups[case['name']].items()},
            by_predicted_group_purity_ge_095={
                name: finish_stats(row['purity_ge_095'])
                for name, row in groups[case['name']].items()},
            per_event={
                name: finish_stats(row['all_hard_labelled'])
                for name, row in events[case['name']].items()},
            per_event_purity_ge_095={
                name: finish_stats(row['purity_ge_095'])
                for name, row in events[case['name']].items()})
            for case in CASES},
        frozen_source_verified=True, bank_immutable_verified=True,
        elapsed_seconds=time.monotonic() - begin,
        caution=(
            'These are region-retrieval diagnostics, not full-image mIoU. '
            'GT connected components are not used and no result is a Test score.'))
    common.save_json(out / 'audit.json', result)
    common.save_json(out / 'summary.json', result)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    manifest['complete'] = True
    common.save_json(out / 'manifest.json', manifest)
    print('RETRIEVAL AUDIT COMPLETE. No weights/buffers updated. No Test.',
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
    parser.add_argument('--limit_images', type=int, default=0)
    args = parser.parse_args()
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()

