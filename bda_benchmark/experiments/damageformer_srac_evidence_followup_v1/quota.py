"""Train-only event quota controls. No image inference or query GT is used here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def source_hash():
    checksum = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        checksum.update(path.name.encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def partial_quotas(capacity, total, floor=16):
    """Feasible floor then sqrt(capacity) Hamilton allocation, with capacity caps.

    The floor is reduced uniformly, if necessary, rather than silently exceeding
    the fixed class budget. Ties are broken by event index, not validation scores.
    """
    capacity = list(capacity)
    if any(type(x) is not int or x < 0 for x in capacity):
        raise ValueError('capacities must be nonnegative integers')
    if type(total) is not int or not 0 <= total <= sum(capacity):
        raise ValueError('infeasible class total')
    if type(floor) is not int or floor < 0:
        raise ValueError('floor must be a nonnegative integer')
    actual_floor = floor
    while sum(min(x, actual_floor) for x in capacity) > total:
        actual_floor -= 1
    assigned = [min(x, actual_floor) for x in capacity]
    while sum(assigned) < total:
        active = [i for i, x in enumerate(capacity) if assigned[i] < x]
        remaining = total - sum(assigned)
        denominator = sum(math.sqrt(capacity[i]) for i in active)
        raw = {i: remaining * math.sqrt(capacity[i]) / denominator for i in active}
        increments = {i: min(capacity[i] - assigned[i], math.floor(raw[i])) for i in active}
        for i, value in increments.items():
            assigned[i] += value
        remaining = total - sum(assigned)
        # A saturated event's unallocated share is redistributed next iteration.
        order = sorted(active, key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
        for i in order:
            if remaining and assigned[i] < capacity[i]:
                assigned[i] += 1
                remaining -= 1
    return assigned, actual_floor


def priority(seed, image_name, region_id):
    """A stable row-specific random priority, independent of bank arm/order."""
    text = json.dumps([int(seed), str(image_name), int(region_id)], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:16], 'big')


def ordered_rows(rows, seed):
    """Small-data reference used by tests; production uses bounded heaps."""
    return sorted(rows, key=lambda row: (-priority(seed, row[0], row[1]), row[0], row[1]))


def validate_audit_summary(summary):
    for name in ('complete', 'gates_passed', 'full_validation_evidence'):
        if summary.get(name) is not True:
            raise ValueError('Quota testing requires completed full audit: ' + name)
    if summary.get('smoke') is not False or summary.get('test_accessed') is not False:
        raise ValueError('Quota testing requires non-smoke Train/Val-only audits')
    if summary.get('diagnostic_subset', False):
        raise ValueError('A subset audit cannot authorize formal quota testing')
    if summary.get('image_count') != 450 or summary.get('reproduction', {}).get('passed') is not True:
        raise ValueError('Full Val450 reproduction is required')
    if summary.get('failure') or summary.get('subsequent_failures'):
        raise ValueError('A failed audit cannot authorize formal quota testing')
    if summary.get('input_files_unchanged') is not True:
        raise ValueError('Audit inputs were not certified unchanged')
    if not any(summary.get(key) is True for key in ('weights_buffers_unchanged', 'frozen_evidence_unchanged')):
        raise ValueError('Audit weights/buffers were not certified unchanged')
    if not any(summary.get(key) is True for key in ('numerical_execution_unchanged', 'numerical_unchanged')):
        raise ValueError('Audit numerical policy was not certified unchanged')
    if summary.get('code_unchanged', True) is not True:
        raise ValueError('Audit source code changed')
    if summary.get('pool_guard') is not None and summary['pool_guard'].get('restored') is not True:
        raise ValueError('Audit pooling aliases were not restored')


def evidence_gate(args, bundle=None):
    """An explicit human hypothesis decision, not an automated gain claim."""
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import digest
    if args.smoke:
        return dict(passed=None, smoke=True, reason='Smoke is not a quota hypothesis test')
    if not args.confirm_quota_test or not str(args.evidence_note or '').strip():
        raise ValueError('Formal quota test requires --confirm_quota_test and --evidence_note')
    records = {}
    for kind in ('feature', 'memory'):
        value = getattr(args, kind + '_audit', None)
        if not value:
            raise ValueError('Missing --' + kind + '_audit')
        root = Path(value)
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        validate_audit_summary(summary)
        records[kind] = dict(root=str(root.resolve()),
                             summary_sha256=digest(root / 'summary.json'),
                             manifest_sha256=digest(root / 'manifest.json'))
        if bundle is not None:
            h = bundle['hashes']
            if kind == 'feature':
                for key, expected in (('checkpoint_sha256', h['front_sha256']),
                                      ('a1_sha256', h['a1_sha256']),
                                      ('train_sha256', digest(args.train_list)),
                                      ('val_sha256', digest(args.val_list))):
                    if manifest.get(key) != expected:
                        raise ValueError('Feature audit belongs to different evidence: ' + key)
            else:
                if manifest.get('hashes') != h or manifest.get('head_hashes') != bundle['head_hashes']:
                    raise ValueError('Memory audit does not match the source J1/J2/J3 artifacts')
                if manifest.get('reference_hashes') != bundle['reference_hashes']:
                    raise ValueError('Memory audit final reference differs')
                if manifest.get('train_sha256') != digest(args.train_list) or manifest.get('val_sha256') != digest(args.val_list):
                    raise ValueError('Memory audit Train/Val split differs')
    return dict(passed=True, smoke=False, records=records, evidence_note=args.evidence_note,
                warning='Passed gates do not prove quota allocation causes the performance limitation')


def add_arguments(parser):
    for key in ('joint_run', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        parser.add_argument('--' + key, required=True)
    for key in ('checkpoint', 'cache', 'metric_checkpoint', 'feature_audit', 'memory_audit', 'evidence_note'):
        parser.add_argument('--' + key)
    parser.add_argument('--confirm_quota_test', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--smoke', action='store_true')
    return parser
