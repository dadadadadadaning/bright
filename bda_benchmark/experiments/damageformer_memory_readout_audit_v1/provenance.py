"""Read-only loading and fixed reproduction gates for completed Balanced V2 runs."""
import hashlib
import json
import math
import pickle
from pathlib import Path, PurePosixPath

import torch

from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import common
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import (
    BANK_FORMAT, HEAD_FORMAT, device_bank, read_banks,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import canonical
from bda_benchmark.experiments.damageformer_region_memory_v2.core import ResidualSupportHead
from bda_benchmark.experiments.damageformer_region_memory_v2.common import require_hard
from bda_benchmark.experiments.damageformer_partition_learning_v1.reproduction import json_safe_evidence


TOLERANCE = 1e-5
HEADS = {'nomemory': 'nomemory', 'normal': 'balanced_hard'}
BANK_FILES = ('bank_all.pt', 'bank_hard.pt', 'manifest.json')
FLOAT_FIELDS = ('valid_miou', 'present_foreground_miou', 'iou', 'precision', 'recall', 'f1')
VECTOR_FIELDS = ('iou', 'precision', 'recall', 'f1')
EXACT_FIELDS = ('count', 'support')
DIAGNOSTIC_FIELDS = ('confusion', 'background_to_building', 'building_to_background')


def _positive_integer(value):
    return type(value) is int and value > 0


def _nonnegative_integer(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'),
                       object_pairs_hook=_object, parse_constant=_invalid_constant)
    _require(isinstance(value, dict), 'Expected JSON object: ' + str(path))
    return value


def _metric_error(record):
    if not isinstance(record, dict):
        return 'missing metrics object'
    for field in FLOAT_FIELDS:
        if field not in record:
            return 'missing metric ' + field
        value = record[field]
        if field in VECTOR_FIELDS:
            if not isinstance(value, list) or len(value) != 4:
                return 'expected four-class metric ' + field
            values = value
        else:
            values = [value]
        if any(item is not None and not _number(item) for item in values):
            return 'invalid finite [0,1]/None metric ' + field
    count, support, confusion = (record.get(key) for key in ('count', 'support', 'confusion'))
    if not _nonnegative_integer(count):
        return 'invalid GT count'
    if (not isinstance(support, list) or len(support) != 4
            or not all(_nonnegative_integer(x) for x in support) or sum(support) != count):
        return 'invalid GT support/count'
    if (not isinstance(confusion, list) or len(confusion) != 4
            or any(not isinstance(row, list) or len(row) != 4
                   or not all(_nonnegative_integer(x) for x in row) for row in confusion)
            or [sum(row) for row in confusion] != support):
        return 'invalid confusion/GT support'
    for field, expected in (
            ('background_to_building', sum(confusion[0][1:])),
            ('building_to_background', sum(row[0] for row in confusion[1:]))):
        if not _nonnegative_integer(record.get(field)) or record[field] != expected:
            return 'invalid confusion/error count ' + field
    return None


def _bank_hashes(record):
    _require(isinstance(record, dict) and len(record) == len(BANK_FILES),
             'Require exactly the two Bank files and Bank manifest hashes')
    result = {}
    for name, value in record.items():
        _require(isinstance(name, str), 'Invalid Bank provenance path')
        basename = PurePosixPath(name.replace('\\', '/')).name
        _require(basename in BANK_FILES and basename not in result,
                 'Unexpected or ambiguous Bank provenance filename: ' + basename)
        _require(isinstance(value, str) and len(value) == 64
                 and all(character in '0123456789abcdef' for character in value),
                 'Invalid Bank SHA256: ' + basename)
        result[basename] = value
    return result


def _bank_state(bank):
    """Include loaded tensors and derived lookups, not just immutable disk files."""
    result = hashlib.sha256()
    for name, value in sorted(bank.items()):
        result.update(name.encode())
        if torch.is_tensor(value):
            result.update(str((value.dtype, tuple(value.shape))).encode())
            result.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            result.update(canonical(value).encode())
    return result.hexdigest()


def _load_trained(args, bundle):
    require_hard(args)
    root, bank_root = Path(args.memory_run), Path(args.bank)
    paths = [root / 'manifest.json', root / 'summary.json', root / 'validation.jsonl']
    paths += [root / (name + '_last.pth') for name in HEADS.values()]
    paths += [bank_root / name for name in BANK_FILES]
    files = {str(path.resolve()): common.digest(path) for path in paths}
    manifest, summary = _json(root / 'manifest.json'), _json(root / 'summary.json')
    _require(manifest.get('format') == HEAD_FORMAT and manifest.get('complete') is True,
             'Require a completed ROI_MEMORY_BALANCED_HEAD_V2 training manifest')
    _require(summary.get('complete') is True and summary.get('phase') == 'complete',
             'Require a completed Balanced V2 summary')
    for record in (manifest, summary):
        _require(record.get('test_accessed') is False, 'Require a Train/Val-only completed run')
        _require(record.get('smoke') is False,
                 'Readout audit requires formal trained heads, including during audit smoke')
    arguments = manifest.get('arguments')
    _require(isinstance(arguments, dict), 'Missing original training arguments')
    for key in ('source_head', 'pool_mode', 'read_mode'):
        _require(arguments.get(key) == 'hard', 'Training must fix hard source/pool/read: ' + key)
    _require(arguments.get('smoke') is False, 'Readout audit requires formal training arguments')
    final_step = summary.get('final_step')
    _require(_positive_integer(final_step) and type(summary.get('optimizer_steps')) is int
             and type(arguments.get('steps')) is int
             and final_step == summary['optimizer_steps'] == arguments['steps'],
             'Require same final_step, optimizer_steps and original training budget')
    settings = {name: arguments.get(name) for name in
                ('seed', 'topk', 'query_chunk', 'allow_same_event')}
    _require(_nonnegative_integer(settings['seed']), 'Invalid original training seed')
    _require(_positive_integer(settings['topk']) and settings['topk'] % 4 == 0,
             'Original balanced topk must be a positive multiple of four')
    _require(_positive_integer(settings['query_chunk']), 'Invalid original query_chunk')
    _require(type(settings['allow_same_event']) is bool, 'Missing explicit original event policy')
    for key in ('frozen_front', 'frozen_building', 'frozen_q', 'frozen_embedding',
                'frozen_source_classifier'):
        _require(manifest.get(key) is True, 'Missing frozen-source training contract: ' + key)
    for key in ('frozen_source_verified', 'bank_immutable_verified', 'v2_code_immutable_verified'):
        _require(summary.get(key) is True, 'Missing completed training integrity check: ' + key)
    _require(summary.get('baseline_reproduction_pass') is True,
             'Formal training did not pass its baseline reproduction gate')
    identity = common.source_identity(bundle, args)
    code_hash = common.experiment_hash()
    _require(canonical(manifest.get('identity')) == canonical(identity),
             'Trained run source identity differs from the loaded frozen source')
    _require(manifest.get('code_hash') == code_hash, 'Balanced V2 training code hash mismatch')
    _require(manifest.get('bank_format') == BANK_FORMAT, 'Unexpected trained Bank format')
    expected_bank_files = _bank_hashes(manifest.get('bank_file_sha256'))
    for name, expected in expected_bank_files.items():
        _require(files[str((bank_root / name).resolve())] == expected,
                 'Training Bank provenance mismatch: ' + name)
    banks_cpu, bank_manifest = read_banks(
        bank_root, identity, False, common.digest, bundle['train'])
    _require(bank_manifest.get('source_hash') == code_hash,
             'Bank extraction code differs from the completed Balanced V2 training code')
    for policy, bank in banks_cpu.items():
        _require(len(bundle['train']) == 2918 and len(set(bundle['train'])) == 2918
                 and set(bank['image_names']) == set(bundle['train']),
                 'Formal dual Banks must cover complete unique Train2918: ' + policy)
        _require(manifest.get(policy + '_bank_regions') == len(bank['tokens']),
                 'Training manifest Bank region count mismatch: ' + policy)
        _require(bank['per_bucket'] == bank_manifest['per_bucket'],
                 'Dual Bank per-bucket capacities disagree')
    _require(manifest.get('bank_limit_per_policy') == bank_manifest.get('bank_limit_per_policy'),
             'Training/extraction Bank capacity mismatch')
    rows = [json.loads(line, object_pairs_hook=_object, parse_constant=_invalid_constant)
            for line in (root / 'validation.jsonl').read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    _require(rows and all(isinstance(row, dict) for row in rows), 'Invalid training validation history')
    steps = [row.get('step') for row in rows]
    _require(all(_positive_integer(step) for step in steps)
             and steps == sorted(set(steps)) and steps[-1] == final_step,
             'Training validation steps must increase uniquely through the final step')
    final_row = rows[-1]
    _require(type(final_row.get('image_count')) is int
             and final_row['image_count'] == 450
             and final_row.get('test_accessed') is False,
             'Final training validation must cover the expected Val images only')
    final_metrics = summary.get('final_metrics')
    _require(isinstance(final_metrics, dict) and isinstance(final_row.get('metrics'), dict),
             'Missing final training metrics')
    heads = {}
    for name, original_name in HEADS.items():
        expected_metrics = final_metrics.get(original_name)
        error = _metric_error(expected_metrics)
        _require(error is None, 'Malformed final metrics for ' + original_name + ': ' + str(error))
        _require(canonical(final_row['metrics'].get(original_name)) == canonical(expected_metrics),
                 'Final validation history/summary metrics disagree: ' + original_name)
        checkpoint = torch.load(root / (original_name + '_last.pth'),
                                map_location='cpu', weights_only=False)
        _require(isinstance(checkpoint, dict) and checkpoint.get('format') == HEAD_FORMAT,
                 'Invalid trained correction checkpoint format: ' + original_name)
        _require(checkpoint.get('name') == original_name
                 and type(checkpoint.get('step')) is int and checkpoint['step'] == final_step,
                 'Correction head name or same-final-step mismatch: ' + original_name)
        _require(canonical(checkpoint.get('source_identity')) == canonical(identity),
                 'Correction checkpoint source identity mismatch: ' + original_name)
        _require(checkpoint.get('code_hash') == code_hash,
                 'Correction checkpoint code hash mismatch: ' + original_name)
        _require(canonical(checkpoint.get('bank_file_sha256'))
                 == canonical(manifest['bank_file_sha256']),
                 'Correction checkpoint Bank provenance mismatch: ' + original_name)
        _require(canonical(checkpoint.get('metrics')) == canonical(expected_metrics),
                 'Correction checkpoint final metrics mismatch: ' + original_name)
        weights = checkpoint.get('head')
        _require(isinstance(weights, dict) and weights
                 and all(torch.is_tensor(value) and bool(torch.isfinite(value).all())
                         for value in weights.values()),
                 'Invalid/nonfinite correction head state: ' + original_name)
        head = ResidualSupportHead(identity['dim'])
        template = head.state_dict()
        _require(set(weights) == set(template)
                 and all(value.shape == template[key].shape and value.dtype == template[key].dtype
                         for key, value in weights.items()),
                 'Correction head tensor keys/shapes/dtypes mismatch: ' + original_name)
        head.load_state_dict(weights, strict=True)
        heads[name] = head.to(bundle['device']).requires_grad_(False).eval()
    banks = {name: device_bank(bank, bundle['device']) for name, bank in banks_cpu.items()}
    state = dict(heads=heads, banks=banks, manifest=manifest, summary=summary,
                 files=files, states={name: common.state_digest(head) for name, head in heads.items()},
                 settings=settings, code_hash=code_hash, identity=identity,
                 bank_manifest=bank_manifest,
                 bank_states={name: _bank_state(bank) for name, bank in banks.items()})
    assert_trained_unchanged(state, bundle)
    return state


def load_trained(args, bundle):
    """Load formal final heads and full Train Banks, also when audit smoke limits queries.

    Return heads/banks/manifest/summary/files/states/settings. Audit smoke is
    deliberately unrelated to training/Bank smoke and never permits smoke sources.
    """
    try:
        return _load_trained(args, bundle)
    except (OSError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError('Cannot load immutable Balanced V2 training result: ' + str(exc)) from exc


def assert_trained_unchanged(state, bundle):
    """Check old source/code, all tracked files, frozen trained heads and loaded Bank tensors."""
    try:
        common.assert_v2_immutable(bundle, state['code_hash'])
        for name, head in state['heads'].items():
            _require(all(not module.training for module in head.modules())
                     and all(not value.requires_grad and value.grad is None for value in head.parameters()),
                     'Trained correction head is not frozen: ' + name)
            _require(common.state_digest(head) == state['states'][name],
                     'Trained correction head state changed: ' + name)
        for name, bank in state['banks'].items():
            _require(_bank_state(bank) == state['bank_states'][name],
                     'Loaded Bank tensors or metadata changed: ' + name)
        for path, expected in state['files'].items():
            _require(common.digest(path) == expected, 'Trained source file changed: ' + path)
    except (OSError, KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('Trained-source integrity check failed: ' + str(exc)) from exc


def verify_trained_metrics(actual, state, smoke):
    """Compare final normal/nomemory without fitting tolerances to the new evaluation.

    All floating metrics use absolute 1e-5. GT count/support are exact. Confusion
    and error counts are recorded diagnostically: imposing exact predictions
    there would silently negate the explicitly allowed floating metric tolerance.
    """
    if smoke:
        return dict(passed=None, status='smoke_not_full_val', tolerance=TOLERANCE,
                    checks=[], diagnostic_differences=[])
    actual = actual if isinstance(actual, dict) else {}
    expected = state.get('summary', {}).get('final_metrics', {})
    checks, diagnostics = [], []
    for name, original_name in HEADS.items():
        observed, reference = actual.get(name), expected.get(original_name)
        error_a, error_b = _metric_error(observed), _metric_error(reference)
        checks.append(dict(name=name, expected_name=original_name, metric='record_contract',
                           passed=error_a is None and error_b is None,
                           reason='; '.join(error for error in
                                            (('actual: ' + error_a) if error_a else None,
                                             ('expected: ' + error_b) if error_b else None) if error)
                                  or 'complete finite metric records'))
        if error_a or error_b:
            continue
        for field in FLOAT_FIELDS:
            first = observed[field] if field in VECTOR_FIELDS else [observed[field]]
            second = reference[field] if field in VECTOR_FIELDS else [reference[field]]
            for index, (value, target) in enumerate(zip(first, second)):
                delta = None if value is None or target is None else value - target
                passed = (value is None and target is None) if delta is None else abs(delta) <= TOLERANCE
                checks.append(dict(name=name, expected_name=original_name, metric=field,
                                   class_index=index if field in VECTOR_FIELDS else None,
                                   actual=value, expected=target, delta=delta,
                                   tolerance=TOLERANCE, passed=passed,
                                   reason='within fixed tolerance' if passed else
                                          'absent-class mismatch or fixed tolerance exceeded'))
        for field in EXACT_FIELDS:
            checks.append(dict(name=name, expected_name=original_name, metric=field,
                               actual=observed[field], expected=reference[field], tolerance=0.,
                               passed=observed[field] == reference[field], reason='GT metadata must agree exactly'))
        for field in DIAGNOSTIC_FIELDS:
            diagnostics.append(dict(name=name, metric=field, actual=observed[field],
                                    expected=reference[field], equal=observed[field] == reference[field],
                                    gate=False))
    passed = all(check['passed'] for check in checks)
    return json_safe_evidence(dict(passed=passed, status='passed' if passed else 'failed',
                                   tolerance=TOLERANCE, checks=checks,
                                   diagnostic_differences=diagnostics))
