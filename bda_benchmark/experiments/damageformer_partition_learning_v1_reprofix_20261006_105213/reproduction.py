"""Auditable baseline comparison calibrated only from immutable source history.

This module deliberately has no torch dependency and never writes source files.
Current evaluation values cannot influence the policy built before evaluation.
"""
import hashlib
import json
import math
from pathlib import Path


DEFAULT_TOL = 1e-5
MAX_TOL = 5e-5
MIN_HISTORY = 3
BASELINES = (('old_pixel', 'old_pixel'), ('old_region', 'old_region'),
             ('source_hard', 'hard'))
FIELDS = ('valid_miou', 'iou', 'precision', 'recall')
CLASSES = ('Background', 'Intact', 'Damaged', 'Destroyed')
FORMAT = 'PARTITION_BASELINE_REPRODUCTION_V1'
_MISSING = object()


def _safe(value):
    """Preserve invalid evidence without producing invalid JSON numbers."""
    if value is _MISSING:
        return '<missing>'
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return '<unsupported ' + type(value).__name__ + '>'


def json_safe_evidence(value):
    """Keep nonfinite evidence as explicit strings, never as repaired metrics."""
    return _safe(value)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (ValueError, OverflowError):
        return False


def _cell_error(value):
    if value is _MISSING:
        return 'missing value'
    if value is None:
        return None
    if not _number(value):
        return 'expected a finite number or absent-class None'
    if not 0 <= value <= 1:
        return 'metric outside [0, 1]'
    return None


def _shape_error(value, field):
    if value is _MISSING:
        return 'missing metric'
    if field == 'valid_miou':
        return 'expected a scalar metric' if isinstance(value, (list, tuple, dict)) else None
    if not isinstance(value, list) or len(value) != 4:
        return 'expected a four-class list'
    return None


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _metadata_error(value, field):
    if field == 'count':
        return None if _integer(value) else 'expected a nonnegative integer count'
    if not isinstance(value, list) or len(value) != 4 or not all(_integer(x) for x in value):
        return 'expected four nonnegative integer GT supports'
    return None


def _record_error(record):
    if not isinstance(record, dict):
        return 'metrics must be an object'
    for field in FIELDS:
        value = record.get(field, _MISSING)
        error = _shape_error(value, field)
        if error:
            return field + ': ' + error
        for item in ([value] if field == 'valid_miou' else value):
            error = _cell_error(item)
            if error:
                return field + ': ' + error
    for field in ('count', 'support'):
        if field in record:
            error = _metadata_error(record[field], field)
            if error:
                return field + ': ' + error
    if 'count' in record and 'support' in record and sum(record['support']) != record['count']:
        return 'count does not equal the sum of GT support'
    return None


def _history_error(rows, final_step=None):
    if len(rows) < MIN_HISTORY:
        return 'need at least ' + str(MIN_HISTORY) + ' complete source validation records'
    previous = 0
    for position, row in enumerate(rows, 1):
        prefix = 'history record ' + str(position) + ': '
        if not isinstance(row, dict):
            return prefix + 'expected an object'
        step = row.get('step')
        if not isinstance(step, int) or isinstance(step, bool) or step <= previous:
            return prefix + 'steps must be positive, unique and increasing'
        if final_step is not None and step > final_step:
            return prefix + 'step exceeds the source final step'
        previous = step
        if row.get('image_count') != 450 or type(row.get('image_count')) is not int:
            return prefix + 'expected image_count=450'
        if row.get('test_accessed') is not False:
            return prefix + 'expected test_accessed=false'
        metrics = row.get('metrics')
        if not isinstance(metrics, dict):
            return prefix + 'missing metrics object'
        for _, source in BASELINES:
            error = _record_error(metrics.get(source))
            if error:
                return prefix + source + ': ' + error
    return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite JSON number: ' + value)


def load_history(run_root, final_step):
    """Return (validated rows, warnings, SHA256); any defect disables calibration."""
    warnings = []
    if not isinstance(final_step, int) or isinstance(final_step, bool) or final_step <= 0:
        return [], ['Source final_step is invalid; retaining strict default tolerances.'], None
    path = Path(run_root) / 'validation.jsonl'
    try:
        content = path.read_bytes()
    except (OSError, ValueError) as exc:
        return [], ['Source history unavailable (' + str(exc) +
                    '); retaining strict default tolerances.'], None
    sha = hashlib.sha256(content).hexdigest()
    try:
        rows = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                for line in content.decode('utf-8-sig').splitlines() if line.strip()]
        error = _history_error(rows, final_step)
        if error:
            raise ValueError(error)
    except (UnicodeError, ValueError, TypeError, OverflowError) as exc:
        warnings.append('Source history rejected (' + str(exc) +
                        '); retaining strict default tolerances.')
        return [], warnings, sha
    return rows, warnings, sha


def _key(baseline, field, index):
    return '/'.join((baseline, field) + (() if index is None else (str(index),)))


def _slots():
    for field in FIELDS:
        for index in ([None] if field == 'valid_miou' else range(4)):
            yield field, index


def _value(record, field, index):
    value = record[field]
    return value if index is None else value[index]


def build_policy(expected, history_rows=(), warnings=()):
    """Freeze per-cell tolerances before seeing current evaluation results.

    Only old_region can be calibrated. A valid historical deviation above the
    cap is an anomaly that blocks acceptance, even if a later evaluation passes.
    Missing, incomplete or malformed history instead retains strict defaults.
    """
    notices = [str(item) for item in warnings]
    try:
        rows = list(history_rows)
        error = _history_error(rows)
    except (TypeError, ValueError, OverflowError) as exc:
        rows, error = [], 'invalid history sequence: ' + str(exc)
    source = expected if isinstance(expected, dict) else {}
    region = source.get('old_region')
    if not error:
        error = _record_error(region)
        if error:
            error = 'invalid source old_region: ' + error
    if not error:
        for row in rows:
            history = row['metrics']['old_region']
            for field, index in _slots():
                if (_value(history, field, index) is None) != (_value(region, field, index) is None):
                    error = 'source history changes absent-class None semantics'
                    break
            if error:
                break
            for _, name in BASELINES:
                reference = source.get(name, {})
                reference = reference if isinstance(reference, dict) else {}
                for field in ('count', 'support'):
                    historical = row['metrics'][name]
                    if field in reference or field in historical:
                        if reference.get(field, _MISSING) != historical.get(field, _MISSING):
                            error = 'source history changes ' + name + '/' + field
                            break
                if error:
                    break
            if error:
                break
    if error:
        notices.append('History calibration disabled (' + error +
                       '); retaining strict default tolerances.')
    usable = error is None
    entries = {}
    anomalies = []
    for baseline, _ in BASELINES:
        for field, index in _slots():
            key = _key(baseline, field, index)
            history = []
            if usable and baseline == 'old_region':
                reference = _value(region, field, index)
                history = [dict(step=row['step'], value=_value(row['metrics']['old_region'], field, index),
                                delta=(None if reference is None else
                                       _value(row['metrics']['old_region'], field, index)-reference))
                           for row in rows]
            maximum = max((abs(row['delta']) for row in history if row['delta'] is not None), default=0.)
            calibrated = bool(history) and maximum > DEFAULT_TOL
            tolerance = min(MAX_TOL, maximum+DEFAULT_TOL) if calibrated else DEFAULT_TOL
            anomaly = bool(history) and maximum > MAX_TOL
            if anomaly:
                anomalies.append(key)
                notices.append('Source history exceeds the hard cap for ' + key +
                               '; baseline acceptance is blocked.')
            entries[key] = dict(tolerance=tolerance, tolerance_pp=100*tolerance,
                                calibrated=calibrated, history_count=len(history),
                                history_max_abs=maximum if history else None,
                                history_max_abs_pp=100*maximum if history else None,
                                history_values=history, history_anomaly=anomaly)
    return _safe(dict(format=FORMAT, default_tolerance=DEFAULT_TOL, max_tolerance=MAX_TOL,
                      calibration_baseline='old_region', minimum_history_records=MIN_HISTORY,
                      history_count=len(rows), history_usable=usable,
                      history_steps=[row['step'] for row in rows] if usable else [],
                      history_anomalies=anomalies, entries=entries, warnings=notices))


def _entry(policy, baseline, field, index):
    entries = policy.get('entries', {}) if isinstance(policy, dict) else {}
    entry = entries.get(_key(baseline, field, index)) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return {}, DEFAULT_TOL, 'missing or invalid policy entry'
    tolerance = entry.get('tolerance')
    if not _number(tolerance) or not DEFAULT_TOL <= tolerance <= MAX_TOL:
        return entry, DEFAULT_TOL, 'policy tolerance outside permitted bounds'
    if baseline != 'old_region' and tolerance != DEFAULT_TOL:
        return entry, DEFAULT_TOL, 'this baseline must retain the strict default tolerance'
    if entry.get('history_anomaly') is True:
        return entry, tolerance, 'source historical deviation exceeds the hard cap'
    return entry, tolerance, None


def compare_baselines(actual, expected, policy):
    """Collect every discrepancy without raising; return an allow_nan=False report."""
    observed = actual if isinstance(actual, dict) else {}
    reference = expected if isinstance(expected, dict) else {}
    checks = []
    notices = list(policy.get('warnings', [])) if isinstance(policy, dict) else ['Invalid policy.']
    for baseline, source in BASELINES:
        a = observed.get(baseline)
        b = reference.get(source)
        a = a if isinstance(a, dict) else {}
        b = b if isinstance(b, dict) else {}
        for field in FIELDS:
            av, bv = a.get(field, _MISSING), b.get(field, _MISSING)
            shape_errors = [('actual: ' + error) for error in [_shape_error(av, field)] if error]
            shape_errors += [('expected: ' + error) for error in [_shape_error(bv, field)] if error]
            indices = [None] if field == 'valid_miou' else range(4)
            for index in indices:
                entry, tolerance, policy_error = _entry(policy, baseline, field, index)
                x = av if index is None or _shape_error(av, field) else av[index]
                y = bv if index is None or _shape_error(bv, field) else bv[index]
                reasons = list(shape_errors)
                if policy_error:
                    reasons.append(policy_error)
                if not shape_errors:
                    for side, value in (('actual', x), ('expected', y)):
                        error = _cell_error(value)
                        if error:
                            reasons.append(side + ': ' + error)
                    if (x is None) != (y is None):
                        reasons.append('absent-class None mismatch')
                delta = x-y if not shape_errors and _number(x) and _number(y) else None
                if delta is not None and abs(delta) > tolerance:
                    reasons.append('absolute difference exceeds tolerance')
                checks.append(dict(baseline=baseline, expected_baseline=source,
                                   metric=field, class_index=index,
                                   class_name=None if index is None else CLASSES[index],
                                   actual=x, expected=y, delta=delta,
                                   delta_pp=None if delta is None else 100*delta,
                                   tolerance=tolerance, tolerance_pp=100*tolerance,
                                   passed=not reasons, reason='; '.join(reasons) or 'within tolerance',
                                   history=entry))
        for field in ('count', 'support'):
            if field not in a and field not in b:
                continue
            av, bv = a.get(field, _MISSING), b.get(field, _MISSING)
            reasons = []
            for side, value in (('actual', av), ('expected', bv)):
                error = _metadata_error(value, field)
                if error:
                    reasons.append(side + ': ' + error)
            if not reasons and av != bv:
                reasons.append('GT count/support must agree exactly')
            for side, record in (('actual', a), ('expected', b)):
                if field == 'count' and 'support' in record and not _metadata_error(record['support'], 'support'):
                    if not _metadata_error(record.get('count'), 'count') and sum(record['support']) != record['count']:
                        reasons.append(side + ': count does not equal the sum of GT support')
            checks.append(dict(baseline=baseline, expected_baseline=source,
                               metric=field, class_index=None, class_name=None,
                               actual=av, expected=bv, delta=None, delta_pp=None,
                               tolerance=0., tolerance_pp=0., passed=not reasons,
                               reason='; '.join(reasons) or 'GT metadata matches', history={}))
    return _safe(dict(format=FORMAT, passed=all(row['passed'] for row in checks),
                      checks=checks, warnings=notices, policy=policy,
                      actual=actual, expected=expected))


def enforce(report):
    """Raise only after the caller has had an opportunity to save the full report."""
    if not isinstance(report, dict) or report.get('passed') is not True:
        checks = report.get('checks', []) if isinstance(report, dict) else []
        failed = [row for row in checks if isinstance(row, dict) and row.get('passed') is not True]
        labels = [str(row.get('baseline')) + '/' + str(row.get('metric')) +
                  ('' if row.get('class_index') is None else '/' + str(row['class_index']))
                  for row in failed]
        raise ValueError('Baseline reproduction failed (' + str(len(failed)) + ' checks): ' +
                         ', '.join(labels))
