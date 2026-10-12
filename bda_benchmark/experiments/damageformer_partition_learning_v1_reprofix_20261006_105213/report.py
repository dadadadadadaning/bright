"""Read-only partition-learning report with explicit completion and class tradeoffs."""
import argparse
import json
import math
from pathlib import Path


GROUPS = {
    'structure': ('fixed_q', 'learn_q', 'learn_q_structure'),
    'multimodal': ('rgb_rgb', 'rgb_sar'),
    'support': ('hard', 'uniform', 'sar'),
}
PAIRS = {
    'structure': ('fixed_q__learn_q', 'learn_q__learn_q_structure'),
    'multimodal': ('rgb_rgb__rgb_sar',),
    'support': ('hard__uniform', 'uniform__sar', 'hard__sar'),
}
LABELS = ('Background', 'Intact', 'Damaged', 'Destroyed')


def _reject_constant(value):
    raise ValueError('non-finite JSON number: ' + value)


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant)


def read_json(path, warnings):
    try:
        value = _loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (OSError, ValueError) as exc:
        warnings.append(f'{Path(path).name} unavailable: {exc}')
        return {}


def read_rows(path, warnings):
    try:
        text = Path(path).read_text(encoding='utf-8-sig')
    except OSError as exc:
        warnings.append(f'{Path(path).name} unavailable: {exc}')
        return []
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _loads(line)
            if (not isinstance(row, dict) or type(row.get('step')) is not int
                    or row['step'] < 0 or not isinstance(row.get('metrics'), dict)):
                raise ValueError('expected step and metrics')
            rows.append(row)
        except ValueError as exc:
            warnings.append(f'{Path(path).name}: skipped incomplete/invalid line {number}: {exc}')
    return rows


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percent(value):
    return f'{100 * value:.4f}%' if numeric(value) else 'N/A'


def _class_value(metric, key, index):
    values = metric.get(key)
    return values[index] if isinstance(values, (list, tuple)) and len(values) > index else None


def metric_line(name, metric):
    if not isinstance(metric, dict):
        return f'{name}: metrics unavailable'
    ious = ', '.join(f'{LABELS[i]}={percent(_class_value(metric, "iou", i))}' for i in (1, 2, 3))
    damaged = ', '.join(f'{label}={percent(_class_value(metric, key, 2))}'
                        for label, key in (('P', 'precision'), ('R', 'recall'), ('F1', 'f1')))
    return f'{name}: FG mIoU={percent(metric.get("valid_miou"))}; IoU[{ious}]; Damaged {damaged}'


def same_step_gains(metrics, pairs):
    gains = {}
    for pair in pairs:
        first, second = pair.split('__')
        a, b = metrics.get(first, {}), metrics.get(second, {})
        av = a.get('valid_miou') if isinstance(a, dict) else None
        bv = b.get('valid_miou') if isinstance(b, dict) else None
        gains[pair] = 100 * (bv - av) if numeric(av) and numeric(bv) else None
    return gains


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _ratio(numerator, denominator):
    if not numeric(numerator) or not numeric(denominator) or denominator <= 0:
        return None
    return numerator / denominator


def _number(value):
    return f'{value:.4f}' if numeric(value) else 'N/A'


def partition_quality_lines(totals, names, stage='read'):
    """Normalize accumulated native-area totals, never average image ratios."""
    label = 'Unsplit base Q' if stage == 'base' else 'Read-route after predicted B split'
    if not isinstance(totals, dict) or not totals:
        return [f'{label} partition quality unavailable at this validation step.']
    lines = [f'{label} partition quality (GT evaluation only; native fractional area, NOT classifier mIoU):',
             'Class/foreground/background denominators are separate. Partition quality does not measure exchanged pooling content.']
    for name in names:
        values = totals.get(name)
        if not isinstance(values, dict):
            lines.append(f'  {name}: unavailable; missing evidence is not zero.')
            continue
        lines.append(f'  {name}: images={values.get("image_count", "N/A")}')
        for c, label in enumerate(LABELS):
            mass, same = values.get(f'class_{c}_mass'), values.get(f'class_{c}_self_sum')
            lines.append(f'    {label}: same-class={percent(_ratio(same, mass))}; '
                         f'self_sum={_number(same)}; native_mass={_number(mass)}')
        fg_mass = values.get('foreground_mass')
        fg_sums = [values.get(f'class_{c}_self_sum') for c in (1, 2, 3)]
        fg_same = sum(fg_sums) if all(numeric(value) for value in fg_sums) else None
        fg_bg = values.get('foreground_receives_background_sum')
        bg_fg = values.get('background_receives_foreground_sum')
        bg_mass = values.get('class_0_mass')
        lines.append(f'    Foreground (all grades): same-grade={percent(_ratio(fg_same, fg_mass))}; '
                     f'receives-background={percent(_ratio(fg_bg, fg_mass))}; '
                     f'background_sum={_number(fg_bg)}; native_mass={_number(fg_mass)}')
        lines.append(f'    Background only: receives-foreground={percent(_ratio(bg_fg, bg_mass))}; '
                     f'foreground_sum={_number(bg_fg)}; native_mass={_number(bg_mass)}')
        majority, area = values.get('native_majority_mass'), values.get('native_valid_area')
        lines.append(f'    Secondary global majority purity (background-weighted): {percent(_ratio(majority, area))}; '
                     f'majority_mass={_number(majority)}; valid_area={_number(area)}')
        used, fragmented = values.get('hard_used_regions'), values.get('hard_fragmented_regions')
        components = values.get('hard_connected_components')
        lines.append(f'    Argmax routing, native 8-connectivity: fragmented={percent(_ratio(fragmented, used))}; '
                     f'fragmented_regions={_number(fragmented)}; used_regions={_number(used)}; '
                     f'components_per_used_region={_number(_ratio(components, used))}')
    return lines


def _audit_lines(audit, pairs):
    lines = [f'Completed endpoint audit: image_count={audit.get("image_count", "N/A")}',
             'Damaged transitions both condition on GT=Damaged: damaged_to_intact_harm means correct Damaged -> wrong Intact; '
             'intact_to_damaged_rescue means wrong Intact -> correct Damaged.']
    for group, bucket in audit.get('groups', {}).items():
        if not isinstance(bucket, dict):
            continue
        comparisons = bucket.get('comparisons', {})
        for pair in pairs:
            result = comparisons.get(pair)
            if not isinstance(result, dict) or result.get('count') == 0:
                continue
            fields = ('count', 'base_correct', 'base_wrong', 'harmed', 'rescued', 'net',
                      'harm_rate', 'recovery_rate', 'damaged_to_intact_harm',
                      'intact_to_damaged_rescue')
            lines.append(f'  {group} {pair}: ' + _json({key: result.get(key) for key in fields}))
    for event, bucket in audit.get('per_event', {}).items():
        if isinstance(bucket, dict):
            lines.append(f'  event={event}: gains_pp=' + _json(same_step_gains(bucket.get('metrics', {}), pairs)))
    lines.append('Subset harm/recovery rates retain their own denominators; subset mIoUs are not additive.')
    return lines


def reproduction_lines(reproduction):
    """Render persisted gate evidence without recalibrating or rerunning checks."""
    if not reproduction:
        return ['Reproduction evidence unavailable; no pass inferred from missing reproduction.json.']
    lines = ['reproduction.json: ' + _json({key: reproduction.get(key) for key in
             ('passed', 'status', 'metric_checks_passed', 'source_integrity_pass')}),
             'Reproduction policy: ' + _json(reproduction.get('policy'))]
    checks = reproduction.get('checks', [])
    if not isinstance(checks, list):
        return lines + ['Reproduction checks unavailable: expected a list of flat checks.']
    fields = ('baseline', 'expected_baseline', 'metric', 'class_index', 'actual', 'expected',
              'delta_pp', 'tolerance_pp', 'tolerance', 'passed', 'reason', 'history')
    failed = [row for row in checks if isinstance(row, dict) and row.get('passed') is not True]
    historical = [row for row in checks if isinstance(row, dict)
                  and isinstance(row.get('history'), dict)
                  and row['history'].get('calibrated') is True]
    for title, selected in (('Failed/unverified reproduction checks', failed),
                            ('Checks using expanded historical tolerance', historical)):
        lines.append(f'{title}: {len(selected)}')
        for row in selected:
            data = {key: row.get(key) for key in fields}
            index = row.get('class_index')
            if type(index) is int and 0 <= index < len(LABELS):
                data['class_name'] = LABELS[index]
            lines.append('  ' + _json(data))
    for warning in reproduction.get('warnings', []):
        lines.append('REPRODUCTION WARNING: ' + str(warning))
    if reproduction.get('error'):
        lines.append('Reproduction failure reason: ' + str(reproduction['error']))
    return lines


def render(root):
    root = Path(root)
    warnings = []
    manifest = read_json(root / 'manifest.json', warnings)
    summary = read_json(root / 'summary.json', warnings)
    gate_required = manifest.get('reproduction_gate_version') == 'history_bounded_v1'
    reproduction = (read_json(root / 'reproduction.json', warnings)
                    if gate_required or (root / 'reproduction.json').is_file() else {})
    validation_path = root / 'validation.jsonl'
    stopped_before_training = summary.get('phase') == 'baseline_check_failed' and summary.get('optimizer_steps') == 0
    rows = ([] if stopped_before_training and not validation_path.exists()
            else read_rows(validation_path, warnings))
    arguments = manifest.get('arguments', {})
    arguments = arguments if isinstance(arguments, dict) else {}
    suite = summary.get('suite', manifest.get('suite'))
    groups = GROUPS.get(suite, ())
    pairs = PAIRS.get(suite, ())
    smoke = any(value is True for value in
                (summary.get('smoke'), manifest.get('smoke'), arguments.get('smoke')))
    target = arguments.get('steps')
    target_valid = type(target) is int and target > 0
    endpoint = next((row for row in reversed(rows) if target_valid and row['step'] == target), None)
    final_metrics = summary.get('final_metrics')
    required = ('old_pixel', 'old_region') + groups
    final_complete = isinstance(final_metrics, dict) and all(isinstance(final_metrics.get(name), dict) for name in required)
    row_complete = endpoint is not None and all(isinstance(endpoint['metrics'].get(name), dict) for name in required)
    metrics_match = final_complete and row_complete and all(final_metrics[name] == endpoint['metrics'][name] for name in required)
    completed = (summary.get('complete') is True and summary.get('smoke') is False and not smoke and bool(groups)
                 and manifest.get('format') == 'PARTITION_LEARNING_V1'
                 and manifest.get('suite') == suite and manifest.get('variants') == list(groups)
                 and target_valid and target == 5000 and arguments.get('eval_every') == 500
                 and summary.get('final_step') == target and metrics_match
                 and endpoint.get('image_count') == 450
                 and summary.get('immutable_front_verified') is True
                 and summary.get('test_accessed') is False
                 and (not gate_required or summary.get('baseline_reproduction_pass') is True))
    lines = [f'format: {manifest.get("format", "N/A")}; suite: {suite or "N/A"}',
             f'complete: {summary.get("complete", "N/A")}; smoke: {smoke}; target step: {target or "N/A"}',
             f'phase: {summary.get("phase", "N/A")}; optimizer_steps: {summary.get("optimizer_steps", "N/A")}; '
             f'baseline_reproduction_pass: {summary.get("baseline_reproduction_pass", "N/A")}',
             f'Primary completed formal endpoint available: {completed}',
             'old_pixel and old_region are frozen source-front references, NOT original A1.']
    if gate_required or reproduction:
        lines += reproduction_lines(reproduction)
    if gate_required and summary.get('baseline_reproduction_pass') is True and reproduction.get('passed') is not True:
        warnings.append('Summary claims baseline reproduction passed but reproduction.json does not confirm it; inspect gate evidence.')
    lines.append('Same-step foreground mIoU gains (pp; second minus first):')
    if not rows:
        lines.append('  No trained validation rows recorded.')
    for row in rows:
        gains = same_step_gains(row['metrics'], pairs)
        values = ', '.join(f'{key}={value:+.4f}' if value is not None else f'{key}=N/A'
                           for key, value in gains.items())
        lines.append(f'  step={row["step"]}: {values or "suite/pairs unavailable"}')
    selected = endpoint if completed else (max(rows, key=lambda row: row['step']) if rows else None)
    if selected:
        label = 'Final same-step metrics' if completed else 'Latest available validation (provisional, not a formal final result)'
        lines.append(f'{label}: step={selected["step"]}')
        names = required + (('source_hard',) if 'source_hard' in selected['metrics'] else ())
        for name in names:
            lines.append(metric_line(name, selected['metrics'].get(name)))
        lines.append('Routing: ' + _json(selected.get('routing', {})))
        lines.append('Q drift: ' + _json(selected.get('q_drift', {})))
        for key, stage in (('base_partition_quality_totals', 'base'), ('partition_quality_totals', 'read')):
            quality = selected.get(key)
            if completed and quality != summary.get(key):
                warnings.append(f'Partition-quality totals differ ({key}) between summary and endpoint; display validation-row evidence only.')
            lines += partition_quality_lines(quality, groups, stage=stage)
    else:
        initial = read_json(root / 'initial.json', warnings) if (root / 'initial.json').is_file() else {}
        initial_metrics = initial.get('metrics')
        if isinstance(initial_metrics, dict):
            lines.append('Initial source-reproduction evaluation (before training; NOT trained validation): '
                         f'image_count={initial.get("image_count", "N/A")}')
            lines.append('Branch metrics below describe untrained initializations, not optimizer updates.')
            for name in ('old_pixel', 'old_region', 'source_hard', *groups):
                if name in initial_metrics:
                    lines.append(metric_line(name, initial_metrics[name]))
            lines.append('Initial parity evidence: ' + _json({key: initial.get(key) for key in
                         ('identity_max_abs', 'region_replay_max_abs', 'initial_replay_max_abs', 'initial_replay_exempt')}))
        else:
            lines.append('No valid validation metrics available; missing results are not treated as zero.')
        if stopped_before_training:
            lines.append('Baseline check failed before optimization: optimizer_steps=0; no trained result exists.')
    lines.append('Secondary best (separate steps; never the primary comparison): ' +
                 _json({'scores': summary.get('best'), 'steps': summary.get('best_steps')}))
    lines.append('Training verification: ' + _json({key: summary.get(key) for key in
                 ('weights_changed', 'nonzero_gradient_steps', 'immutable_front_verified', 'test_accessed')}))
    lines.append('Resources (combined frozen front + all branches in this suite, not per-method): ' +
                 _json({key: summary.get(key) for key in ('elapsed_seconds', 'peak_cuda_allocated_bytes')}))
    if completed:
        computed = same_step_gains(endpoint['metrics'], pairs)
        recorded = summary.get('same_step_gains_pp', {})
        recorded = recorded if isinstance(recorded, dict) else {}
        for pair, value in computed.items():
            stated = recorded.get(pair)
            if value is None or not numeric(stated) or not math.isclose(value, stated, abs_tol=1e-6):
                warnings.append(f'{pair}: summary gain missing/inconsistent; use the displayed same-step metrics.')
        audit = read_json(root / 'audit_final.json', warnings)
        if audit:
            lines += _audit_lines(audit, pairs)
        else:
            lines.append('Final audit unavailable; no diagnostic evidence inferred from aggregate scores.')
    else:
        lines.append('Final audit is not presented until a completed formal endpoint is verified.')
        if summary.get('complete') is not True:
            warnings.append('Incomplete run: all displayed validation evidence is provisional.')
        if smoke:
            warnings.append('Smoke validates execution only, NOT full-Val performance.')
        if summary.get('complete') is True and not smoke:
            warnings.append('Completion evidence does not match the required format, suite, 5000/500 protocol, Val450, metrics, frozen-front, Test-access or baseline-reproduction checks.')
        if gate_required and summary.get('baseline_reproduction_pass') is not True:
            warnings.append('Required baseline reproduction pass is absent or false; formal completion is not verified.')
    if summary.get('test_accessed') is not False:
        warnings.append('No-Test status is unverified; do not infer it from missing metadata.')
    lines += ['WARNING: ' + warning for warning in warnings]
    lines.append('GT is used only for training losses/evaluation; building grades do not identify within-building damage locations.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(render(args.root))


if __name__ == '__main__':
    main()
