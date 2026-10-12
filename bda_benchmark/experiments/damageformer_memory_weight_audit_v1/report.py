"""Read-only report of fixed, within-support attention-weight interventions."""
import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_memory_readout_audit_v1.report import (
    LABELS, _counts, _delta, _fmt, _item, _number, _paired_counts, _read,
)


FORMAT = 'MEMORY_WEIGHT_AUDIT_V1'
CASES = ('source_hard', 'nomemory', 'normal', 'identity', 'uniform', 'cosine',
         'old_pixel', 'old_region')
PRIMARY = 'normal__cosine'
PAIRS = ('nomemory__normal', 'nomemory__uniform', 'nomemory__cosine',
         'normal__uniform', PRIMARY, 'uniform__cosine', 'normal__identity',
         'source_hard__normal')
CI_PAIRS = (PRIMARY, 'nomemory__cosine', 'uniform__cosine')
REPLAY_FIELDS = ('max_abs_context', 'max_abs_logits',
                 'identity_region_argmax_disagreements',
                 'identity_pixel_argmax_disagreements',
                 'max_self_weight_error', 'max_external_mass_error',
                 'max_invalid_weight')


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _identity_passed(summary):
    replay = _mapping(summary.get('replay'))
    return (summary.get('identity_passed') is True and
            all(_number(replay.get(key)) and replay[key] == 0 for key in
                ('identity_region_argmax_disagreements',
                 'identity_pixel_argmax_disagreements')))


def _reproduced(summary):
    reproduction = summary.get('reproduction')
    return reproduction is True or (
        isinstance(reproduction, dict) and reproduction.get('passed') is True)


def _reproduction_summary(summary):
    value = summary.get('reproduction')
    row = _mapping(value)
    passed = value if isinstance(value, bool) else row.get('passed', 'N/A')
    lines = [f'Reproduction / 已训练结果复现: passed={passed}; status={row.get("status", "N/A")}']
    for scope in ('source', 'learned'):
        gate = _mapping(row.get(scope))
        checks = gate.get('checks')
        if isinstance(checks, list):
            passed_n = sum(_mapping(check).get('passed') is True for check in checks)
            failed_n = sum(_mapping(check).get('passed') is False for check in checks)
            counts = (f'checks={len(checks)}; passed_checks={passed_n}; failed_checks={failed_n}; '
                      f'unknown_checks={len(checks)-passed_n-failed_n}')
        else:
            counts = 'checks=N/A; passed_checks=N/A; failed_checks=N/A; unknown_checks=N/A'
        lines.append(f'  {scope}: passed={gate.get("passed", "N/A")}; '
                     f'status={gate.get("status", "N/A")}; ' + counts)
        for check in checks if isinstance(checks, list) else ():
            check = _mapping(check)
            if check.get('passed') is True:
                continue
            label = str(check.get('name', check.get('baseline', 'N/A')))
            label += '/' + str(check.get('metric', 'N/A'))
            if check.get('class_index') is not None:
                label += '/' + str(check['class_index'])
            state = 'FAIL' if check.get('passed') is False else 'UNVERIFIED'
            lines.append(f'    {state} {label}: delta={_fmt(check.get("delta"))}; '
                         f'tolerance={_fmt(check.get("tolerance"))}; '
                         f'reason={check.get("reason", "N/A")}')
    lines.append('  Full reproduction evidence is preserved in reproduction.json.')
    return lines


def _comparison(pair, metrics, bucket):
    first, second = pair.split('__')
    a, b = _mapping(metrics.get(first)), _mapping(metrics.get(second))
    row = _mapping(_mapping(bucket.get('comparisons')).get(pair))
    lines = [f'{"PRIMARY" if pair == PRIMARY else "COMPARE"} {second} minus {first}: '
             f'FG mIoU {_fmt(_delta(a.get("valid_miou"), b.get("valid_miou")), percent=True, signed=True)}; '
             + _paired_counts(row)]
    if not row:
        lines.append(f'  {pair}: paired evidence unavailable; '
                     '不能从两个混淆矩阵推断救回与改错各多少。')
    return lines


def _foreground_counts(row):
    """Use paired transitions by GT class; missing transitions are not zeros."""
    totals = {}
    for key, field in (('rescued', 'rescue_gt_from_wrong'),
                       ('harmed', 'harm_gt_to_wrong')):
        matrix = row.get(field)
        if (not isinstance(matrix, list) or len(matrix) != 4 or any(
                not isinstance(values, list) or len(values) != 4 or any(
                    not _number(value) or value < 0 for value in values)
                for values in matrix)):
            totals[key] = None
        else:
            totals[key] = sum(sum(values) for values in matrix[1:])
    totals['net'] = _delta(totals['harmed'], totals['rescued'])
    return '; '.join(f'{key}={_fmt(value)}' for key, value in totals.items())


def _detail(pair, audit, metrics):
    first, second = pair.split('__')
    a, b = _mapping(metrics.get(first)), _mapping(metrics.get(second))
    lines = [f'DETAIL {second} minus {first}: class tradeoffs / 每类权衡']
    for c, label in enumerate(LABELS):
        fields = [f'd{key}={_fmt(_delta(x, y))}' for key, x, y in
                  zip(('TP', 'FP', 'FN'), _counts(a, c), _counts(b, c))]
        fields += [f'd{key}={_fmt(_delta(_item(a.get(key), c), _item(b.get(key), c)), percent=True, signed=True)}'
                   for key in ('iou', 'precision', 'recall', 'f1')]
        lines.append('  ' + label + ': ' + '; '.join(fields))
    if _counts(a, 0)[0] is not None and _counts(b, 0)[0] is not None:
        lines.append('  Damaged FP change by true class: ' + '; '.join(
            f'{LABELS[c]}={b["confusion"][c][2] - a["confusion"][c][2]:+g}'
            for c in (0, 1, 3)))
    groups = _mapping(audit.get('groups'))
    whole = _mapping(groups.get('all'))
    full_row = _mapping(_mapping(whole.get('comparisons')).get(pair))
    lines.append('  GT foreground (all GT>0): ' + _foreground_counts(full_row))
    lines.append('Subsets / ROI、边界与大小: conditional scores are NOT '
                 'full-image mIoU or additive mIoU components.')
    for key in ('all', 'boundary', 'interior', 'foreground_interior',
                'roi_true_positive', 'roi_false_negative', 'roi_false_positive',
                'roi_true_negative', 'building_prob_mid',
                *(label + '__boundary' for label in LABELS),
                *(label + '__interior' for label in LABELS),
                'size_small', 'size_medium', 'size_large'):
        bucket = _mapping(groups.get(key))
        row = _mapping(_mapping(bucket.get('comparisons')).get(pair))
        lines.append(f'  {key}: ' + _paired_counts(row) +
                     f'; Damaged->Intact harm={_fmt(row.get("damaged_to_intact_harm"))}' +
                     f'; Intact->Damaged rescue={_fmt(row.get("intact_to_damaged_rescue"))}')
        if key == 'boundary':
            lines.append('    GT building-side boundary: ' + _foreground_counts(row))
    lines.append('  ROI uses fixed predicted B and GT for evaluation only; '
                 'size groups are GT foreground connected-component proxies, NOT building instances.')
    events = _mapping(audit.get('per_event'))
    lines.append('Per-event / 事件:')
    signs = []
    for name, bucket in sorted(events.items()):
        bucket = _mapping(bucket)
        rows = _mapping(bucket.get('metrics'))
        ea, eb = _mapping(rows.get(first)), _mapping(rows.get(second))
        delta = _delta(ea.get('valid_miou'), eb.get('valid_miou'))
        if delta is not None:
            signs.append(delta)
        support = ea.get('support')
        support_text = '/'.join(_fmt(v) for v in support) if isinstance(support, list) else 'N/A'
        row = _mapping(_mapping(bucket.get('comparisons')).get(pair))
        lines.append(f'  {name}: delta={_fmt(delta, percent=True, signed=True)}; '
                     f'GT support B/I/D/X={support_text}; ' + _paired_counts(row))
    if events:
        lines.append(f'  Event directions with defined scores: increased={sum(x > 0 for x in signs)}; '
                     f'decreased={sum(x < 0 for x in signs)}; unchanged={sum(x == 0 for x in signs)}; '
                     f'unavailable={len(events)-len(signs)}')
    else:
        lines.append('  Event evidence unavailable / 无事件证据。')
    return lines


def _reader(summary):
    routing = _mapping(summary.get('routing'))
    lines = ['Routing / 读出槽位: ' + '; '.join(
        f'{key}={_fmt(routing.get(key), percent=key.endswith("fraction"))}' for key in
        ('active_slots', 'hard_slots', 'tail_slots', 'tail_mass_fraction')),
        'Reader values are slot averages, not Q-mass-weighted accuracy or pixel accuracy. '
        'Separate mass-weighted lines use pooling mass.',
        'A_true is external correct-class label mass after external-weight normalization, '
        'averaged over attention_label_queries (labelled supported slots with nonzero external attention). '
        'cos_true uses supported_queries; region accuracy uses labelled_queries.',
        'Attention and correct-label mass describe association, NOT causal contribution; '
        'label mass is NOT classification accuracy. '
        'attention_by_head is self-token weight, not a class probability.']
    stats = _mapping(summary.get('reader_stats'))
    cases = _mapping(stats.get('cases', stats))
    columns = (('n', 'queries'), ('labelled', 'labelled_queries'),
               ('supported', 'supported_queries'), ('A_labelled', 'attention_label_queries'),
               ('selfA', 'query_attention_mean'), ('extA', 'external_attention_mean'),
               ('A_true', 'external_label_true_mass'), ('cos_true', 'cosine_label_true_mass'),
               ('sourceAcc', 'source_region_accuracy'), ('readerAcc', 'reader_region_accuracy'),
               ('changed', 'changed_region_fraction'))
    lines.append('Reader columns: ' + ', '.join(f'{short}={key}' for short, key in columns))
    for case in ('normal', 'identity', 'uniform', 'cosine'):
        groups = _mapping(cases.get(case))
        lines += [f'Reader {case}:', 'group | ' + ' | '.join(short for short, _ in columns)]
        for group in ('all', 'hard', 'tail', *LABELS,
                      *(f'hard__{c}' for c in LABELS), *(f'tail__{c}' for c in LABELS)):
            row = _mapping(groups.get(group))
            lines.append(group + ' | ' + ' | '.join(_fmt(row.get(key)) for _, key in columns))
            if group in ('all', 'hard', 'tail'):
                heads = row.get('attention_by_head')
                lines.append(f'  {group} attention_by_head: ' + (
                    ', '.join(_fmt(v) for v in heads) if isinstance(heads, list) else 'N/A'))
                lines.append(f'  {group} mass-weighted: mass={_fmt(row.get("mass"))}; '
                             f'labelled_mass={_fmt(row.get("labelled_mass"))}; ' + '; '.join(
                    f'{key}={_fmt(row.get(key + "_mass_weighted"))}' for key in
                    ('query_attention', 'external_attention', 'delta_abs',
                     'centered_delta_abs', 'prob_l1', 'changed_region')))
        if not groups:
            lines.append('  Reader statistics unavailable / 读取统计缺失。')
    differences = _mapping(summary.get('case_differences'))
    for case in ('identity', 'uniform', 'cosine'):
        row = _mapping(differences.get(case))
        lines.append(f'Intervention {case} versus normal: ' + '; '.join(
            f'{key}={_fmt(row.get(key))}' for key in
            ('queries', 'prob_l1', 'centered_delta_l1', 'changed_region_fraction')))
        for group in ('hard', 'tail'):
            values = _mapping(_mapping(row.get('groups')).get(group))
            lines.append(f'  {group}: ' + '; '.join(f'{key}={_fmt(values.get(key))}' for key in
                         ('queries', 'prob_l1', 'centered_delta_l1', 'changed_region_fraction')))
    return lines


def _bootstrap(pair, summary, audit, repetitions):
    rows = audit.get('per_image')
    eligible = (summary.get('format') == FORMAT and summary.get('complete') is True and
                summary.get('smoke') is False and summary.get('image_count') == 450 and
                summary.get('temperature') == .07 and
                summary.get('test_accessed') is False and audit.get('test_accessed') is False and
                audit.get('image_count') == 450 and isinstance(rows, list) and len(rows) == 450 and
                _identity_passed(summary) and _reproduced(summary) and
                all(summary.get(key) is True for key in
                    ('source_verified', 'bank_verified', 'head_verified')))
    if not eligible:
        return [f'Bootstrap unavailable ({pair}): requires complete non-smoke Val450, '
                'verified source/bank/head, passed reproduction and identity, fixed temperature=0.07, Test=False '
                'and 450 paired records; no interval fabricated.']
    try:
        from bda_benchmark.experiments.damageformer_roi_followup_v1.report import paired_bootstrap
        result = paired_bootstrap(rows, pair, repetitions)
        result_text = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (KeyError, ValueError, TypeError, ImportError, OverflowError) as exc:
        return [f'Bootstrap unavailable ({pair}): invalid/missing paired evidence '
                f'({type(exc).__name__}); no interval fabricated.']
    return ['Paired image/event-stratified bootstrap: ' + result_text]


def render(root, bootstrap=0):
    """Describe all predeclared contrasts without fitting or choosing a model."""
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    root, warnings = Path(root), []
    summary = _read(root / 'summary.json', warnings)
    audit = _read(root / 'audit_final.json', warnings)
    lines = [f'format: {summary.get("format", "N/A")}',
             f'complete: {summary.get("complete", "N/A")}; smoke: {summary.get("smoke", "N/A")}; '
             f'image_count: {summary.get("image_count", "N/A")}; Test: {summary.get("test_accessed", "N/A")}',
             'Provenance / 来源核验: ' + '; '.join(f'{key}={summary.get(key, "N/A")}'
                 for key in ('source_verified', 'bank_verified', 'head_verified')),
             *_reproduction_summary(summary),
             f'Identity passed: {summary.get("identity_passed", "N/A")}',
             'Replay / 手工重放与权重守恒: ' + '; '.join(
                 f'{key}={_fmt(_mapping(summary.get("replay")).get(key))}' for key in REPLAY_FIELDS),
             'normal uses the original production reader path; identity manually replays unchanged weights. '
             'Identity is an implementation check, not a competing model.',
             f'Cosine temperature: {_fmt(summary.get("temperature"))}; '
             'predeclared fixed temperature=0.07; no Val temperature sweep or best-temperature selection.']
    if summary.get('format') != FORMAT:
        warnings.append('Expected MEMORY_WEIGHT_AUDIT_V1; format unavailable or mismatched.')
    if summary.get('complete') is not True:
        warnings.append('Incomplete run / 尚未完成；不能作最终有效性结论。')
    if summary.get('smoke') is not False:
        warnings.append('Smoke or unknown smoke status: execution evidence only, NOT final performance.')
    if not _identity_passed(summary):
        warnings.append('Identity check FAILED or unavailable: diagnostic values only; '
                        '不能将差异解释为邻居权重干预效果。')
    if not _reproduced(summary):
        warnings.append('Trained reproduction not verified / 已训练结果复现尚未通过。')
    if any(summary.get(key) is not True for key in ('source_verified', 'bank_verified', 'head_verified')):
        warnings.append('Source/bank/head provenance incomplete / 来源核验尚未全部通过。')
    if summary.get('test_accessed') is not False or (audit and audit.get('test_accessed') is not False):
        warnings.append('Test=False was not verified / 未确认无 Test 访问。')
    if audit and audit.get('image_count') != summary.get('image_count'):
        warnings.append('summary/audit image_count mismatch / 图像数不一致。')
    if summary.get('temperature') != .07:
        warnings.append('Fixed temperature=0.07 not verified / 固定温度记录缺失或不符。')
    lines.extend('WARNING: ' + value for value in warnings)
    whole = _mapping(_mapping(audit.get('groups')).get('all'))
    metrics = dict(_mapping(summary.get('final_metrics')))
    metrics.update(_mapping(whole.get('metrics')))
    lines.append('Full-image metrics / 全图指标 (FG mIoU = fixed Intact/Damaged/Destroyed mean):')
    for case in CASES:
        row = _mapping(metrics.get(case))
        lines.append(f'{case}: FG mIoU={_fmt(row.get("valid_miou"), percent=True)}; '
                     f'Background->building={_fmt(row.get("background_to_building"))}; '
                     f'building->Background={_fmt(row.get("building_to_background"))}')
        for c, label in enumerate(LABELS):
            lines.append('  ' + label + ': ' + '; '.join(
                f'{key}={_fmt(_item(row.get(key), c), percent=True)}'
                for key in ('iou', 'precision', 'recall', 'f1')) + '; ' + '; '.join(
                f'{key}={_fmt(value)}' for key, value in zip(('TP', 'FP', 'FN'), _counts(row, c))))
    lines.append('Predeclared comparisons: primary cosine-normal; always also cosine-nomemory '
                 'and cosine-uniform. No ranking or automatic model selection.')
    for pair in PAIRS:
        lines.extend(_comparison(pair, metrics, whole))
    for pair in CI_PAIRS:
        lines.extend(_detail(pair, audit, metrics))
    lines.extend(_reader(summary))
    if bootstrap:
        for pair in CI_PAIRS:
            lines.extend(_bootstrap(pair, summary, audit, bootstrap))
        lines.append('CI conditions on observed events; not seed variance or unseen-event generalization; '
                     'spatial tile dependence unresolved. The three intervals are pointwise, '
                     'not a multiple-comparison selection guarantee.')
    lines.extend([
        'Interpretation / 判读：cosine-normal tests redistribution among the same retrieved neighbors; '
        'cosine-uniform tests similarity guidance; cosine-nomemory checks the practical Memory increment.',
        'Per-query, per-head self weight and total external weight are preserved; invalid support weights stay zero. '
        'GT is used only for evaluation, never for support reweighting.',
        'uniform/cosine are fixed-checkpoint inference interventions, NOT retrained ablations. '
        'A negative result does not establish all retrained readers are ineffective.',
        'source_hard, old_pixel and old_region are outputs of the verified frozen source, NOT historical A1. '
        'nomemory is the independently trained matched-capacity head, not this reader with evidence disabled.',
        'The report describes evidence only; no automatic effectiveness verdict or model selection / '
        '不预设有效，不自动选模型。',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('bootstrap must be nonnegative')
    print(render(args.root, bootstrap=args.bootstrap))


if __name__ == '__main__':
    main()
