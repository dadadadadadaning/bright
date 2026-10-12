"""Read-only report of the trained reader; no fitting, selection, or Test access."""
import argparse
import json
import math
from pathlib import Path


FORMAT = 'MEMORY_READOUT_AUDIT_V1'
LABELS = ('Background', 'Intact', 'Damaged', 'Destroyed')
CASES = ('source_hard', 'nomemory', 'normal', 'shuffle_labels', 'off',
         'hard_only', 'tail_only', 'old_pixel', 'old_region')
PRIMARY = 'nomemory__normal'
PAIRS = (PRIMARY, 'source_hard__normal', 'normal__shuffle_labels', 'normal__off',
         'source_hard__hard_only', 'source_hard__tail_only',
         'hard_only__normal', 'tail_only__normal')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value, percent=False, signed=False):
    if not _number(value):
        return 'N/A'
    if percent:
        return f'{100 * value:{"+" if signed else ""}.4f}' + (' pp' if signed else '%')
    return f'{value:.6g}'


def _item(value, index):
    return value[index] if isinstance(value, (list, tuple)) and len(value) > index else None


def _delta(first, second):
    return second - first if _number(first) and _number(second) else None


def _read(path, warnings):
    if not path.is_file():
        warnings.append(f'{path.name}: unavailable / 文件缺失；未记录项不视为 0。')
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        warnings.append(f'{path.name}: unreadable / 无法读取 ({type(exc).__name__})；未记录项不视为 0。')
        return {}


def _counts(metric, c):
    cm = metric.get('confusion')
    if not isinstance(cm, list) or len(cm) != 4 or any(
            not isinstance(row, list) or len(row) != 4 or
            any(not _number(v) or v < 0 for v in row) for row in cm):
        return (None, None, None)
    tp = cm[c][c]
    return tp, sum(row[c] for row in cm) - tp, sum(cm[c]) - tp


def _comparison(pair, metrics, bucket):
    first, second = pair.split('__')
    a, b = metrics.get(first, {}), metrics.get(second, {})
    row = bucket.get('comparisons', {}).get(pair, {})
    delta = _delta(a.get('valid_miou'), b.get('valid_miou'))
    lines = [f'{"PRIMARY" if pair == PRIMARY else "COMPARE"} {second} minus {first}: '
             f'FG mIoU {_fmt(delta, percent=True, signed=True)}; '
             + _paired_counts(row)]
    if not row:
        lines.append(f'  {pair}: paired evidence unavailable；不能从两个混淆矩阵推断净纠错。')
    return lines


def _paired_counts(row):
    return '; '.join(f'{key}={_fmt(row.get(key), percent=key.endswith("rate"))}'
                     for key in ('count', 'base_correct', 'base_wrong', 'rescued',
                                 'harmed', 'net', 'harm_rate', 'recovery_rate'))


def _detail(pair, audit, metrics):
    first, second = pair.split('__')
    a, b = metrics.get(first, {}), metrics.get(second, {})
    lines = [f'DETAIL {second} minus {first}: each class / 每类权衡']
    for c, label in enumerate(LABELS):
        counts_a, counts_b = _counts(a, c), _counts(b, c)
        fields = [f'd{key}={_fmt(_delta(x, y))}' for key, x, y in
                  zip(('TP', 'FP', 'FN'), counts_a, counts_b)]
        fields += [f'd{key}={_fmt(_delta(_item(a.get(key), c), _item(b.get(key), c)), percent=True, signed=True)}'
                   for key in ('iou', 'precision', 'recall', 'f1')]
        lines.append('  ' + label + ': ' + '; '.join(fields))
    if _counts(a, 0)[0] is not None and _counts(b, 0)[0] is not None:
        lines.append('Damaged FP change by true class: ' + '; '.join(
            f'{LABELS[c]}={b["confusion"][c][2] - a["confusion"][c][2]:+g}'
            for c in (0, 1, 3)))
    groups = audit.get('groups', {})
    keys = ('all', 'boundary', 'interior', 'foreground_interior',
            'roi_true_positive', 'roi_false_negative', 'roi_false_positive',
            'roi_true_negative', 'building_prob_mid',
            *(label + '__boundary' for label in LABELS),
            *(label + '__interior' for label in LABELS),
            'size_small', 'size_medium', 'size_large')
    lines.append('Subsets / 前景、边界与 ROI: conditional scores are NOT full-image mIoU or additive components.')
    for key in keys:
        bucket = groups.get(key, {})
        row = bucket.get('comparisons', {}).get(pair)
        if row is not None:
            lines.append(f'  {key}: ' + _paired_counts(row) +
                         f'; Damaged->Intact harm={_fmt(row.get("damaged_to_intact_harm"))}' +
                         f'; Intact->Damaged rescue={_fmt(row.get("intact_to_damaged_rescue"))}')
    if not groups:
        lines.append('  Subset paired evidence unavailable / 无子集配对证据。')
    events = audit.get('per_event', {})
    lines.append('Per-event / 事件:')
    for name, bucket in sorted(events.items()):
        rows = bucket.get('metrics', {})
        ea, eb = rows.get(first, {}), rows.get(second, {})
        support = ea.get('support')
        support_text = '/'.join(_fmt(v) for v in support) if isinstance(support, list) else 'N/A'
        lines.append(f'  {name}: delta={_fmt(_delta(ea.get("valid_miou"), eb.get("valid_miou")), percent=True, signed=True)}; '
                     f'GT support B/I/D/X={support_text}; ' +
                     _paired_counts(bucket.get('comparisons', {}).get(pair, {})))
    if not events:
        lines.append('  Event evidence unavailable / 无事件证据。')
    return lines


def _reader(summary):
    routing = summary.get('routing', {})
    lines = ['Routing / 实际读出槽位: ' + '; '.join(
        f'{k}={_fmt(routing.get(k), percent=k.endswith("fraction"))}' for k in
        ('active_slots', 'hard_slots', 'tail_slots', 'tail_mass_fraction', 'mean_pixel_tail_fraction')),
        'Reader table means are slot averages / 槽位平均, not Q-mass-weighted accuracy. '
        'Separate mass-weighted lines use pooling mass; tail_mass_fraction is routing mass, not slot count.',
        'A_true averages labelled supported slots with nonzero external attention (attention_label_queries); '
        'cos_true averages labelled supported slots (supported_queries); sameA and region accuracy use labelled_queries.',
        'Attention is actual trained-reader attention averaged over heads; cosine posterior is separate label-only evidence. '
        'attention_by_head is the query self-token attention per head, not a class probability.']
    stats = summary.get('reader_stats', {})
    cases = stats.get('cases', stats)
    columns = (('n', 'queries'), ('labelled', 'labelled_queries'),
               ('supported', 'supported_queries'), ('A_labelled', 'attention_label_queries'),
               ('selfA', 'query_attention_mean'), ('extA', 'external_attention_mean'),
               ('A_true', 'external_label_true_mass'), ('cos_true', 'cosine_label_true_mass'),
               ('sameA', 'same_class_external_attention_mean'),
               ('absD', 'delta_abs_mean'), ('centerD', 'centered_delta_abs_mean'),
               ('probL1', 'prob_l1_mean'), ('changed', 'changed_region_fraction'),
               ('sourceAcc', 'source_region_accuracy'), ('readerAcc', 'reader_region_accuracy'))
    lines.append('Reader columns: ' + ', '.join(f'{short}={key}' for short, key in columns))
    present = False
    for case in CASES:
        groups = cases.get(case, {})
        if not isinstance(groups, dict) or not groups:
            continue
        present = True
        lines += [f'Reader {case}:', 'group | ' + ' | '.join(short for short, _ in columns)]
        for group in ('all', 'hard', 'tail', *LABELS,
                      *(f'hard__{c}' for c in LABELS), *(f'tail__{c}' for c in LABELS)):
            row = groups.get(group)
            if not isinstance(row, dict):
                continue
            lines.append(group + ' | ' + ' | '.join(_fmt(row.get(key)) for _, key in columns))
            heads = row.get('attention_by_head')
            if group in ('all', 'hard', 'tail') and isinstance(heads, list):
                lines.append(f'  {group} attention_by_head: ' + ', '.join(_fmt(v) for v in heads))
            if group in ('all', 'hard', 'tail'):
                lines.append(f'  {group} mass-weighted: mass={_fmt(row.get("mass"))}; '
                             f'labelled_mass={_fmt(row.get("labelled_mass"))}; ' + '; '.join(
                    f'{key}={_fmt(row.get(key + "_mass_weighted"))}' for key in
                    ('query_attention', 'external_attention', 'delta_abs',
                     'centered_delta_abs', 'prob_l1', 'changed_region')))
    if not present:
        lines.append('Reader statistics unavailable / 实际读取统计缺失。')
    for case in ('shuffle_labels', 'off'):
        row = summary.get('case_differences', {}).get(case, {})
        lines.append(f'Intervention {case} versus normal: ' + '; '.join(
            f'{key}={_fmt(row.get(key))}' for key in
            ('prob_l1', 'centered_delta_l1', 'changed_region_fraction')))
    return lines


def _tail_bins(bins):
    lines = ['Soft-tail region-to-pixel association / soft-tail 区域到像素关联:',
             'Fixed bin thresholds / 固定门槛: [0, 0.1, 0.5, 1]; bins =0, (0,0.1], (0.1,0.5], (0.5,1].',
             'Bins measure the actual soft backprojection weight fraction from tail slots; '
             'NOT attention or GT purity / 分箱依据实际 soft 回投权重比例，不是 attention 或 GT 纯度。',
             'Each paired rescue/harm count is conditional on its pixel bin; '
             'conditional scores are NOT full-image mIoU or additive mIoU components.']
    if not bins:
        lines.append('  Pixel-bin paired evidence unavailable / 像素分箱配对证据缺失。')
        return lines
    names = (('zero', '=0'), ('positive_le_01', '(0,0.1]'),
             ('gt_01_le_05', '(0.1,0.5]'), ('gt_05', '(0.5,1]'))
    for suffix, scope in (('', 'all valid pixels'), ('__foreground', 'GT foreground pixels')):
        for key, interval in names:
            bucket = bins.get(key + suffix)
            if not isinstance(bucket, dict):
                lines.append(f'  {scope} tail fraction {interval}: unavailable / 未记录。')
                continue
            lines.append(f'  {scope} tail fraction {interval}:')
            for pair in (PRIMARY, 'hard_only__normal'):
                first, second = pair.split('__')
                row = bucket.get('comparisons', {}).get(pair, {})
                lines.append(f'    {second} minus {first}: ' + _paired_counts(row))
    lines.append('normal minus hard_only restores the trained soft-tail residual; '
                 'both share the same Q, weights and hard-slot residuals / 恢复 soft-tail 残差的同权重推理干预。')
    return lines


def _bootstrap(pair, summary, audit, repetitions):
    reproduction = summary.get('reproduction', {})
    reproduced = reproduction is True or (isinstance(reproduction, dict) and
                  (reproduction.get('passed') is True or
                   ('passed' not in reproduction and reproduction.get('status') == 'passed')))
    eligible = (summary.get('complete') is True and summary.get('smoke') is False and
                summary.get('test_accessed') is False and summary.get('image_count') == 450 and
                audit.get('image_count') == 450 and audit.get('test_accessed') is False and
                len(audit.get('per_image', [])) == 450 and
                summary.get('format') == FORMAT and reproduced and
                all(summary.get(k) is True for k in ('source_verified', 'bank_verified', 'head_verified')))
    if not eligible:
        return ['Bootstrap unavailable: requires complete non-smoke Val450, verified source/bank/head, '
                'passed reproduction, Test=False and 450 paired records; no interval fabricated.']
    try:
        from bda_benchmark.experiments.damageformer_roi_followup_v1.report import paired_bootstrap
        result = paired_bootstrap(audit['per_image'], pair, repetitions)
    except (KeyError, ValueError, TypeError, ImportError) as exc:
        return [f'Bootstrap unavailable: invalid/missing paired evidence ({type(exc).__name__}); no interval fabricated.']
    return ['Paired image/event-stratified bootstrap: ' + json.dumps(result, ensure_ascii=False, allow_nan=False),
            'CI conditions on observed events; not seed variance or unseen-event generalization; spatial tile dependence unresolved.']


def render(root, pair=None, bootstrap=0):
    """Render available evidence without changing the experiment or its files.

    ``pair`` adds a secondary detailed comparison; the primary stays normal-nomemory.
    """
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    if pair is not None and (len(pair.split('__')) != 2 or pair.split('__')[0] == pair.split('__')[1]):
        raise ValueError('pair must be two distinct case names joined by __')
    warnings = []
    root = Path(root)
    summary = _read(root / 'summary.json', warnings)
    audit = _read(root / 'audit_final.json', warnings)
    tail_bins = _read(root / 'tail_pixel_bins.json', warnings)
    lines = [f'format: {summary.get("format", "N/A")}',
             f'complete: {summary.get("complete", "N/A")}; smoke: {summary.get("smoke", "N/A")}; '
             f'image_count: {summary.get("image_count", "N/A")}; Test: {summary.get("test_accessed", "N/A")}',
             'Provenance / 来源核验: ' + '; '.join(f'{k}={summary.get(k, "N/A")}'
                 for k in ('source_verified', 'bank_verified', 'head_verified')),
             'Reproduction / 已训练结果复现: ' + json.dumps(summary.get('reproduction'), ensure_ascii=False)]
    if summary.get('format') != FORMAT:
        warnings.append('Expected MEMORY_READOUT_AUDIT_V1; format unavailable or mismatched.')
    if summary.get('complete') is not True:
        warnings.append('Incomplete run / 尚未完成：仅显示已记录证据，不能作最终有效性结论。')
    if summary.get('smoke') is not False:
        warnings.append('Smoke or unknown smoke status: execution evidence only, NOT final performance / smoke 不能支持有效性。')
    if summary.get('test_accessed') is not False:
        warnings.append('Test=False was not verified / 未确认无 Test 访问。')
    if audit and audit.get('image_count') != summary.get('image_count'):
        warnings.append('summary/audit image_count mismatch / 图像数不一致。')
    lines.extend('WARNING: ' + line for line in warnings)
    replay = summary.get('replay', {})
    lines.append('Reader replay / 读取重放: ' + '; '.join(f'{k}={_fmt(replay.get(k))}'
                 for k in ('max_abs_context', 'max_abs_logits', 'argmax_disagreements',
                           'observation_pixel_argmax_disagreements',
                           'label_shuffle_attention_max_abs', 'source_probability_max_abs')))
    lines.append('Replay argmax disagreements concern observational need_weights=True only; '
                 'actual predictions remain on the original need_weights=False path.')
    whole = audit.get('groups', {}).get('all', {})
    metrics = dict(summary.get('final_metrics', {}))
    metrics.update(whole.get('metrics', {}))
    lines.append('Full-image metrics / 全图指标 (FG mIoU = fixed Intact/Damaged/Destroyed mean):')
    for case in CASES:
        row = metrics.get(case, {})
        lines.append(f'{case}: FG mIoU={_fmt(row.get("valid_miou"), percent=True)}; '
                     f'Background->building={_fmt(row.get("background_to_building"))}; '
                     f'building->Background={_fmt(row.get("building_to_background"))}')
        if row:
            for c, label in enumerate(LABELS):
                lines.append('  ' + label + ': ' + '; '.join(
                    f'{key}={_fmt(_item(row.get(key), c), percent=True)}'
                    for key in ('iou', 'precision', 'recall', 'f1')))
    for candidate in PAIRS:
        lines.extend(_comparison(candidate, metrics, whole))
    lines.extend(_detail(PRIMARY, audit, metrics))
    if pair and pair != PRIMARY:
        if pair in whole.get('comparisons', {}):
            lines.extend(_comparison(pair, metrics, whole))
            lines.extend(_detail(pair, audit, metrics))
        else:
            lines.append(f'Requested pair {pair} unavailable; not treated as zero.')
    lines.extend(_reader(summary))
    lines.extend(_tail_bins(tail_bins))
    if bootstrap:
        lines.extend(_bootstrap(PRIMARY, summary, audit, bootstrap))
        if pair and pair != PRIMARY and pair in whole.get('comparisons', {}):
            lines.extend(_bootstrap(pair, summary, audit, bootstrap))
    lines += [
        'Interpretation / 判读：主比较 normal−nomemory；source_hard 是当前实际冻结 source，old_pixel/old_region 也是该 source 的输出，NOT historical A1。',
        'off retains the trained reader with external evidence disabled; off is NOT nomemory or source_hard.',
        'hard_only/tail_only 保留同一已训练权重，仅干预对应槽位残差；是 inference interventions, NOT retrained ablations。',
        'hard_count>0 表示原生槽位有 hard occupancy；tail 表示 soft-mass-only 槽位；二者都不是建筑实例，也不自动表示高纯度。',
        'Actual attention and cosine posterior describe association, NOT causal contribution / 真实 attention 与 cosine posterior 不等于因果贡献。',
        'The report describes evidence only; no automatic effectiveness verdict or model selection / 不自动判定有效，不选模型。',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pair', help='Additional detailed comparison; primary remains nomemory__normal')
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('bootstrap must be nonnegative')
    print(render(args.root, args.pair, args.bootstrap))


if __name__ == '__main__':
    main()
