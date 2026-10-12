"""Compact read-only report: fixed-step evidence and explicit class tradeoffs."""
import argparse
import json
from pathlib import Path
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.report import read_json, read_rows

LABELS = ('Background', 'Intact', 'Damaged', 'Destroyed')


def percent(value):
    return 'N/A' if value is None else f'{100*value:.4f}%'


def class_counts(metric, c):
    cm = metric['confusion']
    tp = cm[c][c]
    return tp, sum(row[c] for row in cm)-tp, sum(cm[c])-tp


def contrast(metric_a, metric_b):
    """Same-evaluation class differences. TP/FP/FN come from full confusion."""
    result = {}
    for c, label in enumerate(LABELS):
        a, b = class_counts(metric_a, c), class_counts(metric_b, c)
        row = dict(zip(('tp_change', 'fp_change', 'fn_change'), (y-x for x,y in zip(a,b))))
        for key in ('iou', 'precision', 'recall', 'f1'):
            x, y = metric_a[key][c], metric_b[key][c]
            row[key+'_pp'] = None if x is None or y is None else 100*(y-x)
        result[label] = row
    return result


def compact_audit(data, pair=None):
    lines = []
    groups = data.get('groups', {})
    whole = groups.get('all', {})
    ms = whole.get('metrics', {})
    pairs = whole.get('comparisons', {})
    if pair is None:
        pair = next((p for p in ('hard__exchange','s0__s2') if p in pairs), None)
    lines.append(f'Full-image scores; image_count={data.get("image_count", "N/A")}')
    for name, metric in ms.items():
        lines.append(f'{name}: FG mIoU={percent(metric.get("valid_miou"))}; background->building={metric.get("background_to_building", "N/A")}')
        for c, label in enumerate(LABELS):
            lines.append('  '+label+' '+', '.join(k+'='+percent(metric[k][c]) for k in ('iou','precision','recall','f1')))
    if pair not in pairs:
        lines.append('Requested primary pair unavailable; not treated as zero.')
        return lines
    first, second = pair.split('__')
    lines.append(f'PRIMARY PAIR: {second} minus {first} (same step)')
    for label, row in contrast(ms[first], ms[second]).items():
        lines.append(label+': '+json.dumps(row, allow_nan=False))
    # False Damaged predictions may originate from any other GT class.
    lines.append('Damaged FP change by true class: '+json.dumps({LABELS[c]: ms[second]['confusion'][c][2]-ms[first]['confusion'][c][2] for c in (0,1,3)}))
    lines.append('Correct -> wrong and reverse rescue by true class (not all wrong -> wrong transitions):')
    row = pairs[pair]
    for c in (1,2,3):
        lines.append(LABELS[c]+': '+json.dumps({LABELS[d]:dict(harm=row['harm_gt_to_wrong'][c][d], rescue=row['rescue_gt_from_wrong'][c][d]) for d in range(4) if d!=c}))
    lines.append('Subsets: rescue/harm and denominators, NOT additive mIoU components:')
    for group, bucket in groups.items():
        if group.startswith('b_bin_'):
            continue
        values = bucket.get('comparisons',{}).get(pair)
        if values is None:
            continue
        lines.append(group+': '+json.dumps({k:values.get(k) for k in ('count','base_correct','base_wrong','rescued','harmed','net','harm_rate','recovery_rate','damaged_to_intact_harm','intact_to_damaged_rescue')}))
    edges = data.get('protocol',{}).get('b_bins')
    if edges:
        lines.append('B confidence x ROI correctness. Only nonempty bins shown; rates use base_correct/base_wrong.')
        for i in range(len(edges)-1):
            for kind in ('false_negative','false_positive','true_positive','true_negative'):
                bucket = groups.get(f'b_bin_{i}__roi_{kind}',{})
                values = bucket.get('comparisons',{}).get(pair)
                if values is None or not values['count']:
                    continue
                label = f'B[{edges[i]:.1f},{edges[i+1]:.1f}{"]" if i==len(edges)-2 else ")"} roi_{kind}'
                lines.append(label+': '+json.dumps({k:values.get(k) for k in ('count','base_correct','base_wrong','rescued','harmed','net','harm_rate','recovery_rate')}))
    else:
        lines.append('B confidence bins unavailable in old audit. Run frozen audit; cannot derive bins from aggregate confusion.')
    lines.append('Per-event foreground mIoU difference (pp; check class supports):')
    for name, bucket in data.get('per_event',{}).items():
        a, b = bucket['metrics'][first], bucket['metrics'][second]
        delta = None if a['valid_miou'] is None or b['valid_miou'] is None else 100*(b['valid_miou']-a['valid_miou'])
        lines.append(name+': '+json.dumps(dict(delta_pp=delta,support=a.get('support'))))
    lines.append(f'Per-image rows retained: {len(data.get("per_image",[]))}; no pixel-independent significance claim.')
    return lines


def render(root, pair=None):
    root = Path(root)
    warnings = []
    manifest = read_json(root/'manifest.json', warnings) or {}
    summary = read_json(root/'summary.json', warnings) or {}
    rows = read_rows(root/'validation.jsonl', warnings)
    audit = read_json(root/'audit_final.json', warnings)
    lines = [f'complete: {summary.get("complete", "N/A")}; smoke: {summary.get("smoke", manifest.get("smoke", "N/A"))}',
             f'format: {manifest.get("format", "N/A")}']
    if manifest.get('format') == 'ROI_EXCHANGE_V1':
        lines += [f'Frozen front step: {manifest.get("checkpoint_step")}; epsilon: {manifest.get("epsilon")}',
                  'old_pixel is the frozen V2 snapshot output, NOT original A1.']
        initial = read_json(root/'initial.json',warnings) or {}
        lines.append('Hard vs old S2 parity: '+json.dumps(initial.get('hard_s2_parity','not yet checked')))
        target = manifest.get('arguments',{}).get('steps')
        lines.append('Same-step foreground mIoU exchange-hard (pp):')
        for row in rows:
            m = row['metrics']
            if all(k in m for k in ('hard','exchange')):
                lines.append(f'  step={row["step"]} hard={percent(m["hard"]["valid_miou"])} exchange={percent(m["exchange"]["valid_miou"])} delta={100*(m["exchange"]["valid_miou"]-m["hard"]["valid_miou"]):+.4f}')
        final = next((r for r in reversed(rows) if r['step']==target),None)
        full = summary.get('complete') is True and final is not None and final.get('image_count')==450 and not manifest.get('smoke', True)
        lines.append(f'Primary completed full-Val endpoint available: {full}')
        latest = final or (rows[-1] if rows else None)
        if latest:
            lines.append('Routing counts (equal 2K slots do NOT mean equal occupied regions): '+json.dumps(latest.get('routing',{})))
        lines.append('Secondary best (do not use different best steps as primary): '+json.dumps(dict(scores=summary.get('best'),steps=summary.get('best_steps'))))
        lines.append('Resource scope is combined front + heads, not per-method: '+json.dumps({k:summary.get(k) for k in ('elapsed_seconds','peak_cuda_allocated_bytes','resource_scope')}))
    if manifest.get('format') == 'ROI_EXCHANGE_AUDIT_V1':
        lines.append('Frozen audit verification: '+json.dumps({k:manifest.get(k) for k in
                     ('final_step','immutable_front_verified','immutable_readouts_verified','same_step_verified','final_metrics_reproduced')}))
    if audit:
        lines += compact_audit(audit, pair)
    else:
        lines.append('Final audit unavailable: initial/periodic validation does not contain the full diagnostic.')
    if summary.get('complete') is not True:
        warnings.append('Incomplete run: shown evidence is provisional.')
    if summary.get('smoke', manifest.get('smoke', False)):
        warnings.append('Smoke validates execution only, NOT full-Val performance.')
    lines += ['WARNING: '+w for w in warnings]
    lines.append('GT is diagnosis only; components are object proxies, NOT instances. No Test used.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pair', help='e.g. s0__s2 or hard__exchange')
    args = parser.parse_args()
    print(render(args.root,args.pair))


if __name__ == '__main__':
    main()
