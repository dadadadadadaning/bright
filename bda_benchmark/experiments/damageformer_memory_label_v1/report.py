"""Read-only final-validation evidence for region label memory.

The gate is a conservative, single-run screening rule, not statistical proof.
All comparisons use unrounded values; undefined ratios remain None.
"""
import argparse
import json
import math
from pathlib import Path


FORMAT = 'REGION_LABEL_MEMORY_V1'
LABELS = ('Background', 'Intact', 'Damaged', 'Destroyed')


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _delta(first, second):
    return 100 * (second - first) if _number(first) and _number(second) else None


def _class_value(metric, key, index):
    values = metric.get(key)
    return values[index] if isinstance(values, (list, tuple)) and len(values) == 4 else None


def _accuracy(metric):
    cm = metric.get('confusion')
    if not isinstance(cm, list) or len(cm) != 4 or any(len(row) != 4 for row in cm):
        return None
    total = sum(map(sum, cm))
    return sum(cm[i][i] for i in range(4)) / total if total else None


def metric_delta(first, second):
    """Second minus first, in percentage points; preserve absent class scores."""
    classes = {}
    for index, label in enumerate(LABELS):
        row = {key + '_pp': _delta(_class_value(first, key, index),
                                  _class_value(second, key, index))
               for key in ('iou', 'precision', 'recall', 'f1')}
        a, b = first.get('confusion'), second.get('confusion')
        if a is not None and b is not None:
            row.update(tp_change=int(b[index][index]) - int(a[index][index]),
                       fp_change=sum(int(r[index]) for r in b) - int(b[index][index])
                       - sum(int(r[index]) for r in a) + int(a[index][index]),
                       fn_change=sum(map(int, b[index])) - int(b[index][index])
                       - sum(map(int, a[index])) + int(a[index][index]))
        classes[label] = row
    return dict(foreground_miou_pp=_delta(first.get('valid_miou'), second.get('valid_miou')),
                accuracy_pp=_delta(_accuracy(first), _accuracy(second)), classes=classes)


def rescue_harm(row, base_metric):
    """Restrict the joint transition table by TRUE class, never prediction class."""
    matrices = [row.get('harm_gt_to_wrong'), row.get('rescue_gt_from_wrong')]
    if any(not isinstance(m, list) or len(m) != 4 or
           any(not isinstance(r, list) or len(r) != 4 for r in m) for m in matrices):
        raise ValueError('Missing 4x4 paired harm/rescue matrices; not treated as zero')
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
           for m in matrices for r in m for v in r):
        raise ValueError('Harm/rescue matrices must contain nonnegative integer counts')
    harm, rescue = matrices
    cm = base_metric['confusion']
    by_class = {}
    for c, label in enumerate(LABELS):
        harmed, rescued = sum(harm[c]), sum(rescue[c])
        correct = int(cm[c][c])
        by_class[label] = dict(count=sum(map(int, cm[c])), base_correct=correct,
                               base_wrong=sum(map(int, cm[c])) - correct,
                               rescued=rescued, harmed=harmed, net=rescued - harmed)

    def combine(indices):
        values = [by_class[LABELS[c]] for c in indices]
        result = {key: sum(v[key] for v in values)
                  for key in ('count', 'base_correct', 'base_wrong', 'rescued', 'harmed', 'net')}
        result['harm_rate'] = result['harmed'] / result['base_correct'] if result['base_correct'] else None
        result['recovery_rate'] = result['rescued'] / result['base_wrong'] if result['base_wrong'] else None
        return result

    return dict(all=combine(range(4)), gt_foreground=combine((1, 2, 3)),
                gt_damaged=combine((2,)), by_class=by_class,
                damaged_to_intact_harm=harm[2][1], intact_to_damaged_rescue=rescue[2][1],
                harm_gt_to_wrong=harm, rescue_gt_from_wrong=rescue)


def _events(audit, first, second):
    rows, missing = {}, []
    for name, bucket in audit.get('per_event', {}).items():
        metrics = bucket.get('metrics', {})
        if first not in metrics or second not in metrics:
            missing.append(name)
            continue
        row = metric_delta(metrics[first], metrics[second])
        row['support'] = metrics[first].get('support')
        rows[name] = row
    finite = {name: row['foreground_miou_pp'] for name, row in rows.items()
              if row['foreground_miou_pp'] is not None}
    worst = min(finite, key=finite.get) if finite else None
    return dict(rows=rows, event_count=len(rows), available_delta_count=len(finite),
                deteriorated_count=sum(value < 0 for value in finite.values()),
                worst_event=worst, worst_delta_pp=finite[worst] if worst is not None else None,
                missing_events=missing,
                policy='Count and worst deterioration are descriptive; no arbitrary event threshold.')


def assess(summary, audit):
    """Return reviewable evidence, without training, selection, or file writes.

    knn_mix is the prespecified audit candidate; knn_only remains diagnostic.
    Both source_hard and the paired nomemory baseline must be beaten in train.
    """
    if not isinstance(audit, dict) or not audit.get('groups', {}).get('all'):
        raise ValueError('Final audit unavailable: report not ready; missing evidence is not zero')
    if summary.get('format') != FORMAT or summary.get('mode') not in ('audit', 'train'):
        raise ValueError('Expected REGION_LABEL_MEMORY_V1 summary with audit or train mode')
    mode = summary['mode']
    candidate = 'memory' if mode == 'train' else 'knn_mix'
    primary = 'nomemory' if mode == 'train' else 'source_hard'
    prefix = 'memory' if mode == 'train' else 'knn'
    interventions = {kind: prefix + '_' + kind for kind in ('random', 'shuffle_labels', 'off')}
    baselines = ['source_hard', 'nomemory'] if mode == 'train' else ['source_hard']
    required_names = ['old_pixel', 'old_region', 'source_hard', 'knn_only', candidate,
                      *interventions.values(), *baselines]
    bucket = audit['groups']['all']
    metrics = bucket.get('metrics', {})
    pairs = bucket.get('comparisons', {})
    missing = ['metric:' + name for name in dict.fromkeys(required_names) if name not in metrics]
    comparisons = {}
    for pair, row in pairs.items():
        names = pair.split('__')
        if len(names) != 2 or any(name not in metrics for name in names):
            missing.append('invalid_pair:' + pair)
            continue
        first, second = names
        try:
            counts = rescue_harm(row, metrics[first])
        except (ValueError, KeyError, TypeError) as error:
            missing.append(pair + ': ' + str(error))
            continue
        comparisons[pair] = dict(baseline=first, candidate=second,
                                 delta=metric_delta(metrics[first], metrics[second]),
                                 rescue_harm=counts, per_event=_events(audit, first, second))
    # Require direct paired evidence: aggregate confusion cannot reconstruct it.
    required_pairs = [('source_hard', name) for name in dict.fromkeys(required_names)
                      if name not in ('source_hard', 'old_pixel', 'old_region')]
    required_pairs += [('old_pixel', name) for name in dict.fromkeys(required_names)
                       if name not in ('old_pixel', 'old_region')]
    required_pairs += [(candidate, name) for name in interventions.values()]
    required_pairs += [(baseline, candidate) for baseline in baselines]
    for first, second in dict.fromkeys(required_pairs):
        pair = first + '__' + second
        if pair not in comparisons:
            missing.append('comparison:' + pair)
    if not audit.get('per_event'):
        missing.append('per_event')
    final_metrics = summary.get('final_metrics', {})
    for name in dict.fromkeys(required_names):
        if name not in final_metrics:
            missing.append('summary.final_metrics:' + name)
        elif name in metrics:
            a, b = final_metrics[name].get('valid_miou'), metrics[name].get('valid_miou')
            if _number(a) and _number(b) and abs(a - b) > 1e-12:
                missing.append('summary/audit metric mismatch:' + name)
    full_val = (summary.get('complete') is True and summary.get('smoke') is False
                and summary.get('image_count') == 450 and audit.get('image_count') == 450)
    gates = dict(complete_non_smoke_val450=full_val,
                 baseline_reproduction_verified=summary.get('baseline_reproduction_pass') is True,
                 frozen_source_verified=summary.get('frozen_source_verified') is True,
                 test_not_accessed=summary.get('test_accessed') is False and audit.get('test_accessed') is False)
    baseline_checks = {}
    for baseline in baselines:
        comparison = comparisons.get(baseline + '__' + candidate)
        counts = comparison['rescue_harm'] if comparison else {}
        delta = comparison['delta'] if comparison else {}
        f1 = delta.get('classes', {}).get('Damaged', {}).get('f1_pp')
        miou = delta.get('foreground_miou_pp')
        checks = dict(foreground_miou_improved=miou is not None and miou > 0,
                      all_rescue_exceeds_harm=counts.get('all', {}).get('net', 0) > 0,
                      gt_foreground_rescue_exceeds_harm=counts.get('gt_foreground', {}).get('net', 0) > 0,
                      damaged_net_nonnegative=comparison is not None and counts['gt_damaged']['net'] >= 0,
                      damaged_f1_available=f1 is not None,
                      damaged_f1_nondecreasing=f1 is not None and f1 >= 0)
        baseline_checks[baseline] = checks
        gates.update({baseline + ':' + name: value for name, value in checks.items()})
    perturbations = {}
    for kind, name in interventions.items():
        if candidate in metrics and name in metrics:
            # Always show normal minus intervention, so positive means dependence.
            perturbations[kind] = dict(normal=candidate, intervention=name,
                                       normal_minus_intervention=metric_delta(metrics[name], metrics[candidate]))
        else:
            perturbations[kind] = dict(normal=candidate, intervention=name,
                                       normal_minus_intervention=None)
    for kind in ('random', 'shuffle_labels'):
        delta = perturbations[kind]['normal_minus_intervention']
        value = delta['foreground_miou_pp'] if delta is not None else None
        gates['normal_beats_' + kind] = value is not None and value > 0
    gates['required_evidence_available'] = not missing
    positive = all(gates.values())
    return dict(format=FORMAT, mode=mode, candidate=candidate, primary_baseline=primary,
                final_step=summary.get('final_step'), ready=not missing, missing_evidence=missing,
                complete_non_smoke_val450=full_val, metrics=metrics, comparisons=comparisons,
                baseline_checks=baseline_checks, interventions=perturbations,
                controls=summary.get('controls'), gates=gates,
                verification={name: summary.get(name) for name in
                              ('baseline_reproduction_pass', 'frozen_source_verified', 'test_accessed')},
                positive_signal=positive,
                conclusion=('Conservative single-run positive signal; not statistical proof.' if positive
                            else 'Positive-signal gate not passed; this does not imply an implementation bug.'),
                limitations=['No training-seed or unseen-event uncertainty estimate is supplied.',
                             'Interventions use the same weights; they are not retrained ablations.',
                             'knn_only is a standalone diagnostic, not a selected efficacy candidate.',
                             'Val GT is diagnosis only and does not enter query routing or training; Train GT supplies the memory labels.',
                             'old_pixel is the frozen source snapshot, not historical A1.',
                             'No automatic model selection. This report targets Val only; recorded Test access must be false for a positive signal.'])


def _pct(value):
    return 'N/A' if value is None else f'{100 * value:.4f}%'


def render(root, bootstrap=0, details=False):
    """Read only the explicitly supplied run directory; never discover other runs."""
    if bootstrap < 0:
        raise ValueError('Bootstrap repetitions must be nonnegative')
    root = Path(root)
    try:
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
        audit = json.loads((root / 'audit_final.json').read_text(encoding='utf-8'))
        evidence = assess(summary, audit)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return 'Report not ready: ' + str(error) + '\nPositive signal: False; missing evidence is not zero.'
    lines = [f'format: {FORMAT}; mode: {evidence["mode"]}; final_step: {evidence["final_step"]}',
             f'complete: {summary.get("complete")}; smoke: {summary.get("smoke")}; Val images: {audit.get("image_count")}',
             f'Report ready: {evidence["ready"]}; completed non-smoke Val450: {evidence["complete_non_smoke_val450"]}',
             f'Prespecified candidate: {evidence["candidate"]}; primary baseline: {evidence["primary_baseline"]}',
             'Full-image metrics; all deltas below are percentage points (pp).']
    for name, metric in evidence['metrics'].items():
        f1 = '/'.join(_pct(_class_value(metric, 'f1', c)) for c in (1, 2, 3))
        lines.append(f'{name}: foreground mIoU={_pct(metric.get("valid_miou"))}; Intact/Damaged/Destroyed F1={f1}')
    for pair, row in evidence['comparisons'].items():
        if not details and row['candidate'] not in (evidence['candidate'], 'knn_only'):
            continue
        lines.append(f'PAIR {pair}: {row["candidate"]} minus {row["baseline"]}')
        lines.append('Metric deltas: ' + json.dumps(row['delta'], allow_nan=False))
        for group in ('all', 'gt_foreground', 'gt_damaged', 'by_class'):
            lines.append(group + ' rescue/harm: ' + json.dumps(row['rescue_harm'][group], allow_nan=False))
        events = row['per_event'] if details else dict(
            event_count=row['per_event']['event_count'],
            deteriorated_count=row['per_event']['deteriorated_count'],
            worst_event=row['per_event']['worst_event'], worst_delta_pp=row['per_event']['worst_delta_pp'],
            deltas_pp={k:v['foreground_miou_pp'] for k,v in row['per_event']['rows'].items()})
        lines.append('Per-event deterioration: ' + json.dumps(events, allow_nan=False))
    lines.append('Interventions, NORMAL minus intervention: ' + json.dumps(evidence['interventions'], allow_nan=False))
    lines.append('Separate gate checks: ' + json.dumps(evidence['gates'], allow_nan=False))
    lines.append('Positive signal: ' + str(evidence['positive_signal']))
    lines.append(evidence['conclusion'])
    if not details:
        lines.append('Use --details for all paired/control class and event details; full evidence is retained in audit_final.json.')
    if evidence['controls'] is not None:
        lines.append('Controls: ' + json.dumps(evidence['controls'], allow_nan=False))
    lines.append('Recorded run verification: ' + json.dumps(evidence['verification'], allow_nan=False))
    label_path = root / 'label_evidence.json'
    if label_path.is_file():
        try:
            labels = json.loads(label_path.read_text(encoding='utf-8'))
            lines.append('Neighbor label alignment (observational overlap, not causal proof): '
                         + json.dumps(labels, allow_nan=False))
        except (OSError, ValueError) as error:
            lines.append('Optional neighbor-label alignment unavailable: ' + str(error))
    if evidence['missing_evidence']:
        lines.append('Missing evidence: ' + json.dumps(evidence['missing_evidence']))
    if not evidence['complete_non_smoke_val450']:
        lines.append('Smoke/incomplete/partial-Val results establish execution only, not efficacy.')
    if bootstrap:
        if not evidence['ready'] or not evidence['complete_non_smoke_val450']:
            raise ValueError('Bootstrap requires ready completed non-smoke Val450')
        from bda_benchmark.experiments.damageformer_roi_followup_v1.report import paired_bootstrap
        pair = evidence['primary_baseline'] + '__' + evidence['candidate']
        lines.append('Optional paired validation uncertainty: ' + json.dumps(
            paired_bootstrap(audit.get('per_image', []), pair, repetitions=bootstrap), allow_nan=False))
    lines.extend(evidence['limitations'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, help='Explicit run directory containing summary and final audit')
    parser.add_argument('--bootstrap', type=int, default=0, help='Optional existing paired image bootstrap repetitions')
    parser.add_argument('--details', action='store_true', help='Print all paired comparisons and full per-event class metrics')
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    print(render(args.root, args.bootstrap, args.details))


if __name__ == '__main__':
    main()
