"""Read-only summaries and optional paired image/event-stratified bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np
from bda_benchmark.experiments.damageformer_roi_exchange_v1.report import contrast

LABELS = ('Background','Intact','Damaged','Destroyed')


def score(cm):
    tp = np.diag(cm)
    union = cm.sum(0)+cm.sum(1)-tp
    return float((tp/np.maximum(union,1))[1:].mean())


def paired_bootstrap(rows, pair, repetitions=2000, seed=3407):
    """Images paired; resample within each fixed event stratum, never pixels.

    This conditions on the observed events. It is NOT an unseen-event estimate,
    a training-seed interval, nor a remedy for dependence between nearby tiles.
    """
    if repetitions < 1 or not rows:
        raise ValueError('Positive repetitions and per-image records required')
    first, second = pair.split('__')
    a = np.asarray([r['metrics'][first]['confusion'] for r in rows], dtype=np.int64)
    b = np.asarray([r['metrics'][second]['confusion'] for r in rows], dtype=np.int64)
    if a.shape != (len(rows),4,4) or b.shape != a.shape or (a < 0).any() or (b < 0).any():
        raise ValueError('Invalid per-image confusion records')
    if not np.array_equal(a.sum(2), b.sum(2)):
        raise ValueError('Paired images have inconsistent GT supports')
    strata = {}
    for i, row in enumerate(rows):
        strata.setdefault(row['event'], []).append(i)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        selected = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in strata.values()])
        deltas.append(100*(score(b[selected].sum(0))-score(a[selected].sum(0))))
    return dict(pair=pair,delta_pp=100*(score(b.sum(0))-score(a.sum(0))),
                percentile_95_pp=np.quantile(deltas,[.025,.975]).tolist(),
                repetitions=repetitions,seed=seed,image_count=len(rows),event_count=len(strata),
                scope='paired images within fixed events; not seed variance or unseen-event generalization; spatial tile dependence not resolved')


def render(root, pair=None, bootstrap=0):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    lines = [f'format: {manifest.get("format")}',
             f'complete: {summary.get("complete")}; smoke: {summary.get("smoke")}; Test: {summary.get("test_accessed")}']
    if (root/'audit_final.json').is_file():
        audit = json.loads((root/'audit_final.json').read_text(encoding='utf-8'))
        metrics = audit['groups']['all']['metrics']
    else:
        audit = None
        metrics = summary.get('final_metrics',{})
    for name,m in metrics.items():
        def pct(v): return 'N/A' if v is None else f'{100*v:.4f}%'
        lines.append(f'{name}: FG mIoU={pct(m["valid_miou"])}; D P/R/F1='+
                     '/'.join(pct(m[k][2]) for k in ('precision','recall','f1'))+
                     f'; background false positives={m["background_to_building"]}')
    for key in ('best','best_steps','support','diagonal_probability_max_abs','learned_base_q_mean_abs_change','learned_q_gradient_verified','bank_regions'):
        if key in summary:
            lines.append(key+': '+json.dumps(summary[key],allow_nan=False))
    if audit:
        pairs = audit['groups']['all']['comparisons']
        if pair is None:
            pair = next((p for p in ('exchange_EE__exchange_EH','hard_EE__hard_EH',
                'nomemory__memory','fixed_q__learn_q','hard_HH__hard_EH',
                'exchange_HH__exchange_EH','hard_HH__hard_HE','exchange_HH__exchange_HE') if p in pairs),
                next(iter(pairs),None))
        lines.append('Available same-evaluation pairs: '+', '.join(pairs))
        if pair not in pairs:
            raise ValueError('Choose a pair available in this audit with --pair')
        a,b = pair.split('__')
        lines.append(f'PRIMARY {b} minus {a}: {100*(metrics[b]["valid_miou"]-metrics[a]["valid_miou"]):+.4f} pp')
        for label,row in contrast(metrics[a],metrics[b]).items():
            lines.append(label+': '+json.dumps(row,allow_nan=False))
        lines.append('Damaged FP change by GT source: '+json.dumps({LABELS[c]:metrics[b]['confusion'][c][2]-metrics[a]['confusion'][c][2] for c in (0,1,3)}))
        for group in ('all','roi_true_positive','roi_false_negative','roi_false_positive','roi_true_negative',
                      'Background__boundary','Intact__boundary','Damaged__boundary','Destroyed__boundary',
                      'Damaged__interior','foreground_interior','size_small','size_medium','size_large'):
            row = audit['groups'].get(group,{}).get('comparisons',{}).get(pair)
            if row is not None:
                keys = ('count','rescued','harmed','net','harm_rate','recovery_rate','damaged_to_intact_harm','intact_to_damaged_rescue')
                lines.append(group+': '+json.dumps({k:row[k] for k in keys},allow_nan=False))
        for name,event in audit.get('per_event',{}).items():
            m = event['metrics']
            lines.append(f'event {name}: {100*(m[b]["valid_miou"]-m[a]["valid_miou"]):+.4f} pp; GT support={m[a]["support"]}')
        if 'routing' in audit:
            lines.append('Gather/read occupancy and blocked mass: '+json.dumps(audit['routing']))
        if bootstrap:
            if summary.get('complete') is not True or summary.get('smoke') or audit['image_count'] != 450:
                raise ValueError('Bootstrap requires completed non-smoke Val450')
            lines.append('Paired validation uncertainty: '+json.dumps(paired_bootstrap(audit['per_image'],pair,bootstrap)))
    if summary.get('complete') is not True or summary.get('smoke'):
        lines.append('WARNING: incomplete/smoke is execution evidence only, NOT a final performance result.')
    lines += ['No automatic model selection. Inspect foreground/background and ROI error tradeoffs before choosing a route.',
              'Same-weight routing and neighbor perturbations are not retrained ablations.',
              'Conditional scores are not full-image mIoU. GT components are object proxies, not building instances.',
              'old_pixel is the frozen 3000-step V2 snapshot, not historical A1. No Test used.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--pair')
    parser.add_argument('--bootstrap',type=int,default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('Bootstrap repetitions must be nonnegative')
    print(render(args.root,args.pair,args.bootstrap))


if __name__ == '__main__':
    main()
