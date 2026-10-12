"""Read-only factorial report; missing evidence is never interpreted as zero."""
import argparse
import json
import math
from pathlib import Path


def read_json(path, warnings):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        warnings.append(f'{path.name}: unreadable/incomplete JSON ({exc})')
        return None


def read_rows(path, warnings):
    rows = []
    if path.exists():
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('step'), int) or not isinstance(row.get('metrics'), dict):
                    raise ValueError('missing step/metrics')
                rows.append(row)
            except ValueError as exc:
                warnings.append(f'{path.name}:{number}: skipped incomplete/invalid line ({exc})')
    return rows


def score(row, name):
    value = row.get('metrics', {}).get(name, {}).get('valid_miou')
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def contrasts(row):
    """All terms come from this single evaluation step, in percentage points."""
    terms = {
        'Q effect without Bsplit (s0-g0)': {'s0': 1, 'g0': -1},
        'Q effect with Bsplit (s2-g1)': {'s2': 1, 'g1': -1},
        'Bsplit effect on grid (g1-g0)': {'g1': 1, 'g0': -1},
        'Bsplit effect on Q (s2-s0)': {'s2': 1, 's0': -1},
        'interaction (s2-s0)-(g1-g0)': {'s2': 1, 's0': -1, 'g1': -1, 'g0': 1},
        'Q main effect': {'s0': .5, 'g0': -.5, 's2': .5, 'g1': -.5},
        'Bsplit main effect': {'g1': .5, 'g0': -.5, 's2': .5, 's0': -.5},
        'hardening without split (h0-s0)': {'h0': 1, 's0': -1},
        'hardening with split (h1-s2)': {'h1': 1, 's2': -1},
        'hard Q versus grid with split (h1-g1)': {'h1': 1, 'g1': -1},
        'Bsplit effect on hard Q (h1-h0)': {'h1': 1, 'h0': -1},
    }
    return {label: (None if any(score(row, name) is None for name in weights)
                    else 100 * sum(weight * score(row, name) for name, weight in weights.items()))
            for label, weights in terms.items()}


def render(root, details=False):
    root = Path(root)
    warnings = []
    manifest = read_json(root/'manifest.json', warnings) or {}
    summary = read_json(root/'summary.json', warnings)
    rows = read_rows(root/'validation.jsonl', warnings)
    args = manifest.get('arguments', {})
    names = args.get('variants', ['g0', 'g1', 's0', 's2'])
    target = args.get('steps')
    complete = isinstance(summary, dict) and summary.get('complete') is True
    lines = [f"summary.complete: {summary.get('complete', 'unavailable') if isinstance(summary, dict) else 'unavailable'}",
             f"smoke: {manifest.get('smoke', 'unavailable')}; front step: {manifest.get('checkpoint_step', 'unavailable')}",
             f'planned final step: {target}']
    final_rows = [r for r in rows if r['step'] == target and all(score(r, n) is not None for n in names)]
    final = final_rows[-1] if final_rows else None
    full = final is not None and final.get('image_count') == 450 and not manifest.get('smoke', True)
    lines.append(f'completed full-Val fixed-step evidence: {complete and full}')
    if not complete:
        warnings.append('Run is not marked complete; latest/best rows are provisional.')
    if not full:
        warnings.append('Primary final-step full-Val evidence unavailable (missing/incomplete/smoke).')
    if not rows:
        lines.append('No completed validation rows; initial validation can take time.')
    else:
        last = max(rows, key=lambda r: r['step'])
        selected = final if final is not None else last
        lines.append(f"{'Final fixed-step' if final is not None else 'Latest provisional'} metrics, step {selected['step']}:")
        for name in names:
            value = score(selected, name)
            lines.append(f"  {name}: FG mIoU={100*value:.4f}%" if value is not None else f'  {name}: unavailable')
            metric = selected.get('metrics', {}).get(name, {})
            lines.append('    class IoU / F1 [Background, Intact, Damaged, Destroyed]: '+
                         json.dumps({'iou':metric.get('iou','unavailable'),'f1':metric.get('f1','unavailable')}))
            lines.append('    actual routing: '+json.dumps(selected.get('routing', {}).get(name, 'unavailable'), ensure_ascii=False))
        lines.append('Same-step factorial contrasts (pp; never differences of independently selected best steps):')
        for row in sorted(rows, key=lambda r: r['step']):
            lines.append(f"  step {row['step']}:")
            lines.extend(f"    {key}: {'unavailable' if value is None else f'{value:+.4f}'}" for key, value in contrasts(row).items())
        lines.append('Secondary best-per-variant (selection-biased; not the factorial primary endpoint):')
        for name in names:
            candidates = [r for r in rows if score(r, name) is not None]
            if candidates:
                best = max(candidates, key=lambda r: score(r, name))
                lines.append(f"  {name}: step={best['step']} FG mIoU={100*score(best,name):.4f}%")
            else:
                lines.append(f'  {name}: unavailable')
        diagnostic = selected.get('diagnostics')
        if selected.get('diagnostics_file') == 'audit_final.json':
            diagnostic = read_json(root/'audit_final.json', warnings)
        if not isinstance(diagnostic, dict):
            lines.append('Diagnostic subsets: unavailable (not zero)')
        else:
            lines.append('Diagnostic subsets (conditional scores are not full-image mIoU):')
            if not diagnostic.get('groups'):
                lines.append('  unavailable (not zero)')
            for label, bucket in diagnostic.get('groups', {}).items():
                if details:
                    compact = bucket
                else:
                    compact = {pair: {k:values.get(k,'unavailable') for k in
                               ('count','rescued','harmed','net','damaged_to_intact_harm','intact_to_damaged_rescue')}
                               for pair,values in bucket.get('comparisons',{}).items() if pair in ('s0__s2','g1__s2')}
                    if not compact:
                        compact = {'evidence':'unavailable (not zero)'}
                lines.append('  '+label+': '+json.dumps(compact, ensure_ascii=False))
            lines.append('Per-event same-step differences (pp):')
            for label,bucket in diagnostic.get('per_event',{}).items():
                subset = {key:value for key,value in contrasts(bucket).items() if key in
                          ('Q effect with Bsplit (s2-g1)','Bsplit effect on Q (s2-s0)')}
                lines.append('  '+label+': '+json.dumps(subset,ensure_ascii=False))
            if not diagnostic.get('per_event'):
                lines.append('  unavailable (not zero)')
            all_metrics = diagnostic.get('groups',{}).get('all',{}).get('metrics',{})
            lines.append('Final full-image Damaged precision/recall/F1 and background-to-building errors:')
            for name in names:
                metric = all_metrics.get(name,{})
                compact = {key:(metric[key][2] if isinstance(metric.get(key),list) and len(metric[key])>2 else 'unavailable')
                           for key in ('precision','recall','f1')}
                compact['background_to_building'] = metric.get('background_to_building','unavailable')
                lines.append('  '+name+': '+json.dumps(compact))
            image_rows = diagnostic.get('per_image')
            lines.append('Per-image paired evidence: '+(str(len(image_rows))+' rows in audit_final.json' if isinstance(image_rows,list) else 'unavailable (not zero)'))
            lines.append('Diagnostic protocol: '+json.dumps(diagnostic.get('protocol', 'unavailable'), ensure_ascii=False))
    if isinstance(summary, dict):
        lines.append('Resources (all heads combined): '+json.dumps({key:summary.get(key, 'unavailable') for key in
                     ('elapsed_seconds','peak_cuda_allocated_bytes','resource_scope')}, ensure_ascii=False))
    lines.extend('WARNING: '+w for w in warnings)
    lines.append('Slot ceilings are not occupied-region counts; predicted mixing is not GT purity. Smoke is not full-Val evidence.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--details', action='store_true', help='Include full diagnostic group matrices')
    args = parser.parse_args()
    print(render(args.root, details=args.details))


if __name__ == '__main__':
    main()
