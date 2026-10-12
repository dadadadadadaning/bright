"""CPU-only JSON report. Does not import torch, run inference, or write files."""
import argparse
import json
from pathlib import Path


def number(value, scale=1.):
    return 'N/A' if value is None else f'{value*scale:.4f}'


def report(root):
    path = Path(root)
    if path.is_dir():
        summary = path/'summary.json'
        if summary.is_file():
            s = json.loads(summary.read_text(encoding='utf-8'))
            print('complete:', s.get('complete'), 'smoke:', s.get('smoke'),
                  'reproduction:', s.get('reproduction_pass', s.get('final_metrics_reproduced')))
            if s.get('complete') is not True:
                print('WARNING: incomplete run; candidate evidence is not an accepted full audit.')
        final = path/'audit_final.json'
        path = final if final.is_file() else path/'audit_candidate.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if 'groups' not in data:
        raise ValueError('Use an audit_final.json (or its containing run), not a training metrics log')
    print('Evidence:', path, '\nImages:', data.get('image_count'))
    all_row = data['groups']['all']
    print('\nFull-image scores (NOT conditional scores):')
    for name, m in all_row['metrics'].items():
        print(name, 'FG mIoU%=', number(m['valid_miou'], 100),
              'Damaged P/R/F1%=', '/'.join(number(m[k][2], 100) for k in ('precision','recall','f1')))
    pairs = all_row['comparisons']
    preferred = ('old_pixel__source_hard', 'old_pixel__hard', 's0__s2')
    pair = next((p for p in preferred if p in pairs), next(iter(pairs), None))
    if pair is None:
        raise ValueError('No paired prediction comparisons in this file')
    print('\nExact prediction comparison:', pair)
    for key, row in data['groups'].items():
        if key not in ('all','boundary','foreground_interior','size_small','size_medium','size_large') and not key.startswith(('roi_', 'b_bin_')):
            continue
        c = row['comparisons'].get(pair)
        if c is None or not c.get('count'):
            continue
        print(key, 'n=', c['count'], 'harm=', c['harmed'], 'rescue=', c['rescued'],
              'net=', c['net'], 'harm%=', number(c['harm_rate'],100),
              'D->I harm=', c['damaged_to_intact_harm'], 'I->D rescue=', c['intact_to_damaged_rescue'])
    pooling = data.get('pooling_evidence')
    if pooling is None:
        print('\nThis existing file already covers ROI/confidence errors, but NOT actual attention/stats pooling mixtures.')
        return
    print('\nPooling stratum associations (FRACTIONAL pixels, NOT causal contribution):')
    for label, row in pooling['associations'].items():
        context = label.rsplit('/',1)[-1]
        if context not in ('foreground','roi_true_positive','roi_false_negative') or not row['weighted_pixels']:
            continue
        print(label, 'weighted_n=', number(row['weighted_pixels']),
              'harm%=', number(row['harm_rate'],100), 'net=', number(row['net']),
              'D->I harm=', number(row['damaged_to_intact_harm']))
    print('\nSlot support (mass units = native feature positions; empty slots excluded from purity bins):')
    for key, row in pooling['slots'].items():
        if not row['slots']:
            continue
        fg = row['foreground_mass']; mass = row['valid_mass']
        print(key, 'slots=', row['slots'], 'mass=', number(row['mass']),
              'mean_ESS=', number(row['ess_sum']/row['slots']),
              'GTpurity_massweighted=', number(row['purity_valid_mass_sum']/mass if mass else None),
              'FGgradepurity_fgweighted=', number(row['foreground_purity_fg_mass_sum']/fg if fg else None))
    print('\nInterpretation: B-error harm suggests checking routing; mixed + B-correct harm suggests scale/support;')
    print('clean/supported + B-correct harm suggests checking representation/classification. These are hypotheses, not proof.')
    print('GT is offline only. Different criteria overlap. Do not sum them or treat pixels as independent samples.')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True, help='Existing run directory or explicit audit JSON file')
    report(p.parse_args().root)


if __name__ == '__main__':
    main()
