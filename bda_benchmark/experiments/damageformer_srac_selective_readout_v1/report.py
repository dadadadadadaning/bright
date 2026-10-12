"""Read JSON evidence only; no GPU, inference or training."""
import argparse
import json
from pathlib import Path


def percentage(value):
    return 'N/A' if value is None else f'{100 * value:.4f}%'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    path = root / 'audit_final.json'
    if not path.is_file():
        path = root / 'audit.json'
    if not path.is_file():
        failure = root / 'failure.json'
        if failure.is_file():
            print('AUDIT FAILED. This is not a completed/full-Val result.')
            print(failure.read_text(encoding='utf-8'))
            return
        raise SystemExit('No completed audit.json/audit_final.json here; inspect the running log: ' + str(root))
    result = json.loads(path.read_text(encoding='utf-8'))
    print('complete:', result.get('complete'), 'smoke:', result.get('smoke'), 'Test:', result.get('test_accessed'))
    if 'audit_scope' in result:
        print('audit scope:', result['audit_scope'])
        if not result['audit_scope']['full_validation']:
            print('SUBSET ONLY: these scores are NOT full-Val450 or a resumed final result.')
    print('reproduction:', result.get('reproduction', 'see step0.json for readout training'))
    for name, metrics in result['metrics'].items():
        print(f"{name:18s} FG mIoU={percentage(metrics['valid_miou'])} "
              f"Damaged P/R/F1={percentage(metrics['precision'][2])}/{percentage(metrics['recall'][2])}/{percentage(metrics['f1'][2])}")
        print('  class IoU:', [percentage(v) for v in metrics['iou']])
    for pair, values in result.get('paired', {}).items():
        print(pair, 'harm/rescue/net=', values['harmed'], values['rescued'], values['net'],
              'Damaged->Intact harm=', values['damaged_to_intact_harm'],
              'reverse rescue=', values['damaged_from_intact_rescue'])
    for group in ('roi_false_negative', 'roi_false_positive', 'foreground_boundary',
                  'background_boundary', 'foreground_interior', 'B_09_1', 'proxy_small'):
        row = result.get('groups', {}).get(group)
        if row:
            print(group, {pair: {'net': value['net'], 'D_to_I_harm': value['damaged_to_intact_harm'],
                                'I_to_D_rescue': value['damaged_from_intact_rescue']}
                          for pair, value in row.get('paired', {}).items()})
    if 'routing_mean_per_image' in result:
        print('routing (image means):', result['routing_mean_per_image'])
    print('Conditional groups overlap. Pixel counts are not independent samples or building-instance counts.')
    print('Smoke is NOT full-Val evidence. Boundary interventions are NOT retrained ablations.')


if __name__ == '__main__':
    main()
