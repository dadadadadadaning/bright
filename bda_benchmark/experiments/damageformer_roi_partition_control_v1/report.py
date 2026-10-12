"""Read-only report; no data/model loading."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    print('smoke:', manifest['smoke'], 'front step:', manifest['checkpoint_step'])
    print('complete:', (root/'summary.json').exists())
    rows = []
    path = root/'validation.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print('Warning: skipped incomplete/invalid validation line')
    if not rows:
        print('No completed validation yet; initial full Val can take time.')
        return
    last = max(rows, key=lambda r: r['step'])
    for name in manifest['arguments']['variants']:
        best = max(rows, key=lambda r: r['metrics'][name]['valid_miou'])
        m = best['metrics'][name]
        print(f"{name}: best step={best['step']} FG mIoU={100*m['valid_miou']:.4f}% "
              f"Damaged F1={100*m['f1'][2]:.4f}% last={100*last['metrics'][name]['valid_miou']:.4f}%")
        print('best class IoU:', m['iou'])
        print('last routing:', last['routing'][name])
    print('Same-step gains relative to S0 (pp):')
    for row in rows:
        if 's0' in row['metrics']:
            base = row['metrics']['s0']['valid_miou']
            print(row['step'], {k: round(100*(row['metrics'][k]['valid_miou']-base), 4)
                               for k in manifest['arguments']['variants']})
    print('S2/C2 have equal slot ceilings, not necessarily equal effective region counts.')
    print('Predicted mixing is not GT purity. Smoke is not full-Val evidence.')


if __name__ == '__main__':
    main()
