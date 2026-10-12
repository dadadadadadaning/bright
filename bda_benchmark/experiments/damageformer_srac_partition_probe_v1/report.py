"""Read saved probe evidence only; never evaluates a network."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get('mode') == 'repeatability':
        print('PASS' if summary.get('complete') and summary.get('gates_passed') else 'INCOMPLETE/FAILED',
              'single-image repeatability only; see repeats.jsonl. No full-Val performance conclusion.')
        return
    if not summary.get('complete') or not summary.get('gates_passed'):
        print('INCOMPLETE/FAILED: do not interpret intervention gains. Inspect per_image.jsonl / failure diagnostics.')
        return
    if not summary.get('full_validation_evidence'):
        print('SMOKE/SUBSET ONLY: not full-Val evidence.')
    result = json.loads((root/'audit.json').read_text(encoding='utf-8'))
    for name, m in result['metrics'].items():
        print(name, 'FG mIoU %=', round(100*m['valid_miou'], 5),
              'Damaged P/R/F1=', [m[k][2] for k in ('precision', 'recall', 'f1')])
    for pair, row in result['paired'].items():
        print(pair, 'net=', row['net'], 'D->I harm=', row['damaged_to_intact_harm'],
              'I->D rescue on GT-D=', row['damaged_from_intact_rescue'])
    for name, row in result.get('routing', {}).items():
        print('ROUTING', name, row['image_means'])
        print('CLASS-AREA diagnostics', name, json.dumps(row['per_class_sums']))
    print('Full per-event, ROI-error/B-confidence, boundary/interior and size-proxy evidence: audit.json.')
    print('Per-image iteration convergence, probability contracts, timings: per_image.jsonl.')
    print('Inference interventions only. Frozen trained head may dislike a better new partition. No Test.')


if __name__ == '__main__':
    main()
