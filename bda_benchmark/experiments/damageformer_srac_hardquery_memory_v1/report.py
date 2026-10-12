"""Read saved evidence only; failed/smoke stages never become formal results."""
from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_json


def render(root):
    root = Path(root)
    lines = ['SRAC Hard-query Memory V1', '']
    for stage, filename in (('protocol', 'protocol.json'), ('evidence', 'evidence.json'),
                            ('queries', 'query_manifest.json'), ('metric', 'summary.json'),
                            ('reader', 'summary.json')):
        path = root / stage / filename
        if not path.exists():
            lines.append(f'{stage}: missing / not run')
            continue
        value = read_json(path)
        lines.append(f"{stage}: complete={value.get('complete')} smoke={value.get('smoke')} "
                     f"full_validation_evidence={value.get('full_validation_evidence', False)}")
        if value.get('complete') is not True:
            lines.append('  Failed or unfinished: do not interpret as method effectiveness.')
            continue
        if value.get('smoke') is True:
            lines.append('  Smoke only: interface/optimization check; not a BRIGHT performance result.')
            continue
        if stage == 'reader':
            final = value.get('final', {})
            if value.get('full_validation_evidence') is not True:
                lines.append('  Missing valid full-Val evidence; no score comparison promoted.')
                continue
            scores = final.get('metrics', {})
            baseline = scores.get('j1', {}).get('valid_miou')
            lines += ['', '| arm | foreground mIoU (%) | difference vs fixed J1 (pp) | D rescue | D harm | D FP change |',
                      '|---|---:|---:|---:|---:|---:|']
            for arm in ('j1', 'j2', 'j3', 'query_ungated', 'query_gated', 'bank_ungated', 'bank_gated', 'hard_ungated', 'hard_gated'):
                score = scores.get(arm, {}).get('valid_miou')
                if score is None or baseline is None:
                    continue
                transition = final.get('transitions', {}).get(arm, {}).get('classes', [])
                damaged = transition[2] if len(transition) == 4 else {}
                lines.append('| %s | %.6f | %+.6f | %s | %s | %s |' % (
                    arm, score * 100, (score - baseline) * 100, damaged.get('rescue', '-'),
                    damaged.get('harm', '-'), damaged.get('delta_false_positive', '-')))
            lines += ['', 'Use final matched-step results; best checkpoints were selected separately on Val.',
                      'No single-seed result establishes statistical reliability. Historical J2/J3 are contextual references, not matched correction-head controls.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(render(args.root))


if __name__ == '__main__':
    main()
