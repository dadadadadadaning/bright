"""Read existing audit_final.json only; never invent unavailable probability bins."""
import argparse
import json
from pathlib import Path

LABELS = ('Background', 'Intact', 'Damaged', 'Destroyed')


def print_audit(data, pairs=None):
    print('Rows=GT; columns=prediction. Conditional groups are not full-image scores.')
    print('B bins:', 'available' if data.get('protocol', {}).get('b_bins') else 'NOT AVAILABLE in this audit; rerun frozen evaluation to obtain them')
    groups = data['groups']
    for name, group in groups.items():
        if name not in ('all', 'roi_false_negative', 'roi_false_positive') and not name.startswith(('Intact__', 'Destroyed__', 'b_bin_')):
            continue
        print('\nGROUP', name)
        for model, values in group['metrics'].items():
            cm = values['confusion']
            print('MODEL', model, 'confusion', json.dumps(cm))
            for c, label in enumerate(LABELS):
                tp = cm[c][c]; fn = sum(cm[c])-tp; fp = sum(row[c] for row in cm)-tp
                ratio = lambda n, d: n/d if d else None
                print(label, json.dumps(dict(TP=tp, FN=fn, FP=fp, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn), IoU=ratio(tp,tp+fp+fn))))
        for pair, values in group['comparisons'].items():
            if pairs and pair not in pairs:
                continue
            print('PAIR', pair, json.dumps(values))
            # Preserve all 4x4 rows, including Intact and Destroyed transitions.
            print('harm_gt_to_wrong', json.dumps(values['harm_gt_to_wrong']))
            print('rescue_gt_from_wrong', json.dumps(values['rescue_gt_from_wrong']))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pairs', nargs='+')
    args = parser.parse_args()
    data = json.loads((Path(args.root)/'audit_final.json').read_text(encoding='utf-8'))
    print_audit(data, args.pairs)


if __name__ == '__main__':
    main()
