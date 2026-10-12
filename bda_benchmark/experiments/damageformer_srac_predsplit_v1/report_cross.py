"""Read audit JSON only; can also report size rates from old comparison_v1."""
import argparse
import json
from pathlib import Path
from .audit import LABELS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--summary',required=True)
    a=p.parse_args()
    data=json.loads(Path(a.summary).read_text(encoding='utf-8'))
    if not data.get('complete') or not all(data.get('reproduction_pass',{}).values()):
        raise ValueError('Audit incomplete or reproduction failed')
    def percent(x): return 'N/A' if x is None else f'{100*x:.4f}%'
    for pair in ['A1__S3_split','A1__Grid_split','Grid_split__S3_split']:
        print('\n===',pair,'===')
        for group in ['all','boundary','nonboundary','foreground_interior',
                      'size_small','size_medium','size_large'] + [
                      label+'__'+part for label in LABELS for part in ['boundary','interior']]:
            if group not in data['groups']: continue
            r=data['groups'][group]['comparisons'][pair]
            print(f"{group:25s} pixels={r['count']} base_wrong={r['base_wrong']} "
                  f"base_correct={r['base_correct']} rescue={r['rescued']} harm={r['harmed']} "
                  f"net={r['net']:+d} recovery={percent(r['recovery_rate'])} harm_rate={percent(r['harm_rate'])}")
        for group in ['all','Damaged__boundary','Damaged__interior']:
            if group not in data['groups']: continue
            r=data['groups'][group]['comparisons'][pair]
            if 'harm_gt_to_wrong' not in r: continue
            print('\nTransitions:',group)
            for key in ['harm_gt_to_wrong','rescue_gt_from_wrong']:
                print(key,'rows=GT; columns=wrong class; order=',LABELS)
                for name,row in zip(LABELS,r[key]): print(name,row)
    if 'Damaged__boundary' not in data['groups']:
        print('\nOld audit: size rates available, but cross/transition statistics require --cross_stats rerun.')
    print('\nCaution: class-conditioned/size-conditioned scores are not full-image mIoU; GT components are not building instances.')


if __name__=='__main__': main()
