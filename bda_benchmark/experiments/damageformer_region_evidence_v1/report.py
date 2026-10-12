"""Print completed audit evidence without loading any model or accessing data."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--root',required=True); a=p.parse_args()
    d=json.loads((Path(a.root)/'summary.json').read_text())
    print('complete:',d['complete'],'smoke:',d['smoke'],'checkpoint step:',d['checkpoint_step'])
    print('Q unchanged:',d['q_unchanged'],'Test accessed:',d['test_accessed'])
    print('\nFull Val metrics (smoke is NOT full Val):')
    for mode,m in d['metrics'].items():
        print(mode,'foreground mIoU:',round(100*m['valid_miou'],4),'IoU:',m['iou'])
    print('\nMixing bucket associations (cross-component bucket overlaps purity buckets):')
    for bucket,row in d['mixing_buckets'].items(): print(bucket,row)
    print('\nGT Damaged: pixel correct -> candidate Intact / reverse rescue:')
    for mode,t in d['transitions'].items():
        print(mode,'harm D->I:',t[2][2][1],'rescue I->D:',t[2][1][2])
    print('\nRegion counts:',d['region_counts'])
    print('Pixel-weighted soft purity:',d['pixel_weighted_soft_purity'])
    print('These associations/interventions are not causal proof or retrained ablations.')

if __name__=='__main__': main()
