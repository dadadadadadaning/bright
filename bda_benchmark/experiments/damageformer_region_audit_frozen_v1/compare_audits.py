#!/usr/bin/env python3
import argparse,json
from pathlib import Path
KEYS=[('all','regions_per_building_mean'),('all','regions_per_building_median'),('all','building_1_region_fraction'),('all','building_ge5_region_fraction'),
('all','mean_region_building_purity'),('all','mean_background_crossing'),('all','mean_damage_entropy_norm'),('all','mean_dominant_damage_purity'),
('region_level_all','multi_building_crossing_fraction'),('damaged_evidence_coverage','recall_at_0.5'),('damaged_evidence_coverage','recall_at_0.6'),('damaged_evidence_coverage','recall_at_0.7')]
p=argparse.ArgumentParser(); p.add_argument('summaries',nargs='+'); a=p.parse_args(); data=[(Path(x).parent.name,json.loads(Path(x).read_text(encoding='utf-8'))) for x in a.summaries]
print('metric\t'+'\t'.join(n for n,_ in data))
for g,k in KEYS:
    vals=[]
    for _,d in data:
        v=d.get(g,{}).get(k); vals.append('' if v is None else f'{float(v):.6f}')
    print(f'{g}.{k}\t'+'\t'.join(vals))
