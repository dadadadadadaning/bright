#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from typing import Dict,List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.core import extract_state,parse_batch,seed_worker
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import DamageFormerA1CrossModalSuperpixel
from .instance_utils import InstanceIndex,load_instance_map
from .region_metrics import AuditConfig,add_size_bucket,association_dense_statistics,audit_one_image,summarize

def read_list(path): return [x.strip() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def strip_module_prefix(state): return {k[7:]:v for k,v in state.items()} if state and all(k.startswith('module.') for k in state) else state
def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser('Audit learned region structure on BRIGHT Validation only')
    p.add_argument('--checkpoint',required=True); p.add_argument('--dataset_path',required=True); p.add_argument('--val_list',required=True)
    p.add_argument('--output_dir',required=True); p.add_argument('--instance_dir',default=''); p.add_argument('--region_stride',type=int,default=8)
    p.add_argument('--batch_size',type=int,default=1); p.add_argument('--num_workers',type=int,default=4); p.add_argument('--device',default='cuda:0')
    p.add_argument('--max_batches',type=int,default=0); p.add_argument('--effective_region_threshold',type=float,default=.05)
    p.add_argument('--multi_building_fraction',type=float,default=.10); p.add_argument('--allow_semantic_cc_fallback',action='store_true')
    a=p.parse_args()
    if 'test' in Path(a.val_list).name.lower(): raise RuntimeError('TEST ACCESS IS FORBIDDEN')
    device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=DamageFormerA1CrossModalSuperpixel('c5_aligned_assoc',a.region_stride)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); state=strip_module_prefix(extract_state(payload))
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint/model mismatch; refusing to guess. missing={missing[:20]}, unexpected={unexpected[:20]}')
    model.to(device).eval()
    ds=MultimodalDamageAssessmentDatset(a.dataset_path,read_list(a.val_list),640,type='val')
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,persistent_workers=a.num_workers>0)
    idx=InstanceIndex(a.instance_dir or None); cfg=AuditConfig(a.effective_region_threshold,a.multi_building_fraction)
    br=[]; rr=[]; source_counts={}; errors=[]; processed=0
    with torch.no_grad():
        for bi,batch in enumerate(loader):
            if a.max_batches and bi>=a.max_batches: break
            pre,post,_loc,damage,names=parse_batch(batch,device); _ol,_od,aux=model(pre,post,return_aux=True)
            probs=aux['association_probabilities']; ids=aux['candidate_ids']; h,w=aux['region_content'].shape[-2:]
            dfeat=F.interpolate(damage.unsqueeze(1).float(),size=(h,w),mode='nearest').squeeze(1).long()
            for i in range(pre.shape[0]):
                name=str(names[i]); dfull=damage[i].detach().cpu().numpy().astype('int32')
                inst,source,err=load_instance_map(name,dfull,idx); source_counts[source]=source_counts.get(source,0)+1
                if err: errors.append({'sample':name,'error':err})
                if source=='semantic_cc_fallback' and not a.allow_semantic_cc_fallback:
                    raise RuntimeError(f'No usable GT instance annotation for {name}. Re-run with --allow_semantic_cc_fallback only if explicitly accepted.')
                it=torch.from_numpy(inst).to(device); ifeat=F.interpolate(it[None,None].float(),size=(h,w),mode='nearest')[0,0].long()
                stats=association_dense_statistics(probs[i],ids,ifeat,dfeat[i])
                b_rows,r_rows=audit_one_image(name,stats,ifeat.cpu().numpy(),dfeat[i].cpu().numpy(),source,cfg)
                br.extend(b_rows); rr.extend(r_rows); processed+=1
            if (bi+1)%25==0: print(f'[audit] batches={bi+1} images={processed} buildings={len(br)} regions={len(rr)}',flush=True)
    sizes=add_size_bucket(br); summary=summarize(br,rr)
    summary.update({'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_step':int(payload.get('step',-1)) if isinstance(payload,dict) else -1,
      'region_stride':a.region_stride,'effective_region_threshold':a.effective_region_threshold,'multi_building_fraction':a.multi_building_fraction,
      'processed_images':processed,'instance_index':idx.describe(),'instance_source_counts':source_counts,'instance_error_count':len(errors),
      'size_bucket_thresholds_feature_pixels':sizes,'test_accessed':False})
    save_csv(out/'building_region_audit.csv',br); save_csv(out/'region_level_audit.csv',rr)
    (out/'damaged_evidence_coverage.json').write_text(json.dumps(summary['damaged_evidence_coverage'],indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'instance_errors.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
