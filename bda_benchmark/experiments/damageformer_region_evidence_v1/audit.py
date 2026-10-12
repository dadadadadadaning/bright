import argparse
import csv
import gzip
import torch
from .core import *
from .mixing import region_rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser('Fixed Q mixing and optical modulation interventions; no training/Test')
    for key in ('checkpoint','dataset','train_list','val_list','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    _,val=lists(a.train_list,a.val_list); out=new_output(a.output); device=torch.device(a.device)
    seed_everything(3407); model,ckpt,sha=load_fixed(a.checkpoint,device)
    modes=('normal','off','dim_only','edge_only'); cms={}; events={}; trans={}; buckets={}
    changes={}; total_regions=used=fragmented=crossed=0; soft_mass=purity_mass=0.; count=0
    writer=None
    with gzip.open(out/'regions.csv.gz','wt',newline='',encoding='utf-8') as stream:
        for batch in loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers):
            pre,post,_,gt,names=parse_batch(batch,device); base=extract(model,pre,post)
            ev=event(names[0]); events.setdefault(ev,{})
            preds={'pixel':base['pixel'].argmax(1),'normal':base['prob'].argmax(1)}
            reference=preds['normal']
            rows=region_rows(base,gt,preds['pixel'],reference)
            for row in rows:
                row=dict(image=names[0],event=ev,**row)
                if writer is None: writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                total_regions+=1; used+=int(row['count']>0)
                fragmented+=int(row['hard_feature_components']>1)
                crossed+=int(row['hard_gt_foreground_components']>1)
                soft_mass+=row['soft_valid_area']; purity_mass+=row['soft_valid_area']*(row['purity'] or 0)
                labels=[row['bucket']]
                if row['hard_gt_foreground_components']>1: labels.append('crosses_gt_component_proxy')
                for label in labels:
                    b=buckets.setdefault(label,dict(regions=0,count=0,base_correct=0,base_wrong=0,harm=0,rescue=0))
                    b['regions']+=1
                    for key in ('count','base_correct','base_wrong','harm','rescue'): b[key]+=row[key]
            for mode in modes[1:]:
                r=intervene(model,pre,base,mode)
                if not torch.equal(r['q'],base['q']) or not torch.equal(r['ids'],base['ids']):
                    raise RuntimeError('Q changed; intervention is not isolated')
                torch.testing.assert_close(r['pixel'],base['pixel'],rtol=0,atol=0)
                preds[mode]=r['prob'].argmax(1)
                delta=float((r['prob']-base['prob']).abs().mean())
                changes[mode]=changes.get(mode,0.)+delta
                del r
            ok=(gt>=0)&(gt<4)
            for key,pred in preds.items():
                cm_update(cms.setdefault(key,torch.zeros(4,4,dtype=torch.long)),gt,pred)
                cm_update(events[ev].setdefault(key,torch.zeros(4,4,dtype=torch.long)),gt,pred)
                if key!='pixel':
                    t=trans.setdefault(key,torch.zeros(4,4,4,dtype=torch.long))
                    t+=torch.bincount((gt[ok]*16+preds['pixel'][ok]*4+pred[ok]).long(),minlength=64).reshape(4,4,4).cpu()
            count+=1
            if count%10==0: print(f'audit {count}/{2 if a.smoke else 450}',flush=True)
    for row in buckets.values():
        row['harm_rate']=row['harm']/row['base_correct'] if row['base_correct'] else None
        row['rescue_rate']=row['rescue']/row['base_wrong'] if row['base_wrong'] else None
    save_json(out/'summary.json',dict(complete=True,arguments=vars(a),checkpoint_sha256=sha,checkpoint_step=ckpt['step'],
         frontend_source_hash=source_hash(),audit_source_hash=own_hash(),image_count=count,smoke=a.smoke,test_accessed=False,
         metrics={k:metrics(v) for k,v in cms.items()},per_event={e:{k:metrics(v) for k,v in rows.items()} for e,rows in events.items()},
         transition_axes=['GT','pixel_prediction','candidate_prediction'],transitions={k:v.tolist() for k,v in trans.items()},
         mixing_buckets=buckets,region_counts=dict(seed_regions=total_regions,hard_valid_used=used,hard_fragmented=fragmented,
         hard_cross_gt_components=crossed),pixel_weighted_soft_purity=purity_mass/soft_mass if soft_mass else None,
         probability_mean_abs_change={k:v/count for k,v in changes.items()},q_unchanged=True,
         intervention_protocol='cached Q/IDs/building/SAR; recompute only modulated optical content and region classification',
         caution='GT audit only; components are proxies; purity associations are not causal; interventions are not retrained ablations'))
    print({k:metrics(v)['valid_miou'] for k,v in cms.items()},flush=True)
    print('EVIDENCE AUDIT COMPLETE. No weights updated. No Test.',flush=True)

if __name__=='__main__': main()
