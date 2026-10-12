"""Stage B: full-image deterministic Train shards and class/event reservoir."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import *
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool

def region_targets(gt,r):
    valid=(gt>=0)&(gt<4)
    one=F.one_hot(gt.clamp(0,3),4).permute(0,3,1,2).float()*valid[:,None]
    fractions=F.interpolate(one,size=r['shape'],mode='area')
    pooled=soft_pool_unpool(fractions,r['q'],r['ids'],r['tokens'].shape[1])
    counts=pooled['tokens']*pooled['mass'][:,:,None]
    weight=counts.sum(-1)
    return counts/weight.clamp_min(1e-8)[:,:,None],weight

def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('checkpoint','dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--per_bucket',type=int,default=64); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=3407); p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.per_bucket<1: p.error('Positive per_bucket required')
    train,val=lists(a.train_list,a.val_list)
    device=torch.device(a.device); seed_everything(a.seed)
    model,front=load_front(a.checkpoint,device); model.eval().requires_grad_(False)
    if bool(front['smoke'])!=a.smoke: p.error('Do not mix smoke and formal stages')
    if front['train_sha256']!=digest(a.train_list) or front['val_sha256']!=digest(a.val_list):
        p.error('Checkpoint data split changed')
    if a.smoke:
        chosen=[]; seen=set()
        for name in train:
            if event(name) not in seen: chosen.append(name); seen.add(event(name))
            if len(chosen)==4: break
        train=chosen
    out=new_output(a.output); (out/'shards').mkdir()
    events=sorted({event(n) for n in train})
    meta=dict(complete=False,split='train',test_accessed=False,smoke=a.smoke,
              checkpoint_sha256=digest(a.checkpoint),source_hash=source_hash(),
              train_sha256=digest(a.train_list),val_sha256=digest(a.val_list),
              config=front['config'],event_names=events,image_count=len(train),
              per_bucket=a.per_bucket,seed=a.seed,shards=[],
              gt_role='Training labels/distributions only; no GT partition or query mask',
              extraction='FP32 full-image1024, deterministic; no Train random crop',
              region_labels='area-downsampled four-class label fractions; includes background',
              geometry='not injected; no query GT-based region selection')
    save_json(out/'manifest.json',meta)
    reservoir={}; rng=torch.Generator().manual_seed(a.seed)
    with torch.inference_mode():
        for i,batch in enumerate(loader(a.dataset,train,workers=a.workers)):
            pre,post,loc,gt,names=parse_batch(batch,device); r=model(pre,post)
            dist,weights=region_targets(gt,r)
            tokens=r['tokens'][0].cpu(); dist=dist[0].cpu(); weights=weights[0].cpu()
            ev=events.index(event(names[0]))
            shard=dict(tokens=tokens,distribution=dist,weights=weights,event=ev,image=i,name=names[0],
                       region_ids=torch.arange(len(tokens)),mass=r['mass'][0].cpu())
            filename=f'shards/{i:05d}.pt'; save_pt(out/filename,shard)
            meta['shards'].append(dict(path=filename,sha256=digest(out/filename)))
            for cls in range(4):
                ids=((dist.argmax(-1)==cls)&(weights>0)).nonzero().flatten()
                if not len(ids): continue
                new=dict(tokens=tokens[ids].clone(),distribution=dist[ids].clone(),
                         events=torch.full((len(ids),),ev),images=torch.full((len(ids),),i),
                         regions=ids.clone(),priority=torch.rand(len(ids),generator=rng))
                key=(ev,cls)
                if key in reservoir:
                    new={k:torch.cat([reservoir[key][k],v]) for k,v in new.items()}
                keep=new['priority'].topk(min(a.per_bucket,len(new['tokens']))).indices
                reservoir[key]={k:v[keep].clone() for k,v in new.items()}
            print(f'cache {i+1}/{len(train)} regions={len(tokens)}',flush=True)
    if not reservoir: raise ValueError('No labelled Train regions; cache incomplete')
    bank={k:torch.cat([v[k] for _,v in sorted(reservoir.items())]) for k in next(iter(reservoir.values())) if k!='priority'}
    bank.update(event_names=events,checkpoint_sha256=meta['checkpoint_sha256'])
    save_pt(out/'bank.pt',bank)
    meta.update(complete=True,bank_sha256=digest(out/'bank.pt'),bank_regions=len(bank['tokens']))
    save_json(out/'manifest.json',meta)
    print('TRAIN MEMORY COMPLETE. No Val/Test entered bank.',flush=True)

if __name__=='__main__': main()
