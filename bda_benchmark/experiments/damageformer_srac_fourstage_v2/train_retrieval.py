"""Stage C: frozen tokens; independent same-parameter memory/no-memory runs."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import *
from .retrieval import SupportClassifier,retrieve
from .build_neighbors import open_table,gather_support

def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('checkpoint','cache','dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['memory','nomemory'],required=True)
    p.add_argument('--neighbors',help='Required frozen mmap table for memory training')
    for k,v in dict(steps=5000,eval_every=500,batch=128,topk=8,workers=4,seed=3407).items(): p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.eval_every,a.batch,a.topk)<1: p.error('Positive sizes required')
    train,val=lists(a.train_list,a.val_list)
    root,meta,bank=read_cache(a.cache,a.checkpoint)
    table=table_meta=None
    if a.mode=='memory':
        if not a.neighbors: p.error('--neighbors required for memory')
        table,table_meta=open_table(a.neighbors,root,meta['bank_sha256'],a.topk)
    if meta['train_sha256']!=digest(a.train_list) or meta['val_sha256']!=digest(a.val_list): p.error('Data split hash mismatch')
    if bool(meta['smoke'])!=a.smoke: p.error('Do not mix smoke/formal cache')
    # Verify every shard once before training; payload loading thereafter is bounded.
    for s in meta['shards']:
        if digest(root/s['path'])!=s['sha256']: raise ValueError('Shard hash mismatch: '+s['path'])
    device=torch.device(a.device); seed_everything(a.seed)
    model,front=load_front(a.checkpoint,device); model.eval().requires_grad_(False)
    bank=device_bank(bank,device)
    head=SupportClassifier(meta['config']['dim'],len(meta['event_names'])).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.005)
    rng=torch.Generator().manual_seed(a.seed+1)
    vl=loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers)
    out=new_output(a.output)
    manifest=dict(arguments=vars(a),format='SRAC_FOURSTAGE_SUPPORT_V2',config=meta['config'],
                  checkpoint_sha256=meta['checkpoint_sha256'],bank_sha256=meta['bank_sha256'],
                  source_hash=source_hash(),mode=a.mode,topk=a.topk,smoke=a.smoke,test_accessed=False,
                  loss='area-weighted soft-label region CE; full-image Val checkpoint selection',
                  sampling='uniform image then uniform labelled region; background included',
                  exclusion='same image OR same event forbidden',geometry_injected=True,
                  neighbor_table_sha256=table_meta['table_sha256'] if table_meta else None,
                  metadata='support soft labels, disaster type, Train event region proxy, predicted geometry')
    save_json(out/'manifest.json',manifest); best=-1
    for step in range(1,a.steps+1):
        # Label mask only selects supervised TRAIN examples, never inference regions.
        for attempt in range(100):
            sid=int(torch.randint(len(meta['shards']),(1,),generator=rng))
            shard=torch.load(root/meta['shards'][sid]['path'],map_location='cpu',weights_only=False)
            eligible=(shard['weights']>0).nonzero().flatten()
            if len(eligible): break
        else: raise ValueError('100 sampled images have no labels')
        ids=eligible[torch.randint(len(eligible),(a.batch,),generator=rng)]
        q=shard['tokens'][ids].to(device); target=shard['distribution'][ids].to(device)
        weights=shard['weights'][ids].to(device)
        head.train()
        if a.mode=='memory':
            start=table_meta['offsets'][sid]['start']
            selected=torch.from_numpy(table[start+ids.numpy()].copy()).long().to(device)
            support,dist,valid,support_meta=gather_support(selected,bank)
            logits=head(q,support,dist,valid,metadata=support_meta)
            coverage=float(valid.any(1).float().mean())
        else:
            logits=head(q,mode='nomemory'); coverage=None
        loss=(-(target*F.log_softmax(logits.float(),-1)).sum(-1)*weights).sum()/weights.sum().clamp_min(1e-8)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite head loss')
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True)
        for g in opt.param_groups: g['lr']=1e-4*.5*(1+math.cos(math.pi*(step-1)/a.steps))
        opt.step()
        if step==1 or step%20==0:
            row=dict(step=step,loss=float(loss.detach()),support_coverage=coverage)
            with (out/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            result=evaluate_front(model,vl,device,head,bank,a.mode,a.topk)
            score=result['metrics']['region']['valid_miou']
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**result))+'\n')
            state=dict(model=head.state_dict(),optimizer=opt.state_dict(),step=step,valid_miou=score,**manifest)
            if score>best: best=score; save_pt(out/'best.pth',state)
            if step==a.steps: save_pt(out/'last.pth',state)
            print(f'VAL {a.mode} step={step} full-image foreground mIoU={score:.6f}',flush=True)
    save_json(out/'summary.json',dict(complete=True,best_valid_miou=best,test_accessed=False,smoke=a.smoke))

if __name__=='__main__': main()
