"""Same frozen modulated feature/Q, equal-capacity independent fresh classifiers."""
import argparse
import copy
import json
import math
import torch
from .core import *
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import validate_batch_labels
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax


def objective(prob,gt):
    if not (gt!=255).any(): return prob.sum()*0
    return region_nll(prob,gt)+lovasz_softmax(prob,gt,classes='present',ignore=255)


@torch.no_grad()
def evaluate(model,heads,data,device):
    model.eval()
    for head in heads.values(): head.eval()
    cms={}; count=0
    for batch in data:
        pre,post,_,gt,_=parse_batch(batch,device); r=extract(model,pre,post)
        probs={'old_pixel':r['pixel'].softmax(1),'old_region':r['prob']}
        for name,head in heads.items(): probs[name]=head(r['region_content'],r,name,gt.shape[-2:])
        for name,p in probs.items(): cm_update(cms.setdefault(name,torch.zeros(4,4,dtype=torch.long)),gt,p.argmax(1))
        count+=len(pre)
        if count%25==0: print(f'evaluate {count}',flush=True)
    return dict(image_count=count,metrics={k:metrics(cm) for k,cm in cms.items()},test_accessed=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('checkpoint','dataset','train_list','val_list','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=5000); p.add_argument('--eval_every',type=int,default=500)
    p.add_argument('--batch',type=int,default=2); p.add_argument('--accum',type=int,default=8)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--dim',type=int,default=128)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if min(a.steps,a.eval_every,a.batch,a.accum,a.dim,a.lr)<=0 or a.batch*a.accum!=16: p.error('Positive settings and EBS16 required')
    train,val=lists(a.train_list,a.val_list); out=new_output(a.output); device=torch.device(a.device)
    seed_everything(a.seed); model,ckpt,sha=load_fixed(a.checkpoint,device)
    proto=MatchedHead(dim=a.dim).to(device)
    heads={k:copy.deepcopy(proto) for k in ('pixel','region')}; del proto
    opts={k:torch.optim.AdamW(h.parameters(),lr=a.lr,weight_decay=.005) for k,h in heads.items()}
    manifest=dict(arguments=vars(a),format='REGION_FEATURE_PROBE_V1',checkpoint_sha256=sha,checkpoint_step=ckpt['step'],
                  frontend_source_hash=source_hash(),probe_source_hash=own_hash(),test_accessed=False,
                  train_sha256=digest(a.train_list),val_sha256=digest(a.val_list),
                  feature_source='same second fusion_layer_3 output (modulated RGB + unchanged SAR)',
                  frontend_frozen=True,bn_eval=True,head_parameters={k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()},
                  loss='identical full-image NLL+Lovasz; fresh pointwise MLP; region uses Q mean pooling, not V2 attention/stats')
    save_json(out/'manifest.json',manifest)
    tr=loader(a.dataset,train,True,a.batch,a.workers,seed=a.seed+2)
    vl=loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers,seed=a.seed+3)
    initial=evaluate(model,heads,vl,device); save_json(out/'initial.json',initial)
    if not a.smoke and abs(initial['metrics']['old_region']['valid_miou']-ckpt['valid_miou'])>1e-5:
        raise RuntimeError('Fixed frontend baseline reproduction failed')
    best={k:-1. for k in heads}; it=iter(tr)
    for step in range(1,a.steps+1):
        model.eval(); warm=min(100,max(1,a.steps//10))
        factor=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,a.steps-warm)))
        totals={k:0. for k in heads}
        for k,h in heads.items():
            h.train(); opts[k].zero_grad(set_to_none=True)
            for g in opts[k].param_groups: g['lr']=a.lr*factor
        for micro in range(a.accum):
            try: batch=next(it)
            except StopIteration: it=iter(tr); batch=next(it)
            if not validate_batch_labels(batch,step,micro)['valid']: raise ValueError('Invalid labels')
            pre,post,_,gt,_=parse_batch(batch,device)
            r=extract(model,pre,post)  # no_grad, BN/dropout eval; no Train-feature cache required
            for k,h in heads.items():
                prob=h(r['region_content'],r,k,gt.shape[-2:]); loss=objective(prob,gt)/a.accum
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite probe loss')
                loss.backward(); totals[k]+=float(loss.detach())
        for k,h in heads.items():
            torch.nn.utils.clip_grad_norm_(h.parameters(),1,error_if_nonfinite=True); opts[k].step()
        if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Frozen frontend received gradients')
        if step==1 or step%20==0:
            row=dict(step=step,crops=16*step,losses=totals,lr=a.lr*factor)
            with (out/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(row,flush=True)
        if step%a.eval_every==0 or step==a.steps:
            result=evaluate(model,heads,vl,device)
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**result))+'\n')
            for k,h in heads.items():
                score=result['metrics'][k]['valid_miou']
                state=dict(head=h.state_dict(),optimizer=opts[k].state_dict(),step=step,mode=k,metrics=result['metrics'][k],manifest=manifest)
                if score>best[k]: best[k]=score; save_pt(out/(k+'_best.pth'),state)
                if step==a.steps: save_pt(out/(k+'_last.pth'),state)
            print('VAL',step,{k:v['valid_miou'] for k,v in result['metrics'].items()},flush=True)
    save_json(out/'summary.json',dict(complete=True,best=best,smoke=a.smoke,test_accessed=False))

if __name__=='__main__': main()
