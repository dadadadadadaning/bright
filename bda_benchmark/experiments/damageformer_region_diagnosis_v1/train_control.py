"""Short matched-start controls; 50k LR horizon preserved, no resume/Test."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.a1_init import load_a1,optimizer_groups
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.common import (
    lists,loader,new_output,save_json,save_pt,seed_everything,parse_batch,cm_update,metrics,source_hash,digest)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import validate_batch_labels
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll,pixel_ce
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax


def pixel_forward(model,pre,post):
    base=model.base
    rp=[base._to_nchw(f) for f in base.encoder_1(pre)[1:]]
    sp=[base._to_nchw(f) for f in base.encoder_2(post)[1:]]
    rgb=base.fusion_layer_1(base._multi_scale_fuse(*rp))
    sar=base.fusion_layer_2(base._multi_scale_fuse(*sp))
    loc=base.clf_1(rgb)
    b=loc.float().softmax(1)[:,1:2].detach()
    pixel=base.clf_2(base.fusion_layer_3(torch.cat([rgb,sar],1))*(.1+.9*b))
    return {k:F.interpolate(v,pre.shape[-2:],mode='bilinear',align_corners=False)
            for k,v in dict(loc=loc,pixel=pixel).items()}


@torch.inference_mode()
def validate(model,data,device,mode):
    model.eval(); cms={}; count=0
    for batch in data:
        pre,post,_,gt,_=parse_batch(batch,device)
        r=pixel_forward(model,pre,post) if mode=='a1' else model(pre,post)
        outputs={'pixel':r['pixel']}
        if mode!='a1': outputs['region']=r['prob']
        for k,v in outputs.items(): cm_update(cms.setdefault(k,torch.zeros(4,4,dtype=torch.long)),gt,v.argmax(1))
        count+=len(pre)
        if count%25==0: print(f'evaluate {count}',flush=True)
    return dict(image_count=count,metrics={k:metrics(v) for k,v in cms.items()},test_accessed=False)


def ce_lov(logits,target):
    if not (target!=255).any(): return logits.sum()*0
    return pixel_ce(logits,target)+lovasz_softmax(logits.float().softmax(1),target,classes='present',ignore=255)


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('a1','dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['a1','joint','joint_frozen_bn'],required=True)
    p.add_argument('--steps',type=int,default=5000); p.add_argument('--eval_every',type=int,default=1000)
    p.add_argument('--batch',type=int,default=2); p.add_argument('--accum',type=int,default=8)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--base_lr',type=float,default=1e-5); p.add_argument('--new_lr',type=float,default=1e-4)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--amp',choices=['bf16','none'],default='bf16')
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if a.batch*a.accum!=16 or not 1<=a.steps<=50000 or a.eval_every<1: p.error('EBS16, 1..50000 steps required')
    train,val=lists(a.train_list,a.val_list); device=torch.device(a.device)
    if a.amp=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()): p.error('BF16 unsupported')
    out=new_output(a.output); seed_everything(a.seed)
    # Construct identical initial model in all arms; unused modules frozen for A1.
    model=SRACSwin().to(device); report=load_a1(model,a.a1)
    if a.mode=='a1':
        for n,v in model.named_parameters():
            if not n.startswith('base.'): v.requires_grad_(False)
    seed_everything(a.seed+1)
    tr=loader(a.dataset,train,True,a.batch,a.workers,seed=a.seed+2)
    vl=loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers,seed=a.seed+3)
    manifest=dict(arguments=vars(a),format='REGION_DIAG_CONTROL_V1',load_report=report,
                  frontend_source_hash=source_hash(),trainer_sha256=digest(__file__),config=model.config,
                  train_sha256=digest(a.train_list),val_sha256=digest(a.val_list),
                  lr_horizon=50000,warmup=1000,test_accessed=False,
                  loss='loc+pixel' if a.mode=='a1' else 'loc+region+0.25*pixel',
                  note='A1 control changes objective and architecture; not a pure gradient-conflict test')
    save_json(out/'manifest.json',manifest)
    initial=validate(model,vl,device,a.mode); save_json(out/'initial.json',initial)
    if not a.smoke and abs(initial['metrics']['pixel']['valid_miou']-report['expected_valid_miou'])>1e-5:
        raise RuntimeError('Initial A1 reproduction failed; no training performed')
    groups=([dict(params=[v for v in model.base.parameters() if v.requires_grad],lr=a.base_lr,initial_lr=a.base_lr)]
            if a.mode=='a1' else optimizer_groups(model,a.base_lr,a.new_lr))
    opt=torch.optim.AdamW(groups,weight_decay=.005); it=iter(tr); best=-1
    target='pixel' if a.mode=='a1' else 'region'
    for step in range(1,a.steps+1):
        model.train()
        if a.mode=='joint_frozen_bn':
            # Freeze running stats only, not affine parameters, at the A1 initialization.
            for n,m in model.named_modules():
                if isinstance(m,torch.nn.BatchNorm2d) and (n.startswith('base.fusion_layer_1.') or n.startswith('base.fusion_layer_3.')): m.eval()
        opt.zero_grad(set_to_none=True)
        factor=step/1000 if step<=1000 else .5*(1+math.cos(math.pi*(step-1000)/49000))
        for g in opt.param_groups: g['lr']=g['initial_lr']*factor
        total=0.
        for micro in range(a.accum):
            try: batch=next(it)
            except StopIteration: it=iter(tr); batch=next(it)
            if not validate_batch_labels(batch,step,micro)['valid']: raise ValueError('Illegal labels')
            pre,post,loc,gt,_=parse_batch(batch,device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=a.amp=='bf16'):
                r=pixel_forward(model,pre,post) if a.mode=='a1' else model(pre,post)
            ll=ce_lov(r['loc'],loc); pl=ce_lov(r['pixel'],gt)
            if a.mode=='a1': loss=ll+pl
            else:
                rl=(region_nll(r['prob'],gt)+lovasz_softmax(r['prob'],gt,classes='present',ignore=255)) if (gt!=255).any() else r['prob'].sum()*0
                loss=ll+rl+.25*pl
            loss=loss/a.accum
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); total+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); opt.step()
        if step==1 or step%20==0:
            row=dict(step=step,crops=step*16,loss=total,grad_norm=float(grad),lr=opt.param_groups[0]['lr'])
            with (out/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(row,flush=True)
        if step%a.eval_every==0 or step==a.steps:
            result=validate(model,vl,device,a.mode); score=result['metrics'][target]['valid_miou']
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**result))+'\n')
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,**manifest)
            if score>best: best=score; save_pt(out/'best.pth',state)
            if step==a.steps: save_pt(out/'last.pth',state)
            print('VAL',step,{k:v['valid_miou'] for k,v in result['metrics'].items()},flush=True)
    save_json(out/'summary.json',dict(complete=True,steps=a.steps,best_valid_miou=best,smoke=a.smoke,test_accessed=False))

if __name__=='__main__': main()
