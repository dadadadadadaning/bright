"""Stage A: region-path training; separate output from all previous experiments."""
import argparse
import math
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import SRACSwin, load_encoder_pretrain
from .common import *
from .losses import region_nll, pixel_ce
from .label_audit import validate_batch_labels, transfer_checked_batch

def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument('--pretrain')
    init.add_argument('--a1_checkpoint')
    p.add_argument('--base_lr',type=float,default=1e-5)
    p.add_argument('--new_lr',type=float,default=1e-4)
    for k,v in dict(steps=50000,eval_every=1000,batch=4,accum=4,workers=4,seed=3407,dim=128,stride=8).items(): p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--amp',choices=['bf16','none'],default='bf16')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--diagnostic_steps',type=int,default=0,
                   help='Stop early without a completion checkpoint; preserve full LR schedule')
    a=p.parse_args()
    if a.diagnostic_steps<0 or a.diagnostic_steps>=a.steps:
        p.error('diagnostic_steps must be 0 (disabled) or smaller than steps')
    if min(a.steps,a.eval_every,a.batch,a.accum,a.dim,a.stride)<1 or a.dim%4: p.error('Invalid dimensions')
    if a.batch*a.accum!=16 or (not a.smoke and a.steps!=50000): p.error('Require EBS16 and formal50000')
    init_path=a.a1_checkpoint or a.pretrain
    if not Path(init_path).is_file(): p.error('Missing initialization file')
    if min(a.base_lr,a.new_lr)<=0: p.error('Learning rates must be positive')
    if Path(a.output).exists(): p.error('Output exists; no overwrite/resume')
    train,val=lists(a.train_list,a.val_list)
    device=torch.device(a.device)
    if a.amp=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()): p.error('BF16 unavailable')
    seed_everything(a.seed)
    model=SRACSwin(a.dim,a.stride).to(device)
    from .a1_init import load_a1,optimizer_groups
    report=load_a1(model,a.a1_checkpoint) if a.a1_checkpoint else load_encoder_pretrain(model,a.pretrain)
    seed_everything(a.seed+1)
    tr=loader(a.dataset,train,True,a.batch,a.workers,seed=a.seed+2)
    vl=loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers,seed=a.seed+3)
    out=new_output(a.output)
    manifest=dict(arguments=vars(a),format='SRAC_SWIN_FRONT_V1',config=model.config,
                  source_hash=source_hash(),pretrain_sha256=digest(init_path),load_report=report,
                  train_sha256=digest(a.train_list),val_sha256=digest(a.val_list),test_accessed=False,
                  adaptation='shared RGB Swin; nested decoder; feature modulation; no hard building split; no Memory',
                  loss='loc CE+Lovasz + region NLL+Lovasz + 0.25*(pixel CE+Lovasz)',
                  init='A1 task fine-tuning; fresh region modules' if a.a1_checkpoint else 'generic Swin weights; fresh heads',smoke=a.smoke)
    save_json(out/'manifest.json',manifest); print(json.dumps(manifest,indent=2),flush=True)
    if a.a1_checkpoint:
        # Full Val gate for formal training. Smoke only verifies the small path.
        initial=evaluate_front(model,vl,device)
        observed=initial['metrics']['pixel']['valid_miou']
        passed=None if a.smoke else abs(observed-report['expected_valid_miou'])<=1e-5
        save_json(out/'initial_a1_audit.json',dict(**initial,reproduction_pass=passed,
                  expected=report['expected_valid_miou'],smoke=a.smoke))
        if passed is False:
            raise RuntimeError(f'A1 baseline mismatch: {observed}; no optimizer updates performed')
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax
    def ce_lov(logits,target):
        if not (target!=255).any(): return logits.sum()*0
        return pixel_ce(logits,target)+lovasz_softmax(logits.float().softmax(1),target,classes='present',ignore=255)
    groups=optimizer_groups(model,a.base_lr,a.new_lr) if a.a1_checkpoint else optimizer_groups(model,1e-4,1e-4)
    opt=torch.optim.AdamW(groups,weight_decay=.005)
    it=iter(tr); best=-1
    for step in range(1,a.steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        warm=min(1000,max(1,a.steps//10))
        factor=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,a.steps-warm)))
        for g in opt.param_groups: g['lr']=g['initial_lr']*factor
        logged=[0.,0.,0.]
        for microstep in range(a.accum):
            try: batch=next(it)
            except StopIteration: it=iter(tr); batch=next(it)
            label_report=validate_batch_labels(batch,step,microstep)
            save_json(out/'last_batch_labels.json',label_report)
            if not label_report['valid']:
                save_json(out/'invalid_labels.json',label_report)
                raise ValueError('Invalid CPU labels; inspect invalid_labels.json. No labels remapped.')
            pre,post,loc,gt,_=transfer_checked_batch(batch,device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=a.amp=='bf16'): r=model(pre,post)
            ll=ce_lov(r['loc'],loc); pl=ce_lov(r['pixel'],gt)
            rl=(region_nll(r['prob'],gt)+lovasz_softmax(r['prob'],gt,classes='present',ignore=255)) if (gt!=255).any() else r['prob'].sum()*0
            loss=(ll+rl+.25*pl)/a.accum
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            for j,x in enumerate([ll,rl,pl]): logged[j]+=float(x.detach())/a.accum
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); opt.step()
        if step==1 or step%20==0:
            row=dict(step=step,crops=step*16,loc_loss=logged[0],region_loss=logged[1],pixel_aux=logged[2],
                     lr=opt.param_groups[0]['lr'],lr_new=opt.param_groups[1]['lr'],grad_norm=float(grad),entropy=float(r['entropy'].detach()))
            with (out/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            result=evaluate_front(model,vl,device); score=result['metrics']['region']['valid_miou']
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**result))+'\n')
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,valid_miou=score,**manifest)
            if score>best: best=score; save_pt(out/'best.pth',state)
            if step==a.steps: save_pt(out/'last.pth',state)
            print(f'VAL step={step} region={score:.6f} pixel={result["metrics"]["pixel"]["valid_miou"]:.6f}',flush=True)
        if a.diagnostic_steps and step>=a.diagnostic_steps:
            save_json(out/'diagnostic_complete.json',dict(diagnostic_complete=True,step=step,
                      training_complete=False,test_accessed=False))
            print('DIAGNOSTIC COMPLETE; formal training NOT complete.',flush=True)
            return
    save_json(out/'summary.json',dict(complete=True,best_valid_miou=best,test_accessed=False,smoke=a.smoke))

if __name__=='__main__': main()
