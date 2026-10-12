#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import segmentation_loss
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.core import atomic_torch_save, extract_state, load_a1_parent, parse_batch, seed_everything, seed_worker, write_json
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.losses import region_losses
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.train import evaluate, format_metrics, read_list
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.variants import get_variant
from .model import DamageFormerA1JointSoftPartition, FIXED_VARIANT

def main():
    p=argparse.ArgumentParser('Joint A1 + soft partition')
    p.add_argument('--dataset_path',required=True); p.add_argument('--train_list',required=True); p.add_argument('--val_list',required=True)
    p.add_argument('--a1_init_ckpt',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--resume',default='')
    p.add_argument('--optimizer_steps',type=int,default=20000); p.add_argument('--crop_size',type=int,default=640)
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--accumulation_steps',type=int,default=8); p.add_argument('--eval_batch_size',type=int,default=2); p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--a1_lr',type=float,default=2e-5); p.add_argument('--plugin_lr',type=float,default=1e-4); p.add_argument('--weight_decay',type=float,default=5e-3)
    p.add_argument('--eval_every',type=int,default=500); p.add_argument('--save_every',type=int,default=1000); p.add_argument('--log_every',type=int,default=20); p.add_argument('--val_max_batches',type=int,default=0)
    p.add_argument('--ignore_index',type=int,default=255); p.add_argument('--amp_dtype',choices=('none','fp16','bf16'),default='bf16'); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=3407); p.add_argument('--region_stride',type=int,default=8)
    a=p.parse_args()
    if any('test' in Path(x).name.lower() for x in (a.train_list,a.val_list)): raise RuntimeError('TEST ACCESS IS FORBIDDEN')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'training_summary.json').exists() and not a.resume: raise RuntimeError(f'completed output exists: {out}')
    device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); gen=seed_everything(a.seed); spec=get_variant(FIXED_VARIANT)
    model=DamageFormerA1JointSoftPartition(a.region_stride).to(device); init_report=load_a1_parent(model,a.a1_init_ckpt); model.enable_joint_training()
    parent=[p for n,p in model.named_parameters() if not n.startswith('cmsp.') and p.requires_grad]; plugin=[p for n,p in model.named_parameters() if n.startswith('cmsp.') and p.requires_grad]
    if not parent or not plugin: raise RuntimeError('expected both A1 and cmsp trainable groups')
    opt=torch.optim.AdamW([{'params':parent,'lr':a.a1_lr},{'params':plugin,'lr':a.plugin_lr}],weight_decay=a.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,a.optimizer_steps)); start=0; best={}
    if a.resume:
        payload=torch.load(a.resume,map_location='cpu',weights_only=False); model.load_state_dict(extract_state(payload),strict=True); opt.load_state_dict(payload['optimizer']); sched.load_state_dict(payload['scheduler']); start=int(payload['step']); best=dict(payload.get('best_metrics',{}))
    train_set=MultimodalDamageAssessmentDatset(a.dataset_path,read_list(a.train_list),a.crop_size,type='train'); val_set=MultimodalDamageAssessmentDatset(a.dataset_path,read_list(a.val_list),a.crop_size,type='val')
    train_loader=DataLoader(train_set,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=device.type=='cuda',drop_last=True,worker_init_fn=seed_worker,generator=gen,persistent_workers=a.num_workers>0)
    val_loader=DataLoader(val_set,batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,persistent_workers=a.num_workers>0)
    loc_ce=nn.CrossEntropyLoss(ignore_index=a.ignore_index); dam_ce=nn.CrossEntropyLoss(ignore_index=a.ignore_index)
    amp=device.type=='cuda' and a.amp_dtype!='none'; amp_dtype=torch.float16 if a.amp_dtype=='fp16' else torch.bfloat16; scaler=torch.amp.GradScaler('cuda',enabled=amp and a.amp_dtype=='fp16')
    print(json.dumps({'protocol':'DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1','initialization':init_report,'optimizer_steps':a.optimizer_steps,'a1_lr':a.a1_lr,'plugin_lr':a.plugin_lr,'a1_frozen':False,'test_accessed':False},ensure_ascii=False,indent=2),flush=True)
    if start==0 and not (out/'initial_metrics.json').exists():
        m0=evaluate(model,val_loader,device,a.ignore_index,a.val_max_batches); print('Initial A1-equivalent Validation | '+format_metrics(m0),flush=True); write_json(out/'initial_metrics.json',m0)
    best_value=float(best.get('valid_miou',-1)); step=start; micro=0; running:Dict[str,float]={}; opt.zero_grad(set_to_none=True); t0=time.time()
    while step<a.optimizer_steps:
        for batch in train_loader:
            model.set_joint_training(); pre,post,location,damage,_=parse_batch(batch,device)
            with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=amp):
                ol,od,aux=model(pre,post,return_aux=True); loc_loss,_,_=segmentation_loss(ol,location,loc_ce,True,1.0,1.0,a.ignore_index); damage_loss,_,_=segmentation_loss(od,damage,dam_ce,True,1.0,1.0,a.ignore_index); regions=region_losses(aux,location,False); total=loc_loss+damage_loss+regions['total']; scaled=total/a.accumulation_steps
            scaler.scale(scaled).backward(); micro+=1
            vals={'total':total,'loc':loc_loss,'damage':damage_loss,**{f'region_{k}':v for k,v in regions.items()}}
            for k,v in vals.items(): running[k]=running.get(k,0.0)+float(v.detach())
            if micro%a.accumulation_steps: continue
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(parent+plugin,5.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step+=1
            if step%a.log_every==0:
                d=float(a.log_every*a.accumulation_steps); rec={'step':step,**{k:v/d for k,v in running.items()},'region_gain':float(torch.tanh(model.cmsp.region_gain).detach()),'lca_gain':float(torch.tanh(model.cmsp.lca.gain).detach()),'lr_a1':float(opt.param_groups[0]['lr']),'lr_cmsp':float(opt.param_groups[1]['lr']),'steps_per_second':step/max(1e-6,time.time()-t0)}; print(json.dumps(rec),flush=True); (out/'history.jsonl').open('a',encoding='utf-8').write(json.dumps(rec)+'\n'); running.clear()
            if step%a.eval_every==0 or step==a.optimizer_steps:
                m=evaluate(model,val_loader,device,a.ignore_index,a.val_max_batches); print('Validation | '+format_metrics(m),flush=True); payload={'protocol':'DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1','step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'best_metrics':best,'contract':model.checkpoint_contract(),'init_report':init_report}; atomic_torch_save(payload,out/'latest.pth')
                if float(m['valid_miou'])>best_value: best_value=float(m['valid_miou']); best=m; payload['best_metrics']=best; atomic_torch_save(payload,out/'best_val_miou.pth')
            if step%a.save_every==0: atomic_torch_save({'protocol':'DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1','step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'best_metrics':best},out/f'step_{step:06d}.pth')
            if step>=a.optimizer_steps: break
    summary={'protocol':'DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1','completed_steps':step,'best_metrics':best,'init_report':init_report,'a1_frozen':False,'test_accessed':False}; write_json(out/'training_summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
