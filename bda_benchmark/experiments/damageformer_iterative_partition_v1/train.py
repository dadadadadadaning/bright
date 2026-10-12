import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import PartitionModel, load_encoder_pretrain, VARIANTS


def worker_seed(worker):
    seed=torch.initial_seed()%2**32
    random.seed(seed); np.random.seed(seed)


def main():
    p=argparse.ArgumentParser('New protocol: joint training, not A1 checkpoint fine-tuning')
    for k in ('pretrain','dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    for k,v in dict(steps=50000,eval_every=1000,batch=4,accum=4,crop=512,workers=4,seed=3407,dim=64,stride=8,warmup=1000).items():
        p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--temperature',type=float,default=0.2)
    p.add_argument('--spatial_weight',type=float,default=0.1)
    p.add_argument('--smoke_val_images',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--schedule',choices=['constant','warmup_cosine'],default='warmup_cosine')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--amp',choices=['none','bf16'],default='bf16')
    a=p.parse_args()
    if min(a.steps,a.eval_every,a.batch,a.accum,a.crop,a.dim,a.stride)<1: p.error('Positive sizes required')
    if a.temperature<=0 or a.spatial_weight<0 or a.smoke_val_images<1: p.error('Invalid partition/smoke parameters')
    if a.warmup<0 or a.lr<=0: p.error('Invalid learning rate/warmup')
    if Path(a.output).exists(): p.error('Output already exists; choose a NEW directory (no overwrite/resume)')
    if a.batch*a.accum!=16: p.error('Protocol requires batch * accum = 16 on ONE GPU')
    if not a.smoke and a.steps!=50000: p.error('Formal protocol requires 50000 optimizer steps; use --smoke for short checks')
    if not Path(a.pretrain).is_file(): p.error('Missing real Swin-T pretrained file: '+a.pretrain)
    if Path(a.train_list).name!='train_set.txt' or Path(a.val_list).name!='val_set.txt': p.error('Train/Val only')
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import parse_batch,seed_everything,sha256_file,event_name
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax
    from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
    seed_everything(a.seed)
    device=torch.device(a.device)
    if a.amp=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()): p.error('BF16 CUDA unavailable; use --amp none')
    names=[[s.strip() for s in Path(path).read_text().splitlines() if s.strip()] for path in [a.train_list,a.val_list]]
    if len(names[0])!=2918 or len(names[1])!=450 or set(names[0])&set(names[1]): raise ValueError('Expected disjoint Train2918/Val450')
    if any(len(n)!=len(set(n)) for n in names): raise ValueError('Duplicate sample names')
    model=PartitionModel(a.variant,a.dim,a.stride,a.temperature,a.spatial_weight)
    load_report=load_encoder_pretrain(model,a.pretrain)
    model.to(device)
    # Reset RNG after variable-size model construction so augmentation/dropout streams match.
    seed_everything(a.seed+1)
    gen=torch.Generator().manual_seed(a.seed+2)
    train=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,names[0],a.crop,type='train'),
        batch_size=a.batch,shuffle=True,num_workers=a.workers,worker_init_fn=worker_seed,generator=gen,drop_last=True)
    val=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,names[1][:a.smoke_val_images] if a.smoke else names[1],1024,type='test'),
        batch_size=1,num_workers=a.workers,worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(a.seed+3))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    def save_json(name,obj): (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    manifest=dict(arguments=vars(a),pretrain_sha256=sha256_file(a.pretrain),load_report=load_report,
        train_sha256=sha256_file(a.train_list),val_sha256=sha256_file(a.val_list),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        effective_batch=a.batch*a.accum,total_planned_crops=a.steps*a.batch*a.accum,
        protocol='ITERATIVE_PARTITION_V1_IMAGE_PROTOCOL_V2',
        validation_image_count=min(a.smoke_val_images,450) if a.smoke else 450,
        memory_enabled=False,partition_inspiration='SSN-like local center refinement, NOT paper reproduction',
        grouping='predicted building/background, detached, NOT instances',
        feature_source='RGB+XY for partition; fused_roi for content',
        temperature=a.temperature,spatial_weight=a.spatial_weight,
        smoke=a.smoke,test_accessed=False,exist_head=False,
        augmentation='random crop, horizontal/vertical flip, 90-degree rotation',
        loss='localization CE+Lovasz plus final damage CE+Lovasz; no partition auxiliary loss',
        initialization='Swin-T feature weights only, fresh task/region heads',
        batchnorm='microbatch statistics; accumulation is not large-batch BatchNorm')
    import hashlib
    from . import model as model_source
    import inspect
    from bda_benchmark.experiments.damageformer_joint_region_v1.model import JointModel
    from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1 import association
    files=[__file__,model_source.__file__,inspect.getfile(JointModel),
           inspect.getfile(type(model.base)),association.__file__]
    manifest['source_sha256']={str(Path(f).resolve().relative_to(Path(__file__).resolve().parents[3])):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    save_json('manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=5e-3)
    def loss(logits,target):
        x=logits.float()
        if not (target!=255).any(): return x.sum()*0,x.sum()*0
        return F.cross_entropy(x,target,ignore_index=255),lovasz_softmax(x.softmax(1),target,classes='present',ignore=255)
    def checkpoint(name,step,score):
        tmp=out/(name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
            valid_miou=score,manifest=manifest),tmp)
        tmp.replace(out/name)
    iterator=iter(train); best=-1; best_step=0
    for step in range(1,a.steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        warm=min(a.warmup,max(1,a.steps//10))
        factor=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,a.steps-warm)))
        if a.schedule=='constant': factor=1.
        for group in opt.param_groups: group['lr']=a.lr*factor
        totals=[0.,0.,0.,0.]
        for micro in range(a.accum):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(train); batch=next(iterator)
            pre,post,loc,gt,_=parse_batch(batch,device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=a.amp=='bf16'):
                ol,od=model(pre,post)
            lc,lv=loss(ol,loc); dc,dv=loss(od,gt)
            total=(lc+lv+dc+dv)/a.accum
            if not torch.isfinite(total): raise RuntimeError('Nonfinite loss')
            total.backward()
            for j,value in enumerate((lc,lv,dc,dv)): totals[j]+=float(value.detach())/a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if step==1 or step%20==0:
            record=dict(step=step,microsteps=step*a.accum,crops_seen=step*a.accum*a.batch,
                total_loss=sum(totals),loc_loss=totals[0]+totals[1],cls_loss=totals[2]+totals[3],
                ce_loss=totals[0]+totals[2],lovasz_loss=totals[1]+totals[3],
                loc_ce=totals[0],loc_lovasz=totals[1],cls_ce=totals[2],cls_lovasz=totals[3],
                exist_loss=None,lr=opt.param_groups[0]['lr'])
            if model.head is not None:
                record['partition']={k:float(v) for k,v in model.head.diagnostics.items()}
                record['association_grad_norm']=float(sum(p.grad.float().square().sum() for p in model.head.association.parameters() if p.grad is not None).sqrt()) if any(p.grad is not None for p in model.head.association.parameters()) else None
            with (out/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            model.eval(); cm=torch.zeros(4,4,dtype=torch.long); events={}; loc_cm=torch.zeros(2,2,dtype=torch.long)
            with torch.inference_mode():
                for batch in val:
                    pre,post,loc,gt,ns=parse_batch(batch,device)
                    # FP32 evaluation, same for all variants.
                    loc_logits,pred=model(pre,post); pred=pred.argmax(1)
                    vl=(loc>=0)&(loc<2)
                    loc_cm+=torch.bincount((2*loc[vl]+loc_logits.argmax(1)[vl]).long(),minlength=4).reshape(2,2).cpu()
                    valid=(gt>=0)&(gt<4)
                    h=torch.bincount((4*gt[valid]+pred[valid]).long(),minlength=16).reshape(4,4).cpu()
                    cm+=h; ev=event_name(str(ns[0])); events.setdefault(ev,torch.zeros_like(cm)).add_(h)
            score=metrics(cm)['valid_miou']
            loc_f1=float(2*loc_cm[1,1]/(loc_cm[1].sum()+loc_cm[:,1].sum()).clamp_min(1))
            row=dict(step=step,loc_confusion=loc_cm.tolist(),loc_f1=loc_f1,metrics=metrics(cm),per_event={k:metrics(v) for k,v in events.items()})
            with (out/'validation.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
            if score>best: best,best_step=score,step; checkpoint('best.pth',step,score)
            if step==a.steps: checkpoint('last.pth',step,score)
            print(f'Val loc_F1={100*loc_f1:.4f}% foreground mIoU={100*score:.4f}% best={100*best:.4f}% step={best_step}',flush=True)
    save_json('summary.json',dict(best_valid_miou=best,best_step=best_step,optimizer_steps=a.steps,
        microsteps=a.steps*a.accum,crops_seen=a.steps*a.accum*a.batch,test_accessed=False))


if __name__=='__main__': main()
