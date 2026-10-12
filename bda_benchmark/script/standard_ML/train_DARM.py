#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
BDA_ROOT = (SCRIPT_DIR / "../..").resolve()
for p in (str(BDA_ROOT), str(BDA_ROOT / "model"), str(SCRIPT_DIR)):
    if p not in sys.path: sys.path.insert(0, p)

from model.DamageFormer_DARM import DamageFormer_DARM
from model.region_ops import build_instance_map_from_binary
from darm_losses import hierarchical_instance_loss, intact_alignment_loss, ramp_weight


def read_list(path: str) -> List[str]:
    return [x.strip() for x in open(path, encoding="utf-8") if x.strip()]


def build_dataset(root: str, list_path: str, size: int, split: str):
    from dataset.make_data_loader import MultimodalDamageAssessmentDatset
    return MultimodalDamageAssessmentDatset(root, read_list(list_path), size, None, split)


def parse_batch(batch, device):
    if isinstance(batch, dict):
        def pick(keys: Iterable[str]):
            for k in keys:
                if k in batch: return batch[k]
            raise KeyError(keys)
        pre=pick(["pre_img","pre_data","pre","image_pre"])
        post=pick(["post_img","post_data","post","image_post"])
        loc=pick(["label_loc","labels_loc","loc","building_mask"])
        dam=pick(["label_dam","labels_clf","damage","label_cls"])
    else:
        if len(batch)==5: pre,post,loc,dam,_=batch
        elif len(batch)==6: pre,post,_,loc,dam,_=batch
        else: raise ValueError(f"Unsupported batch length {len(batch)}")
    pre,post=pre.to(device).float(),post.to(device).float()
    loc,dam=loc.to(device).long(),dam.to(device).long()
    if loc.ndim==4: loc=loc[:,0]
    if dam.ndim==4: dam=dam[:,0]
    return pre,post,loc,dam


def lovasz_grad(gt_sorted):
    p=len(gt_sorted); gts=gt_sorted.sum(); inter=gts-gt_sorted.cumsum(0); union=gts+(1-gt_sorted).cumsum(0)
    j=1-inter/union.clamp_min(1e-6)
    if p>1: j[1:]-=j[:-1]
    return j


def lovasz_softmax(probas, labels, ignore=255):
    c=probas.shape[1]
    p=probas.permute(0,2,3,1).reshape(-1,c); y=labels.reshape(-1)
    valid=y!=ignore; p,y=p[valid],y[valid]
    losses=[]
    for cls in range(c):
        fg=(y==cls).float()
        if fg.sum()==0: continue
        err=(fg-p[:,cls]).abs(); es,perm=torch.sort(err,descending=True)
        losses.append(torch.dot(es,lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else probas.sum()*0


def hist_update(hist, pred, gt, n, ignore=255):
    m=(gt!=ignore)&(gt>=0)&(gt<n)
    x=n*gt[m]+pred[m]
    hist += torch.bincount(x, minlength=n*n).reshape(n,n).double().cpu()


def metrics(hist):
    inter=torch.diag(hist); union=hist.sum(1)+hist.sum(0)-inter
    iou=inter/union.clamp_min(1)
    return {"iou":iou.tolist(),"miou4":float(iou.mean()),"miou3":float(iou[1:4].mean())}

@torch.no_grad()
def validate(model, loader, device, min_instance_area=8):
    model.eval(); lh=torch.zeros(2,2,dtype=torch.float64); dh=torch.zeros(4,4,dtype=torch.float64)
    inst_gt=inst_pred=inst_correct=0
    for batch in loader:
        pre,post,loc,dam=parse_batch(batch,device)
        inst=build_instance_map_from_binary(loc==1,min_instance_area)
        out=model(pre,post,instance_map=inst,damage_gt=dam,update_memory=False)
        pl=out["output_loc"].argmax(1); pd=out["output_dam"].argmax(1)
        hist_update(lh,pl,loc,2); hist_update(dh,pd,dam,4)
        if out["instance_labels"].numel():
            ip=out["building_logits3"].argmax(-1)+1; iy=out["instance_labels"]
            inst_gt += int(iy.numel()); inst_pred += int((ip==3).sum()); inst_correct += int((ip==iy).sum())
    m=metrics(dh); m.update({"instance_acc":inst_correct/max(inst_gt,1),"destroyed_pred_instances":inst_pred,"loc_hist":lh.tolist(),"dam_hist":dh.tolist()})
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_dataset_path",required=True); ap.add_argument("--train_data_list_path",required=True)
    ap.add_argument("--val_dataset_path",required=True); ap.add_argument("--val_data_list_path",required=True)
    ap.add_argument("--output_root",required=True); ap.add_argument("--exp_name",default="DARM")
    ap.add_argument("--pretrained_model_path",default="")
    ap.add_argument("--image_size",type=int,default=640); ap.add_argument("--train_batch_size",type=int,default=2); ap.add_argument("--eval_batch_size",type=int,default=1)
    ap.add_argument("--accumulation_steps",type=int,default=8); ap.add_argument("--max_steps",type=int,default=50000); ap.add_argument("--num_workers",type=int,default=4)
    ap.add_argument("--learning_rate",type=float,default=1e-4); ap.add_argument("--weight_decay",type=float,default=5e-3); ap.add_argument("--seed",type=int,default=3407)
    ap.add_argument("--amp_dtype",choices=["none","fp16","bf16"],default="bf16"); ap.add_argument("--ignore_index",type=int,default=255)
    ap.add_argument("--lambda_inst",type=float,default=0.2); ap.add_argument("--inst_warmup",type=int,default=3000); ap.add_argument("--inst_ramp",type=int,default=3000)
    ap.add_argument("--lambda_ia",type=float,default=0.005); ap.add_argument("--ia_warmup",type=int,default=8000); ap.add_argument("--ia_ramp",type=int,default=2000)
    ap.add_argument("--topk_ratio",type=float,default=0.1); ap.add_argument("--boundary_radius",type=int,default=1); ap.add_argument("--ring_radius",type=int,default=3); ap.add_argument("--min_instance_area",type=int,default=8)
    ap.add_argument("--use_prototypes",action="store_true"); ap.add_argument("--proto_k",type=int,default=3)
    ap.add_argument("--use_retrieval",action="store_true"); ap.add_argument("--retrieval_k",type=int,default=5); ap.add_argument("--memory_capacity",type=int,default=4096); ap.add_argument("--uncertainty_threshold",type=float,default=0.55)
    ap.add_argument("--val_interval",type=int,default=1000); ap.add_argument("--log_interval",type=int,default=20)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=build_dataset(args.train_dataset_path,args.train_data_list_path,args.image_size,"train")
    val=build_dataset(args.val_dataset_path,args.val_data_list_path,args.image_size,"val")
    tl=DataLoader(train,batch_size=args.train_batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=True)
    vl=DataLoader(val,batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    model=DamageFormer_DARM(topk_ratio=args.topk_ratio,boundary_radius=args.boundary_radius,ring_radius=args.ring_radius,use_prototypes=args.use_prototypes,proto_k=args.proto_k,use_retrieval=args.use_retrieval,retrieval_k=args.retrieval_k,memory_capacity=args.memory_capacity,uncertainty_threshold=args.uncertainty_threshold).to(device)
    if args.pretrained_model_path:
        ck=torch.load(args.pretrained_model_path,map_location="cpu"); sd=ck.get("state_dict",ck.get("model_state_dict",ck))
        sd={k.removeprefix("module."):v for k,v in sd.items()}
        missing,unexpected=model.load_state_dict(sd,strict=False)
        print("load missing/unexpected",len(missing),len(unexpected))
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    ce_loc=nn.CrossEntropyLoss(ignore_index=args.ignore_index); ce_dam=nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and args.amp_dtype=="fp16"))
    dtype=torch.float16 if args.amp_dtype=="fp16" else torch.bfloat16
    outdir=Path(args.output_root)/args.exp_name; outdir.mkdir(parents=True,exist_ok=True)
    json.dump(vars(args),open(outdir/"config.json","w"),indent=2,ensure_ascii=False)
    best=-1; opt_step=0; micro_step=0; opt.zero_grad(set_to_none=True)
    while opt_step<args.max_steps:
        for batch in tl:
            model.train(); pre,post,loc,dam=parse_batch(batch,device)
            inst=build_instance_map_from_binary(loc==1,args.min_instance_area)
            amp=(device.type=="cuda" and args.amp_dtype!="none")
            with torch.autocast("cuda",dtype=dtype,enabled=amp):
                out=model(pre,post,instance_map=inst,damage_gt=dam,update_memory=True)
                l_loc=ce_loc(out["output_loc"].float(),loc)+lovasz_softmax(torch.softmax(out["output_loc"].float(),1),loc,args.ignore_index)
                l_dam=ce_dam(out["output_dam"].float(),dam)+lovasz_softmax(torch.softmax(out["output_dam"].float(),1),dam,args.ignore_index)
                l_inst,istat=hierarchical_instance_loss(out)
                l_ia=intact_alignment_loss(out,dam,args.ignore_index)
                wi=ramp_weight(opt_step,args.lambda_inst,args.inst_warmup,args.inst_ramp)
                wa=ramp_weight(opt_step,args.lambda_ia,args.ia_warmup,args.ia_ramp)
                loss=(l_loc+l_dam+wi*l_inst+wa*l_ia)/args.accumulation_steps
            scaler.scale(loss).backward()
            micro_step += 1
            if micro_step % args.accumulation_steps != 0:
                continue
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step%args.log_interval==0:
                print(f"step={opt_step} loss={float(loss)*args.accumulation_steps:.4f} loc={float(l_loc):.4f} dam={float(l_dam):.4f} inst={float(l_inst):.4f} ia={float(l_ia):.4f} wi={wi:.4f} gamma_inst={float(model.gamma_inst):.5f} destroyed_gt/pred={istat.get('destroyed_gt_instances',0)}/{istat.get('destroyed_pred_instances',0)}",flush=True)
            if opt_step%args.val_interval==0 or opt_step==args.max_steps:
                m=validate(model,vl,device,args.min_instance_area); print("VAL",m,flush=True)
                payload={"state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"step":opt_step,"metrics":m,"args":vars(args)}
                torch.save(payload,outdir/"last.pth")
                if m["miou3"]>best:
                    best=m["miou3"]; torch.save(payload,outdir/"best_val_miou3.pth"); json.dump(m,open(outdir/"best_metrics.json","w"),indent=2)
            if opt_step>=args.max_steps: break

if __name__=="__main__": main()
