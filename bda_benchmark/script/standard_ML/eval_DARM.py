#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR=Path(__file__).resolve().parent; BDA_ROOT=(SCRIPT_DIR/"../..").resolve()
for p in (str(BDA_ROOT),str(BDA_ROOT/"model"),str(SCRIPT_DIR)):
    if p not in sys.path: sys.path.insert(0,p)
from model.DamageFormer_DARM import DamageFormer_DARM
from model.region_ops import build_instance_map_from_binary
from train_DARM import build_dataset,parse_batch,hist_update,metrics

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--dataset_path",required=True); ap.add_argument("--data_list_path",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--image_size",type=int,default=640); ap.add_argument("--batch_size",type=int,default=1); ap.add_argument("--num_workers",type=int,default=4); ap.add_argument("--instance_source",choices=["gt","pred","none"],default="pred"); ap.add_argument("--min_instance_area",type=int,default=8); ap.add_argument("--output_json",default="DARM_eval.json")
    args=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(args.checkpoint,map_location="cpu"); cfg=ck.get("args",{})
    allowed={"topk_ratio","boundary_radius","ring_radius","use_prototypes","proto_k","use_retrieval","retrieval_k","memory_capacity","uncertainty_threshold"}
    kwargs={k:cfg[k] for k in allowed if k in cfg}; model=DamageFormer_DARM(**kwargs).to(device)
    sd={k.removeprefix("module."):v for k,v in ck.get("state_dict",ck).items()}; model.load_state_dict(sd,strict=True); model.eval()
    ds=build_dataset(args.dataset_path,args.data_list_path,args.image_size,args.split); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    lh=torch.zeros(2,2,dtype=torch.float64); dh=torch.zeros(4,4,dtype=torch.float64); inst_total=inst_correct=destroyed_gt=destroyed_pred=0
    for batch in dl:
        pre,post,loc,dam=parse_batch(batch,device)
        if args.instance_source=="none": out=model(pre,post,None,None)
        elif args.instance_source=="gt": out=model(pre,post,build_instance_map_from_binary(loc==1,args.min_instance_area),dam)
        else:
            first=model(pre,post,None,None); pred_loc=first["output_loc"].argmax(1); inst=build_instance_map_from_binary(pred_loc==1,args.min_instance_area); out=model(pre,post,inst,dam)
        pl=out["output_loc"].argmax(1); pd=out["output_dam"].argmax(1); hist_update(lh,pl,loc,2); hist_update(dh,pd,dam,4)
        if out.get("instance_labels") is not None and out["instance_labels"].numel():
            iy=out["instance_labels"]; ip=out["building_logits3"].argmax(-1)+1
            inst_total+=int(iy.numel()); inst_correct+=int((iy==ip).sum()); destroyed_gt+=int((iy==3).sum()); destroyed_pred+=int((ip==3).sum())
    m=metrics(dh); m.update({"instance_source":args.instance_source,"instance_acc":inst_correct/max(inst_total,1),"destroyed_gt_instances":destroyed_gt,"destroyed_pred_instances":destroyed_pred,"loc_confusion":lh.tolist(),"damage_confusion":dh.tolist()})
    gt_dmg=dh[2].sum().clamp_min(1); gt_des=dh[3].sum().clamp_min(1); m["damaged_to_intact"]=float(dh[2,1]/gt_dmg); m["destroyed_to_intact"]=float(dh[3,1]/gt_des)
    json.dump(m,open(args.output_json,"w"),indent=2); print(json.dumps(m,indent=2))
if __name__=="__main__": main()
