# -*- coding: utf-8 -*-
"""Leakage-safe building-instance retrieval evaluation for IARM.

The train-only instance cache is combined with coarse class centres:
class centres route the query to candidate class banks; concrete instances
provide fine-grained evidence. The cache only rescues predicted-Intact building
regions when retrieval confidence is sufficient.
"""

from __future__ import annotations

import argparse, json, math, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_PATH=Path(__file__).resolve(); BDA_ROOT=SCRIPT_PATH.parents[1]; PROJECT_ROOT=BDA_ROOT.parent
for root in (PROJECT_ROOT,BDA_ROOT):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_IARM import DamageFormer_IARM

CLASS_NAMES=["Background","Intact","Damaged","Destroyed"]

def parse_args():
    p=argparse.ArgumentParser(description="Evaluate Gaussian-routed instance retrieval")
    p.add_argument("--checkpoint",required=True); p.add_argument("--config_json",required=True); p.add_argument("--memory",required=True)
    p.add_argument("--dataset_path",required=True); p.add_argument("--data_list_path",required=True); p.add_argument("--split_name",default="validation")
    p.add_argument("--device",default="cuda:0"); p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--building_threshold",type=float,default=0.5); p.add_argument("--min_component_area",type=int,default=16); p.add_argument("--connectivity",choices=(1,2),type=int,default=2)
    p.add_argument("--topk_per_class",type=int,default=8); p.add_argument("--route_top_classes",type=int,default=2)
    p.add_argument("--cache_beta",type=float,default=8.0); p.add_argument("--alpha",type=float,default=1.0); p.add_argument("--prototype_weight",type=float,default=0.10)
    p.add_argument("--similarity_threshold",type=float,default=0.55); p.add_argument("--similarity_scale",type=float,default=0.05); p.add_argument("--gate_threshold",type=float,default=0.15)
    p.add_argument("--cross_event_only",action="store_true"); p.add_argument("--allow_severity_correction",action="store_true")
    p.add_argument("--output_json",required=True); return p.parse_args()

def read_list(path): return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def infer_event(name):
    stem=Path(str(name).replace("\\","/")).stem.lower(); stem=re.sub(r"_(pre|post)_disaster$","",stem); return re.sub(r"_\d+$","",stem)
def clean_state(s): return {k.removeprefix("module."):v for k,v in s.items()}
def disable_heads(model):
    for n in ("encoder_1","encoder_2"):
        s=getattr(getattr(model,n),"swin_transformer"); s.norm=nn.Identity(); s.head=nn.Identity()

def build_model(args,device):
    c=json.loads(Path(args.config_json).read_text(encoding="utf-8")); obj=torch.load(args.checkpoint,map_location="cpu"); state=clean_state(obj["model"] if isinstance(obj,dict) and "model" in obj else obj)
    stages=tuple(int(x.strip()) for x in str(c.get("residual_stages","4")).split(",") if x.strip())
    model=DamageFormer_IARM(roi_alpha=float(c.get("roi_alpha",0.1)),use_pretrained_swin=False,residual_stages=stages,residual_dim=int(c.get("residual_dim",state["class_distribution_memory.means"].shape[1])),flow_hidden_dim=int(c.get("flow_hidden_dim",64)),max_offset_pixels=float(c.get("max_offset_pixels",2.0)),detach_backbone_for_residual=bool(c.get("detach_backbone_for_residual",True)),enable_residual_fusion=bool(c.get("enable_residual_fusion",True)),residual_gamma_max=float(c.get("residual_gamma_max",0.25)),num_events=int(c.get("num_events",14)),gaussian_variance_floor=float(c.get("gaussian_variance_floor",2e-3))).to(device)
    disable_heads(model); model.load_state_dict(state,strict=True); model.eval(); return model

def fast_hist(gt,pred,classes=4):
    gt=np.asarray(gt).reshape(-1); pred=np.asarray(pred).reshape(-1); valid=(gt>=0)&(gt<classes)&(pred>=0)&(pred<classes)
    return np.bincount(classes*gt[valid].astype(np.int64)+pred[valid].astype(np.int64),minlength=classes**2).reshape(classes,classes)
def report(hist):
    tp=np.diag(hist).astype(float); union=hist.sum(1)+hist.sum(0)-tp; iou=np.divide(tp,union,out=np.zeros_like(tp),where=union>0); rows=np.divide(hist,hist.sum(1,keepdims=True),out=np.zeros_like(hist,dtype=float),where=hist.sum(1,keepdims=True)>0)
    return {"miou":float(iou.mean()),"target3_miou":float(iou[1:4].mean()),"iou":iou.tolist(),"row_percent":(100*rows).tolist(),"damaged_to_intact":float(100*rows[2,1]),"destroyed_to_intact":float(100*rows[3,1]),"intact_to_damaged":float(100*rows[1,2]),"intact_to_destroyed":float(100*rows[1,3])}

def component_vector(embedding,mask_np):
    mask=torch.from_numpy(mask_np.astype(np.float32))[None,None].to(embedding.device); small=F.interpolate(mask,size=embedding.shape[-2:],mode="nearest")[0,0]>0.5
    if not bool(small.any().item()): return None
    return F.normalize(embedding[0].permute(1,2,0)[small].mean(0),dim=0)

class RetrievalBank:
    def __init__(self,path,device):
        memory=torch.load(path,map_location="cpu"); self.features=F.normalize(memory["features"].float(),dim=1).to(device); self.labels=memory["labels"].long().to(device); self.events=list(memory["events"]); self.samples=set(memory["samples"])
        self.centres=F.normalize(memory["class_centres"].float(),dim=1).to(device); self.variances=memory["class_variances"].float().clamp_min(2e-3).to(device); self.metadata=memory.get("metadata",{})
        self.event_array=np.asarray(self.events,dtype=object)
    def query(self,q,event,route_top,topk,beta,cross_event_only):
        q=F.normalize(q,dim=0); proto_scores=q@self.centres.t(); route=torch.topk(proto_scores,k=min(route_top,3)).indices.tolist()
        class_affinity=q.new_full((3,),-1e4); class_top_similarity=q.new_full((3,),-1.0); neighbors={}
        for class_idx in route:
            label=class_idx+1; mask=(self.labels==label)
            if cross_event_only:
                allowed=torch.tensor(self.event_array!=event,device=q.device,dtype=torch.bool); mask=mask&allowed
            indices=torch.where(mask)[0]
            if indices.numel()==0: continue
            sims=self.features[indices]@q; k=min(topk,int(sims.numel())); values,order=torch.topk(sims,k=k); chosen=indices[order]
            affinity=torch.exp(-beta*(1.0-values)).mean(); class_affinity[class_idx]=torch.log(affinity.clamp_min(1e-12)); class_top_similarity[class_idx]=values[0]
            neighbors[class_idx]={"labels":[label]*k,"similarities":values.detach().cpu().tolist(),"indices":chosen.detach().cpu().tolist()}
        valid=class_affinity>-1e3
        if not bool(valid.any().item()): return None
        cache_prob=torch.softmax(class_affinity.masked_fill(~valid,-1e4),dim=0); retrieval_idx=int(cache_prob.argmax().item()); sorted_prob=torch.sort(cache_prob,descending=True).values; margin=float((sorted_prob[0]-sorted_prob[1]).item())
        top_similarity=float(class_top_similarity[retrieval_idx].item())
        return {"proto_scores":proto_scores,"cache_logits":class_affinity,"cache_prob":cache_prob,"retrieval_label":retrieval_idx+1,"top_similarity":top_similarity,"margin":margin,"neighbors":neighbors}

def main():
    args=parse_args(); device=torch.device(args.device); model=build_model(args,device); bank=RetrievalBank(args.memory,device); names=read_list(args.data_list_path)
    overlap=bank.samples.intersection(names)
    if overlap: raise RuntimeError(f"Memory/query leakage: {len(overlap)} query samples are present in TRAIN memory; example={next(iter(overlap))}")
    dataset=MultimodalDamageAssessmentDatset(args.dataset_path,names,1024,None,"test"); loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    hist_main=np.zeros((4,4),np.int64); hist_hybrid=np.zeros((4,4),np.int64); event_main=defaultdict(lambda:np.zeros((4,4),np.int64)); event_hybrid=defaultdict(lambda:np.zeros((4,4),np.int64)); structure=ndimage.generate_binary_structure(2,args.connectivity)
    stats=defaultdict(int); gate_values=[]; retrieval_region_correct=0; retrieval_region_total=0
    with torch.no_grad():
        for pre,post,loc_gt,damage_gt,sample_names in tqdm(loader,desc=f"Retrieval {args.split_name}"):
            sample=str(sample_names[0]); event=infer_event(sample); pre=pre.to(device); post=post.to(device)
            out_loc,out_damage,_,aux=model(pre,post,return_aux=True); probabilities=F.softmax(out_damage,dim=1); pred_main=out_damage.argmax(1)[0].cpu().numpy(); pred_hybrid=pred_main.copy(); gt=damage_gt[0].numpy()
            building_prob=F.softmax(out_loc,dim=1)[0,1].cpu().numpy(); predicted_building=building_prob>=args.building_threshold; components,num=ndimage.label(predicted_building,structure=structure)
            for component_id in range(1,num+1):
                mask=components==component_id; area=int(mask.sum())
                if area<args.min_component_area: continue
                vector=component_vector(aux["metric_embedding"],mask)
                if vector is None: continue
                mask_t=torch.from_numpy(mask).to(device); region_prob=probabilities[0,:,mask_t].mean(1); main_region_label=int(region_prob.argmax().item())
                result=bank.query(vector,event,args.route_top_classes,args.topk_per_class,args.cache_beta,args.cross_event_only)
                if result is None: stats["no_neighbor"]+=1; continue
                retrieval_label=int(result["retrieval_label"]); retrieval_region_total+=1
                gt_labels=gt[mask]; gt_labels=gt_labels[(gt_labels>=1)&(gt_labels<=3)]
                if gt_labels.size:
                    gt_region=int(np.argmax(np.bincount(gt_labels.astype(np.int64),minlength=4)[1:4])+1); retrieval_region_correct+=int(retrieval_label==gt_region)
                main_building_prob=region_prob[1:4]; main_building_prob=main_building_prob/main_building_prob.sum().clamp_min(1e-8); uncertainty=float(1.0-main_building_prob.max().item())
                similarity_conf=1.0/(1.0+math.exp(-(result["top_similarity"]-args.similarity_threshold)/max(args.similarity_scale,1e-6)))
                retrieval_conf=float(result["cache_prob"].max().item()); gate=math.sqrt(max(uncertainty*similarity_conf,0.0))*retrieval_conf*math.sqrt(max(result["margin"],0.0)+1e-8); gate_values.append(gate)
                permitted=(main_region_label==1 and retrieval_label in (2,3)) or (args.allow_severity_correction and main_region_label in (2,3) and retrieval_label in (2,3))
                if not permitted: stats["not_permitted"]+=1; continue
                if gate<args.gate_threshold: stats["rejected_gate"]+=1; continue
                main_logits=torch.log(main_building_prob.clamp_min(1e-8)); cache_logits=result["cache_logits"]; proto_logits=result["proto_scores"]
                final_logits=main_logits + args.alpha*gate*cache_logits + args.prototype_weight*gate*proto_logits
                final_label=int(final_logits.argmax().item())+1
                if final_label==main_region_label: stats["no_change"]+=1; continue
                pred_hybrid[mask]=final_label; stats["changed_regions"]+=1
                if gt_labels.size:
                    if main_region_label!=gt_region and final_label==gt_region: stats["rescued_regions"]+=1
                    elif main_region_label==gt_region and final_label!=gt_region: stats["harmed_regions"]+=1
            hist_main+=fast_hist(gt,pred_main); hist_hybrid+=fast_hist(gt,pred_hybrid); event_main[event]+=fast_hist(gt,pred_main); event_hybrid[event]+=fast_hist(gt,pred_hybrid)
    main_report=report(hist_main); hybrid_report=report(hist_hybrid); events=[]
    for event in sorted(event_main):
        a=report(event_main[event]); h=report(event_hybrid[event]); events.append({"event":event,"main":a,"hybrid":h,"delta_miou":h["miou"]-a["miou"]})
    output={"settings":vars(args),"memory_metadata":bank.metadata,"main":main_report,"hybrid":hybrid_report,"delta_miou":hybrid_report["miou"]-main_report["miou"],"retrieval_region_accuracy":retrieval_region_correct/max(retrieval_region_total,1),"mean_gate":float(np.mean(gate_values)) if gate_values else 0.0,"stats":dict(stats),"events":events}
    path=Path(args.output_json); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Main mIoU:",100*main_report["miou"],"Hybrid:",100*hybrid_report["miou"],"Delta:",100*output["delta_miou"]); print("Main IoU:",100*np.asarray(main_report["iou"])); print("Hybrid IoU:",100*np.asarray(hybrid_report["iou"])); print("Retrieval region acc:",100*output["retrieval_region_accuracy"],"stats",dict(stats)); print("Saved:",path)

if __name__=="__main__": main()
