# -*- coding: utf-8 -*-
"""Extract a TRAIN-only building-instance memory from IARM embeddings.

Each entry is a connected building region from the training localization mask.
The bank is balanced per (event, damage class) and compressed with greedy
farthest-point coreset selection, following the diversity-preserving principle
of PatchCore rather than averaging all instances into a few prototypes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
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


def parse_args():
    p=argparse.ArgumentParser(description="Build TRAIN-only IARM building memory")
    p.add_argument("--checkpoint",required=True); p.add_argument("--config_json",required=True)
    p.add_argument("--dataset_path",required=True); p.add_argument("--train_data_list_path",required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--min_component_area",type=int,default=16); p.add_argument("--min_label_purity",type=float,default=0.60)
    p.add_argument("--max_candidates_per_event_class",type=int,default=2000)
    p.add_argument("--max_memory_per_event_class",type=int,default=500)
    p.add_argument("--connectivity",choices=(1,2),type=int,default=2)
    p.add_argument("--seed",type=int,default=3407); p.add_argument("--output",required=True)
    return p.parse_args()


def read_list(path): return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def infer_event(name):
    stem=Path(str(name).replace("\\","/")).stem.lower(); stem=re.sub(r"_(pre|post)_disaster$","",stem); return re.sub(r"_\d+$","",stem)
def clean_state(state): return {k.removeprefix("module."):v for k,v in state.items()}

def disable_heads(model):
    for n in ("encoder_1","encoder_2"):
        s=getattr(getattr(model,n),"swin_transformer"); s.norm=nn.Identity(); s.head=nn.Identity()

def build_model(args,device):
    config=json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    obj=torch.load(args.checkpoint,map_location="cpu"); state=clean_state(obj["model"] if isinstance(obj,dict) and "model" in obj else obj)
    stages=tuple(int(x.strip()) for x in str(config.get("residual_stages","4")).split(",") if x.strip())
    model=DamageFormer_IARM(
        roi_alpha=float(config.get("roi_alpha",0.1)),use_pretrained_swin=False,residual_stages=stages,residual_dim=int(config.get("residual_dim",state["class_distribution_memory.means"].shape[1])),
        flow_hidden_dim=int(config.get("flow_hidden_dim",64)),max_offset_pixels=float(config.get("max_offset_pixels",2.0)),detach_backbone_for_residual=bool(config.get("detach_backbone_for_residual",True)),
        enable_residual_fusion=bool(config.get("enable_residual_fusion",True)),residual_gamma_max=float(config.get("residual_gamma_max",0.25)),num_events=int(config.get("num_events",14)),
        gaussian_variance_floor=float(config.get("gaussian_variance_floor",2e-3)),
    ).to(device)
    disable_heads(model); model.load_state_dict(state,strict=True); model.eval(); return model,config


def reservoir_add(group: List[Dict[str,object]], item: Dict[str,object], seen: int, capacity: int, rng: random.Random) -> None:
    if len(group)<capacity: group.append(item)
    else:
        j=rng.randint(0,seen)
        if j<capacity: group[j]=item


def greedy_cosine_coreset(features: torch.Tensor, capacity: int) -> torch.Tensor:
    n=features.shape[0]
    if n<=capacity: return torch.arange(n)
    x=F.normalize(features.float(),dim=1)
    mean=F.normalize(x.mean(0,keepdim=True),dim=1)
    first=int(torch.argmin((x@mean.t()).squeeze(1)).item())
    selected=[first]
    max_similarity=x@x[first]
    min_distance=1.0-max_similarity
    for _ in range(1,capacity):
        idx=int(torch.argmax(min_distance).item()); selected.append(idx)
        similarity=x@x[idx]
        min_distance=torch.minimum(min_distance,1.0-similarity)
        min_distance[selected]=-1.0
    return torch.tensor(selected,dtype=torch.long)


def component_embedding(embedding:torch.Tensor, mask_np:np.ndarray) -> Tuple[torch.Tensor,int]:
    mask=torch.from_numpy(mask_np.astype(np.float32))[None,None].to(embedding.device)
    mask_small=F.interpolate(mask,size=embedding.shape[-2:],mode="nearest")[0,0]>0.5
    count=int(mask_small.sum().item())
    if count<=0: return embedding.new_zeros(embedding.shape[1]),0
    vector=embedding[0].permute(1,2,0)[mask_small].mean(0)
    return F.normalize(vector,dim=0),count


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device(args.device)
    model,config=build_model(args,device); names=read_list(args.train_data_list_path)
    dataset=MultimodalDamageAssessmentDatset(args.dataset_path,names,1024,None,"test")
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    groups:Dict[Tuple[str,int],List[Dict[str,object]]]=defaultdict(list); seen=defaultdict(int); rng=random.Random(args.seed)
    structure=ndimage.generate_binary_structure(2,args.connectivity)

    with torch.no_grad():
        for pre,post,loc,damage,sample_names in tqdm(loader,desc="Extract TRAIN memory"):
            pre=pre.to(device); post=post.to(device); loc_np=loc[0].numpy(); damage_np=damage[0].numpy(); sample=str(sample_names[0]); event=infer_event(sample)
            out_loc,out_damage,_,aux=model(pre,post,return_aux=True); embedding=aux["metric_embedding"]; probabilities=F.softmax(out_damage,dim=1)
            building=(loc_np==1); component_map,num=ndimage.label(building,structure=structure)
            for component_id in range(1,num+1):
                mask=component_map==component_id; area=int(mask.sum())
                if area<args.min_component_area: continue
                labels=damage_np[mask]; labels=labels[(labels>=1)&(labels<=3)]
                if labels.size==0: continue
                counts=np.bincount(labels.astype(np.int64),minlength=4); label=int(np.argmax(counts[1:4])+1); purity=float(counts[label]/labels.size)
                if purity<args.min_label_purity: continue
                vector,feature_pixels=component_embedding(embedding,mask)
                if feature_pixels<=0: continue
                full_mask=torch.from_numpy(mask.astype(np.float32))[None,None].to(device)
                prob_mask=full_mask[0,0]>0.5; region_prob=probabilities[0,:,prob_mask].mean(1)
                main_pred=int(region_prob.argmax().item())
                item={"feature":vector.cpu(),"label":label,"event":event,"sample":sample,"area":area,"feature_pixels":feature_pixels,"purity":purity,"main_prediction":main_pred,"main_confidence":float(region_prob.max().item()),"hard_to_intact":bool(label in (2,3) and main_pred==1)}
                key=(event,label); seen[key]+=1; reservoir_add(groups[key],item,seen[key]-1,args.max_candidates_per_event_class,rng)

    selected=[]; group_summary={}
    for key,items in sorted(groups.items()):
        features=torch.stack([item["feature"] for item in items])
        indices=greedy_cosine_coreset(features,args.max_memory_per_event_class)
        chosen=[items[int(i)] for i in indices]
        selected.extend(chosen)
        group_summary[f"{key[0]}|{key[1]}"]={"seen":seen[key],"reservoir":len(items),"selected":len(chosen),"hard_selected":sum(int(x["hard_to_intact"]) for x in chosen)}

    if not selected: raise RuntimeError("No building instances were extracted")
    features=torch.stack([x["feature"] for x in selected]).float(); labels=torch.tensor([x["label"] for x in selected],dtype=torch.long)
    events=[str(x["event"]) for x in selected]; samples=[str(x["sample"]) for x in selected]
    class_centres=[]; class_variances=[]
    for label in (1,2,3):
        class_features=features[labels==label]
        if class_features.numel()==0: raise RuntimeError(f"Memory has no class {label}")
        class_centres.append(F.normalize(class_features.mean(0),dim=0)); class_variances.append(class_features.var(0,unbiased=False).clamp_min(2e-3))
    output={
        "features":features,"labels":labels,"events":events,"samples":samples,
        "areas":torch.tensor([x["area"] for x in selected]),"purity":torch.tensor([x["purity"] for x in selected]),
        "main_predictions":torch.tensor([x["main_prediction"] for x in selected]),"main_confidence":torch.tensor([x["main_confidence"] for x in selected]),
        "hard_to_intact":torch.tensor([x["hard_to_intact"] for x in selected],dtype=torch.bool),
        "class_centres":torch.stack(class_centres),"class_variances":torch.stack(class_variances),
        "metadata":{"checkpoint":args.checkpoint,"config_json":args.config_json,"train_list":args.train_data_list_path,"train_samples":len(names),"memory_size":len(selected),"group_summary":group_summary,"settings":vars(args)},
    }
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); torch.save(output,path,_use_new_zipfile_serialization=False)
    Path(str(path)+".json").write_text(json.dumps(output["metadata"],indent=2,ensure_ascii=False),encoding="utf-8")
    print("Saved memory:",path); print("Size:",len(selected)); print("Class counts:",[(int(label),int((labels==label).sum())) for label in (1,2,3)])

if __name__=="__main__": main()
