from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .utility import iou_from_confusion,override_intact_to_damaged

def metrics(conf):
    conf=conf.float(); iou=iou_from_confusion(conf); tp=conf.diag()
    precision=tp/conf.sum(0).clamp_min(1); recall=tp/conf.sum(1).clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-8)
    d2i=conf[2,1]/conf[2].sum().clamp_min(1)
    return {"miou4":float(iou.mean()),"valid_miou":float(iou[1:4].mean()),"intact_iou":float(iou[1]),"damaged_iou":float(iou[2]),"destroyed_iou":float(iou[3]),"damaged_f1":float(f1[2]),"damaged_to_intact":float(d2i)}

def evaluate_coverages(global_conf,building_conf,candidate_mask,score,coverages=(0.01,0.02,0.05)):
    idx=torch.where(candidate_mask.bool())[0]
    ranked=idx[torch.argsort(score[idx],descending=True)]
    rows=[]
    for cov in coverages:
        n=max(1,int(round(len(ranked)*cov))); conf=global_conf.clone()
        for b in ranked[:n].tolist():
            old=building_conf[b]; conf=conf-old+override_intact_to_damaged(old)
        rows.append({"coverage":cov,"candidate_count":len(ranked),"selected":n,**metrics(conf)})
    return {"base":metrics(global_conf),"rows":rows}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); args=p.parse_args()
    obj=torch.load(args.input,map_location="cpu",weights_only=False)
    out=evaluate_coverages(obj["global_confusion"],obj["building_confusions"],obj["candidate_mask"],obj["score"])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
