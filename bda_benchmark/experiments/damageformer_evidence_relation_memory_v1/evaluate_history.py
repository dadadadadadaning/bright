from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from .cache import load_stage1_cache, candidate_mask
from .metrics import evaluate_score
from .config import DEFAULT_PROTOCOL

def _scatter(n, rows, values, fill=-1e9):
    out=torch.full((n,),float(fill),dtype=torch.float32)
    out[rows.long()]=values.float()
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--val-cache",required=True)
    ap.add_argument("--history",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    va=load_stage1_cache(args.val_cache,"val")
    h=torch.load(args.history,map_location="cpu",weights_only=False)
    rows=h["val"]["rows"].long()
    cand=candidate_mask(va)
    cov=DEFAULT_PROTOCOL.coverage

    schemes = {
        "E2P_single_prototype":
            h["val"]["prototype"]["delta_di"],
        "E2Pb_event_balanced_prototype":
            h["val"]["prototype_event_balanced"]["delta_di"],
        "E3A_global_class_conditional_memory":
            h["val"]["global_memory"]["delta_di"],
        "E3B_local_region_memory":
            h["val"]["local_memory"]["delta_di"],
        "E3C_weighted_local_region_memory":
            h["val"]["weighted_local_memory"]["delta_di"],
    }

    results={}
    for name,score_rows in schemes.items():
        score=_scatter(len(va["labels"]),rows,score_rows)
        results[name]=evaluate_score(va,score,cand,cov,name)

    # Diagnostic X-veto variants; not the primary causal comparison.
    for prefix,key in [
        ("E3A_global_xveto","global_memory"),
        ("E3B_local_xveto","local_memory"),
        ("E3C_weighted_xveto","weighted_local_memory"),
    ]:
        d=h["val"][key]["delta_di"].float()
        xv=h["val"][key]["x_veto"].float()
        gated=d.clone()
        gated[xv>0]=-1e9
        score=_scatter(len(va["labels"]),rows,gated)
        results[prefix]=evaluate_score(va,score,cand,cov,prefix)

    base=results["E2P_single_prototype"]["base_metrics"]
    out={
        "protocol":"E3_HISTORY_CAUSAL_ABLATION_V1",
        "control":h.get("control","none"),
        "coverage":cov,
        "candidate_count":int(cand.sum()),
        "base_metrics":base,
        "results":results,
        "causal_checks":{
            "local_minus_global_pp":
                results["E3B_local_region_memory"]["delta_valid_miou_pp"]-
                results["E3A_global_class_conditional_memory"]["delta_valid_miou_pp"],
            "weighted_minus_local_pp":
                results["E3C_weighted_local_region_memory"]["delta_valid_miou_pp"]-
                results["E3B_local_region_memory"]["delta_valid_miou_pp"],
            "weighted_minus_prototype_pp":
                results["E3C_weighted_local_region_memory"]["delta_valid_miou_pp"]-
                results["E2P_single_prototype"]["delta_valid_miou_pp"],
        },
        "test_accessed":False,
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
