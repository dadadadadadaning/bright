from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from .cache import load_stage1_cache, building_area
from .metrics import exact_single_utility
from .config import DEFAULT_PROTOCOL

def _quantile(x, q):
    return float(torch.quantile(x.float(), torch.tensor(q)))

def _group_stats(cache, rows, mask):
    r = rows[mask]
    if len(r) == 0:
        return {"count":0}
    y = cache["labels"].long()[r]
    purity = cache["label_purity"].float()[r]
    clean = purity >= DEFAULT_PROTOCOL.min_purity
    area = building_area(cache)[r].float()
    utilities = torch.tensor([
        exact_single_utility(cache["base_pixel_cm"], cache["building_pixel_cm"][int(i)])
        for i in r
    ])
    cm = cache["building_pixel_cm"][r].long()
    return {
        "count":int(len(r)),
        "clean_intact":int((clean & (y==0)).sum()),
        "clean_damaged":int((clean & (y==1)).sum()),
        "clean_destroyed":int((clean & (y==2)).sum()),
        "unknown_or_impure":int(((~clean)|(y<0)).sum()),
        "damaged_prevalence_clean":float((clean & (y==1)).sum()/clean.sum().clamp_min(1)),
        "mean_exact_utility_pp":100.0*float(utilities.mean()),
        "positive_utility_rate":float((utilities>0).float().mean()),
        "mean_building_area":float(area.mean()),
        "rescue_pixels_D_to_I_reversed":int(cm[:,2,1].sum()),
        "harm_pixels_I_correct_to_D":int(cm[:,1,1].sum()),
        "destroyed_pixels_X_pred_I_to_D":int(cm[:,3,1].sum()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-cache",required=True)
    ap.add_argument("--val-cache",required=True)
    ap.add_argument("--specialist-train-scores",required=True)
    ap.add_argument("--specialist-val-scores",required=True)
    ap.add_argument("--history",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    tr=load_stage1_cache(args.train_cache,"train")
    va=load_stage1_cache(args.val_cache,"val")
    st=torch.load(args.specialist_train_scores,map_location="cpu",weights_only=False)
    sv=torch.load(args.specialist_val_scores,map_location="cpu",weights_only=False)["scores"].float()
    hist=torch.load(args.history,map_location="cpu",weights_only=False)

    cal_rows=hist["cal"]["rows"].long()
    val_rows=hist["val"]["rows"].long()
    cal_spec=st["scores"].float()[cal_rows]
    cal_proto=hist["cal"]["prototype"]["delta_di"].float()
    val_spec=sv[val_rows]
    val_proto=hist["val"]["prototype"]["delta_di"].float()

    # Frozen thresholds come from Train-Cal, never Val GT.
    q=DEFAULT_PROTOCOL.relation_high_quantile
    spec_thr=_quantile(cal_spec,q)
    proto_thr=_quantile(cal_proto,q)

    hs=val_spec>=spec_thr
    hp=val_proto>=proto_thr
    groups={
        "HH_current_high_history_high":hs&hp,
        "HL_current_high_history_low":hs&(~hp),
        "LH_current_low_history_high":(~hs)&hp,
        "LL_current_low_history_low":(~hs)&(~hp),
    }
    out={
        "protocol":"E2_1_SPECIALIST_PROTOTYPE_RELATION_AUDIT",
        "threshold_source":"Train-Cal quantiles only",
        "high_quantile":q,
        "specialist_threshold":spec_thr,
        "prototype_threshold":proto_thr,
        "val_candidate_count":int(len(val_rows)),
        "groups":{k:_group_stats(va,val_rows,m) for k,m in groups.items()},
        "test_accessed":False,
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
