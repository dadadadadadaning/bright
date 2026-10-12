from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

from .cache import (
    load_stage1_cache, event_mapping, encode_events, global_region_mean,
    candidate_mask
)
from .prototype import build_prototype_bank, score_queries
from .memory import build_class_memory, global_memory_scores, local_memory_scores
from .regions import build_padded_regions
from .config import DEFAULT_PROTOCOL

def _score_split(
    rows, cache, event_ids, train_cache, train_event_ids, fit_mask,
    proto_bank, mem_bank, train_regions, query_regions,
    control, device,
):
    keys = F.normalize(global_region_mean(cache)[rows], dim=-1)
    ev = event_ids[rows]

    proto = score_queries(keys, ev, proto_bank, event_balanced=False)
    proto_bal = score_queries(keys, ev, proto_bank, event_balanced=True)

    glob = global_memory_scores(
        keys, ev, mem_bank,
        coarse_k=DEFAULT_PROTOCOL.coarse_k_per_class,
        device=device,
    )
    local = local_memory_scores(
        rows, keys, ev,
        train_cache, cache, mem_bank, train_regions, query_regions,
        weighted=False,
        coarse_k=DEFAULT_PROTOCOL.coarse_k_per_class,
        fine_k=DEFAULT_PROTOCOL.fine_k_per_class,
        global_weight=DEFAULT_PROTOCOL.local_global_weight,
        control=control,
        device=device,
    )
    weighted = local_memory_scores(
        rows, keys, ev,
        train_cache, cache, mem_bank, train_regions, query_regions,
        weighted=True,
        coarse_k=DEFAULT_PROTOCOL.coarse_k_per_class,
        fine_k=DEFAULT_PROTOCOL.fine_k_per_class,
        global_weight=DEFAULT_PROTOCOL.local_global_weight,
        control=control,
        device=device,
    )

    # Drop coarse index payload from persisted global output.
    glob = {k:v for k,v in glob.items() if k != "coarse_positions"}
    return {
        "rows": rows,
        "prototype": proto,
        "prototype_event_balanced": proto_bal,
        "global_memory": glob,
        "local_memory": local,
        "weighted_local_memory": weighted,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--specialist-train-scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--control", choices=["none","shuffle_labels","shuffle_events","wrong_regions","swapped_id"], default="none")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tr = load_stage1_cache(args.train_cache, "train")
    va = load_stage1_cache(args.val_cache, "val")
    spec = torch.load(args.specialist_train_scores, map_location="cpu", weights_only=False)
    fit_mask = spec["fit_mask"].bool()
    cal_mask = spec["cal_mask"].bool()

    mapping = event_mapping(tr["events"], va["events"])
    tr_ev = encode_events(tr["events"], mapping)
    va_ev = encode_events(va["events"], mapping)

    proto_bank = build_prototype_bank(tr, fit_mask, tr_ev, DEFAULT_PROTOCOL.min_purity)
    mem_bank = build_class_memory(
        tr, fit_mask, tr_ev, DEFAULT_PROTOCOL.min_purity,
        control=args.control if args.control in {"shuffle_labels","shuffle_events","swapped_id"} else "none",
        seed=DEFAULT_PROTOCOL.seed,
    )

    print("Building temporary padded Region stores...", flush=True)
    train_regions = build_padded_regions(tr, max_regions=16)
    val_regions = build_padded_regions(va, max_regions=16)

    cal_rows = torch.where(cal_mask & candidate_mask(tr))[0]
    val_rows = torch.where(candidate_mask(va))[0]

    print(json.dumps({
        "control":args.control,
        "train_fit_buildings":int(fit_mask.sum()),
        "train_cal_candidates":len(cal_rows),
        "val_candidates":len(val_rows),
        "memory_support":len(mem_bank.support_rows),
    }), flush=True)

    print("Scoring Train-Cal historical evidence...", flush=True)
    cal = _score_split(
        cal_rows,tr,tr_ev,tr,tr_ev,fit_mask,
        proto_bank,mem_bank,train_regions,train_regions,
        args.control,args.device
    )
    print("Scoring Val historical evidence...", flush=True)
    val = _score_split(
        val_rows,va,va_ev,tr,tr_ev,fit_mask,
        proto_bank,mem_bank,train_regions,val_regions,
        args.control,args.device
    )

    out = {
        "protocol":"EVIDENCE_RELATION_HISTORY_V1",
        "control":args.control,
        "event_map":mapping,
        "cal":cal,
        "val":val,
        "coverage":DEFAULT_PROTOCOL.coverage,
        "same_event_excluded":True,
        "test_accessed":False,
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,args.output)
    print("Saved",args.output)

if __name__ == "__main__":
    main()
