#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
import torch

from .late_core import D, candidate_mask, apply_override_to_cm, metrics_from_cm, write_json


def main():
    p = argparse.ArgumentParser("E0: building-level late-correction oracle")
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate", choices=("base_intact", "top2_id", "union"), default="base_intact")
    p.add_argument("--coverages", default="0.01,0.02,0.05")
    args = p.parse_args()
    va = torch.load(args.val_cache, map_location="cpu", weights_only=False, mmap=True)
    if bool(va.get("test_accessed", False)):
        raise RuntimeError("TEST forbidden")
    candidate = candidate_mask(va, args.candidate)
    ids = torch.nonzero(candidate, as_tuple=False).flatten()
    labels = va["labels"].long()
    # Oracle may use GT by definition. It never changes the candidate mask.
    true_missed = ids[labels[ids] == D]
    base = metrics_from_cm(va["base_pixel_cm"])
    rows = []
    for cov in [float(x) for x in args.coverages.split(",") if x.strip()]:
        budget = max(1, int(math.ceil(cov * len(ids)))) if len(ids) else 0
        # Max-rescue-at-budget: abstain instead of filling unused budget with known negatives.
        chosen = true_missed[: min(budget, len(true_missed))]
        for mode in ("intact_only", "whole"):
            cm = apply_override_to_cm(
                va["base_pixel_cm"], va["building_pixel_cm"], chosen, mode=mode
            )
            m = metrics_from_cm(cm)
            rows.append({
                "coverage_budget": cov,
                "budget": int(budget),
                "selected_true_missed": int(len(chosen)),
                "candidate_count": int(len(ids)),
                "true_missed_damaged": int(len(true_missed)),
                "override_mode": mode,
                **m,
                "delta_valid_miou_pp": 100.0 * (m["valid_miou"] - base["valid_miou"]),
                "delta_damaged_iou_pp": 100.0 * (m["damaged_iou"] - base["damaged_iou"]),
            })
    out = {
        "protocol": "LATE_DECISION_E0_ORACLE",
        "candidate_rule": args.candidate,
        "base_metrics": base,
        "rows": rows,
        "oracle_uses_gt_for_selection": True,
        "candidate_mask_uses_gt": False,
        "test_accessed": False,
    }
    write_json(args.output, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
