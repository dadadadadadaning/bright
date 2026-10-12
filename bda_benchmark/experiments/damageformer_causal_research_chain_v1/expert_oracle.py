#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from .core import D, I, cache_fingerprint, coverage_curve, load_cache, top2_is_id, write_json


def main() -> None:
    p = argparse.ArgumentParser("Hard I/D expert headroom audit")
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate", choices=("base_intact", "top2_id", "union"), default="union")
    p.add_argument("--uncertainty_min", type=float, default=0.0)
    args = p.parse_args()
    va = load_cache(args.val_cache, expected_split="val")
    labels = va["labels"].long()
    probs = va["base_probs"].float()
    pred = probs.argmax(1)
    uncertainty = va["uncertainty"].float()
    c1 = pred == I
    c2 = top2_is_id(probs)
    if args.candidate == "base_intact":
        candidate = c1
    elif args.candidate == "top2_id":
        candidate = c2
    else:
        candidate = c1 | c2
    candidate &= uncertainty >= args.uncertainty_min
    valid_id = (labels == I) | (labels == D)
    candidate &= valid_id
    positive = labels == D
    negative = labels == I
    # uncertainty-only ranking is a baseline, not the learned expert.
    curve = coverage_curve(uncertainty, candidate, positive, negative)
    n = int(candidate.sum())
    d = int((candidate & positive).sum())
    i = int((candidate & negative).sum())
    out = {
        "protocol": "CAUSAL_CHAIN_STAGE_C_EXPERT_ORACLE",
        "val_cache": cache_fingerprint(args.val_cache),
        "candidate_rule": args.candidate,
        "uncertainty_min": args.uncertainty_min,
        "candidate_count": n,
        "candidate_damaged": d,
        "candidate_intact": i,
        "candidate_damaged_prevalence": d / max(1, n),
        "perfect_expert_max_rescue": d,
        "risk_pool_intact": i,
        "uncertainty_ranking_curve": curve,
        "test_accessed": False,
    }
    write_json(args.output, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
