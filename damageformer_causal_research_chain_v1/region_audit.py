#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .core import D, I, X, binary_auroc, cache_fingerprint, load_cache, write_json


def q(x: torch.Tensor) -> Dict[str, float]:
    a = x.detach().float().cpu().numpy()
    if a.size == 0:
        return {k: 0.0 for k in ("mean", "p10", "p25", "p50", "p75", "p90")}
    return {
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
    }


def main() -> None:
    p = argparse.ArgumentParser("Stage-A cache audit")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    c = load_cache(args.cache)
    labels = c["labels"].long()
    region_count = c["region_count"].long()
    peak_delta = c["peak_mean_delta"].float()
    out: Dict[str, object] = {
        "protocol": "CAUSAL_CHAIN_STAGE_A_REGION_EVIDENCE_AUDIT",
        "cache": cache_fingerprint(args.cache),
        "split": c["split"],
        "building_count": int(labels.numel()),
        "regions_per_building": q(region_count.float()),
        "peak_minus_mean_all": q(peak_delta),
        "by_class": {},
        "test_accessed": False,
    }
    for cls, name in ((I, "Intact"), (D, "Damaged"), (X, "Destroyed")):
        m = labels == cls
        out["by_class"][name] = {
            "count": int(m.sum()),
            "regions_per_building": q(region_count[m].float()),
            "peak_minus_mean": q(peak_delta[m]),
        }
    id_mask = (labels == I) | (labels == D)
    out["peak_delta_di_auroc"] = float(binary_auroc(peak_delta[id_mask], (labels[id_mask] == D).long()))
    out["damaged_minus_intact_peak_delta_mean"] = float(
        peak_delta[labels == D].mean() - peak_delta[labels == I].mean()
    ) if bool((labels == D).any() and (labels == I).any()) else float("nan")
    write_json(args.output, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
