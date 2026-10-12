#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from .late_core import D, I, candidate_mask, fixed_budget_rows, write_json, zscore_on_mask


def normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def class_centroid(x, y, cls):
    rows = x[y == cls]
    if len(rows) == 0:
        raise RuntimeError(f"no support for class {cls}")
    return normalize(rows.mean(0, keepdim=True))[0]


def main():
    p = argparse.ArgumentParser("E2: single-prototype historical verification")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--specialist_scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_train_purity", type=float, default=0.70)
    args = p.parse_args()

    tr = torch.load(args.train_cache, map_location="cpu", weights_only=False, mmap=True)
    va = torch.load(args.val_cache, map_location="cpu", weights_only=False, mmap=True)
    sp = torch.load(args.specialist_scores, map_location="cpu", weights_only=False)
    candidate = candidate_mask(va, "base_intact")
    if not torch.equal(candidate, sp["candidate_mask"].bool()):
        raise RuntimeError("specialist candidate mask does not match base_intact late-decision candidate")

    xtr = normalize(tr["features"].float())
    xva = normalize(va["features"].float())
    ytr = tr["labels"].long(); purity = tr["label_purity"].float()
    valid_support = (purity >= args.min_train_purity) & ((ytr == I) | (ytr == D))
    proto_delta = torch.zeros(len(xva), dtype=torch.float32)

    # Same-event exclusion: each query uses only other-event train supports.
    val_events = [str(e) for e in va["events"]]
    train_events = [str(e) for e in tr["events"]]
    for e in sorted(set(val_events)):
        q = torch.tensor([x == e for x in val_events], dtype=torch.bool)
        support = valid_support & torch.tensor([x != e for x in train_events], dtype=torch.bool)
        pi = class_centroid(xtr[support], ytr[support], I)
        pd = class_centroid(xtr[support], ytr[support], D)
        proto_delta[q] = xva[q] @ pd - xva[q] @ pi

    spec = sp["score"].float()
    spec_z = zscore_on_mask(spec, candidate)
    proto_z = zscore_on_mask(proto_delta, candidate)
    combined = spec_z + proto_z  # fixed lambda=1, no GT tuning.

    out = {
        "protocol": "LATE_DECISION_E2_SINGLE_PROTOTYPE_VERIFY",
        "same_event_excluded": True,
        "fusion": "z(Specialist)+z(ProtoDelta), lambda=1 fixed",
        "specialist_only": fixed_budget_rows(spec, va, candidate=candidate),
        "prototype_only": fixed_budget_rows(proto_delta, va, candidate=candidate),
        "specialist_plus_prototype": fixed_budget_rows(combined, va, candidate=candidate),
        "test_accessed": False,
    }
    write_json(args.output, out)
    torch.save({"proto_delta": proto_delta.half(), "combined_score": combined.half(), "candidate_mask": candidate}, str(Path(args.output).with_suffix(".pt")))
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
