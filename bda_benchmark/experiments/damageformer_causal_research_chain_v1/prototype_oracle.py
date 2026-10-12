#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .core import D, I, binary_auroc, cache_fingerprint, classification_metrics, load_cache, write_json
from .prototype_utils import build_class_prototypes, score_prototypes
from .representation import transform_features


def main() -> None:
    p = argparse.ArgumentParser("Train-only single/multi prototype oracle")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prototypes", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--exclude_same_event", action="store_true")
    p.add_argument("--tail_ckpt", default="", help="Optional Stage-B multi-prototype checkpoint; use its projected embedding for prototype oracle.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    tr = load_cache(args.train_cache, expected_split="train")
    va = load_cache(args.val_cache, expected_split="val")
    xtr_raw, ytr = tr["features"].float(), tr["labels"].long()
    xva_raw, yva = va["features"].float(), va["labels"].long()
    xtr, _ = transform_features(xtr_raw, args.tail_ckpt, args.device)
    xva, _ = transform_features(xva_raw, args.tail_ckpt, args.device)
    scores = torch.empty((len(xva), 3), dtype=torch.float32)
    if not args.exclude_same_event:
        proto = build_class_prototypes(xtr, ytr, args.prototypes, args.seed)
        scores[:] = score_prototypes(xva, proto, args.temperature)
    else:
        for e in sorted(set(str(x) for x in va["events"])):
            qids = torch.tensor([str(x) == e for x in va["events"]], dtype=torch.bool)
            mids = torch.tensor([str(x) != e for x in tr["events"]], dtype=torch.bool)
            proto = build_class_prototypes(xtr[mids], ytr[mids], args.prototypes, args.seed)
            scores[qids] = score_prototypes(xva[qids], proto, args.temperature)

    metrics = classification_metrics(scores, yva, va["events"])
    base_pred = va["base_probs"].argmax(1)
    candidate = base_pred == I
    id_candidate = candidate & ((yva == I) | (yva == D))
    delta = scores[:, D] - scores[:, I]
    missed = candidate & (yva == D)
    correct_i = candidate & (yva == I)
    rescue_auroc = binary_auroc(delta[id_candidate], (yva[id_candidate] == D).long()) if bool(id_candidate.any()) else float("nan")
    out = {
        "protocol": "CAUSAL_CHAIN_STAGE_D_PROTOTYPE_ORACLE",
        "prototypes_per_class": args.prototypes,
        "exclude_same_event": bool(args.exclude_same_event),
        "tail_checkpoint": args.tail_ckpt or None,
        "train_cache": cache_fingerprint(args.train_cache),
        "val_cache": cache_fingerprint(args.val_cache),
        "metrics_all_buildings": metrics,
        "base_intact_candidate_count": int(candidate.sum()),
        "base_missed_damaged": int(missed.sum()),
        "base_correct_intact": int(correct_i.sum()),
        "proto_delta_rescue_auroc": float(rescue_auroc),
        "delta_mean_missed_damaged": float(delta[missed].mean()) if bool(missed.any()) else float("nan"),
        "delta_mean_correct_intact": float(delta[correct_i].mean()) if bool(correct_i.any()) else float("nan"),
        "test_accessed": False,
    }
    write_json(args.output, out)
    torch.save({"scores": scores.half(), "delta_di": delta.half(), "labels": yva, "names": va["names"], "events": va["events"]}, str(Path(args.output).with_suffix(".pt")))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
