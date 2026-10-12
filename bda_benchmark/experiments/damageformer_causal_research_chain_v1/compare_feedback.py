#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, Tuple

import torch

from .core import D, I, cache_fingerprint, classification_metrics, load_cache, rescue_harm_from_scores, write_json


def key_map(cache: Dict[str, object]) -> Dict[Tuple[str, int], int]:
    return {(str(n), int(b)): i for i, (n, b) in enumerate(zip(cache["names"], cache["pred_building_ids"].tolist()))}


def main() -> None:
    p = argparse.ArgumentParser("Compare Stage-1 identity predictions with Stage-2 feedback")
    p.add_argument("--base_cache", required=True)
    p.add_argument("--feedback_cache", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    a = load_cache(args.base_cache, expected_split="val")
    b = load_cache(args.feedback_cache, expected_split="val")
    ma, mb = key_map(a), key_map(b)
    common = sorted(set(ma) & set(mb))
    if not common:
        raise RuntimeError("no matched buildings between caches")
    ia = torch.tensor([ma[k] for k in common], dtype=torch.long)
    ib = torch.tensor([mb[k] for k in common], dtype=torch.long)
    labels_a = a["labels"][ia].long()
    labels_b = b["labels"][ib].long()
    if not torch.equal(labels_a, labels_b):
        raise RuntimeError("GT building labels differ between matched caches")
    base = a["base_probs"][ia].float().log().clamp_min(-30)
    new = b["base_probs"][ib].float().log().clamp_min(-30)
    events = [a["events"][i] for i in ia.tolist()]
    out = {
        "protocol": "CAUSAL_CHAIN_STAGE_A_FEEDBACK_COMPARISON",
        "base_cache": cache_fingerprint(args.base_cache),
        "feedback_cache": cache_fingerprint(args.feedback_cache),
        "matched_buildings": len(common),
        "base_metrics": classification_metrics(base, labels_a, events),
        "feedback_metrics": classification_metrics(new, labels_a, events),
        "rescue_harm": rescue_harm_from_scores(base, new, labels_a),
        "test_accessed": False,
    }
    by_event = {}
    for e in sorted(set(events)):
        m = torch.tensor([x == e for x in events], dtype=torch.bool)
        r = rescue_harm_from_scores(base[m], new[m], labels_a[m])
        by_event[e] = r
    out["rescue_harm_by_event"] = by_event
    out["events_nonnegative_net"] = int(sum(int(v["net"] >= 0) for v in by_event.values()))
    write_json(args.output, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
