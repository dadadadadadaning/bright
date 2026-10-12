"""Map current Stage1 all-building cache to canonical V2 fields.

1) Inspect:
   python -m ...build_memory --inspect --source /path/stage1_train_all_buildings.pt
2) Replace FIELD_MAP values with real dotted keys.
"""
from __future__ import annotations
import argparse
import torch

FIELD_MAP = {
    "region_tokens": "region_tokens",
    "region_mask": "region_mask",
    "region_meta": "region_meta",
    "labels": "labels",
    "event_ids": "event_ids",
    "building_ids": "building_ids",
    "base_probs": "base_probs",
}

def nested_get(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    src = torch.load(args.source, map_location="cpu", weights_only=False)
    out = {}
    for dst,key in FIELD_MAP.items():
        try:
            out[dst] = nested_get(src,key)
        except Exception:
            if dst == "base_probs": out[dst] = None
            else: raise
    if isinstance(src, dict): out["source_sha256"] = src.get("source_sha256")
    torch.save(out,args.output)
    print("Saved canonical cache:",args.output)
    for k,v in out.items(): print(k, tuple(v.shape) if torch.is_tensor(v) else v)

if __name__ == "__main__": main()
