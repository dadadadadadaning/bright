from __future__ import annotations
import argparse
from pathlib import Path
import torch
from .memory_retriever import build_global_keys

REQUIRED = ["region_tokens","region_mask","region_meta","labels","event_ids","building_ids"]

def print_tree(obj, prefix="root", depth=0, max_depth=3):
    if depth > max_depth: return
    if torch.is_tensor(obj):
        print(f"{prefix}: Tensor{tuple(obj.shape)} {obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}: dict({len(obj)})")
        for k,v in obj.items(): print_tree(v, f"{prefix}.{k}", depth+1, max_depth)
    elif isinstance(obj, (list,tuple)):
        print(f"{prefix}: {type(obj).__name__}({len(obj)})")
        for i,v in enumerate(obj[:5]): print_tree(v, f"{prefix}[{i}]", depth+1, max_depth)
    else:
        print(f"{prefix}: {type(obj).__name__} {str(obj)[:120]}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--half", action="store_true")
    args = p.parse_args()
    obj = torch.load(args.source, map_location="cpu", weights_only=False)
    if args.inspect:
        print_tree(obj); return
    if not args.output: raise SystemExit("--output required unless --inspect")
    if not isinstance(obj, dict): raise TypeError("Canonical source must be a dict")
    missing = [k for k in REQUIRED if k not in obj]
    if missing:
        raise KeyError(f"Missing canonical keys {missing}. Available={list(obj.keys())[:100]}. Inspect cache and edit adapter_template.py")
    rt, rm, meta = obj["region_tokens"].float(), obj["region_mask"].bool(), obj["region_meta"].float()
    keys = build_global_keys(rt, rm)
    payload = {
        "protocol":"REGION_CASE_MEMORY_V2",
        "source_sha256":obj.get("source_sha256"),
        "keys":keys.half() if args.half else keys,
        "region_tokens":rt.half() if args.half else rt,
        "region_mask":rm,
        "region_meta":meta.half() if args.half else meta,
        "labels":obj["labels"].to(torch.int8),
        "event_ids":obj["event_ids"].to(torch.int16),
        "building_ids":obj["building_ids"].to(torch.int32),
        "base_probs":None if obj.get("base_probs") is None else (obj["base_probs"].half() if args.half else obj["base_probs"].float()),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print("Saved:", args.output)
    print("N:", payload["keys"].shape[0], "keys:", tuple(payload["keys"].shape), "regions:", tuple(payload["region_tokens"].shape))

if __name__ == "__main__": main()
