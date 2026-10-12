from __future__ import annotations
import argparse, json
import torch
from .packed_memory import load_stage1_cache, build_train_memory, global_keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--min-purity", type=float, default=0.70)
    args = ap.parse_args()

    tr = load_stage1_cache(args.train_cache)
    va = load_stage1_cache(args.val_cache)
    mem = build_train_memory(tr, min_purity=args.min_purity)

    print("TRAIN MEMORY")
    print(" support rows:", len(mem.support_rows))
    for label in [-1,0,1,2]:
        print(" label", label, int((mem.labels == label).sum()))

    print(" events:", len(mem.event_map))
    print(" global key:", tuple(mem.global_keys.shape))

    print("\nVAL")
    print(" buildings:", len(va["labels"]))
    print(" candidate base-intact:", int((va["base_pred"] == 0).sum()))
    print(" region max:", int(va["region_count"].max()))
    print(" test_accessed:", va.get("test_accessed"))

if __name__ == "__main__":
    main()
