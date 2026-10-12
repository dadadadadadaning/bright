#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from .late_core import assert_cache_identity, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    args = p.parse_args()
    tr = assert_cache_identity(args.train_cache, args.checkpoint, "train")
    va = assert_cache_identity(args.val_cache, args.checkpoint, "val")
    sha = sha256_file(args.checkpoint)
    if tr["source_checkpoint_sha256"] != va["source_checkpoint_sha256"]:
        raise RuntimeError("Train/Val cache source SHA mismatch")
    out = {
        "status": "CACHE IDENTITY PASS",
        "checkpoint_sha256": sha,
        "checkpoint_step_train": tr.get("checkpoint_report", {}).get("checkpoint_step"),
        "checkpoint_step_val": va.get("checkpoint_report", {}).get("checkpoint_step"),
        "train_buildings": int(len(tr["labels"])),
        "val_buildings": int(len(va["labels"])),
        "test_accessed": False,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
