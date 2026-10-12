"""Read-only formal artifact checks before starting a three-head job."""
import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common as parent
from .common import digest, lists, load_metric, read_cache, source_hash


def inspect_parent(parent_run, a1_checkpoint, train_list, val_list):
    run = Path(parent_run)
    checkpoint = run / "front" / "best.pth"
    for path in (checkpoint, Path(a1_checkpoint), run / "train_memory" / "manifest.json",
                 run / "train_memory" / "bank.pt"):
        if not path.is_file():
            raise ValueError("Required completed-parent artifact is missing: " + str(path))
    lists(train_list, val_list)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != parent.FRONT_FORMAT:
        raise ValueError("Use the original completed transformer_v1 front, not the Q-annealing experiment")
    if (not payload.get("run_complete")
            or payload.get("arguments", {}).get("steps", 0) < 1
            or payload.get("completed_steps") != payload.get("arguments", {}).get("steps")
            or payload.get("smoke") is not False or payload.get("test_accessed") is not False):
        raise ValueError("The parent front must be completed and formal, with no Test access")
    if payload.get("source_hash") != parent.source_hash():
        raise ValueError("Parent source changed; restore matching archived source, not a looser check")
    if payload.get("a1_sha256") != digest(a1_checkpoint):
        raise ValueError("The provided A1 is not the exact original parent A1")
    if (payload.get("train_sha256") != digest(train_list)
            or payload.get("val_sha256") != digest(val_list)):
        raise ValueError("Parent Train/Val split hashes differ")
    _, cache, bank = read_cache(run / "train_memory", checkpoint)
    if cache.get("smoke") or len(cache.get("shards", [])) != 2918:
        raise ValueError("A complete formal Train2918 cache is required")
    if not len(bank["tokens"]):
        raise ValueError("Memory bank is empty")
    return checkpoint, cache, dict(parent_front_complete=True, cache_complete=True,
                                   front_step=payload["step"], completed_steps=payload["completed_steps"],
                                   train_images=len(cache["shards"]), bank_tokens=len(bank["tokens"]),
                                   test_accessed=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent_run", "a1_checkpoint", "train_list", "val_list"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--metric_checkpoint")
    parser.add_argument("--probe_root", help="require successful matching short probe before formal run")
    args = parser.parse_args()
    front, cache, result = inspect_parent(args.parent_run, args.a1_checkpoint, args.train_list, args.val_list)
    if args.metric_checkpoint:
        _, payload = load_metric(args.metric_checkpoint, cache, front, "cpu")
        if payload.get("smoke"):
            raise ValueError("J3 requires the formal 2000-update metric")
        result["metric_complete"] = True
        result["metric_step"] = payload["step"]
    if args.probe_root:
        if not args.metric_checkpoint:
            raise ValueError("Probe verification also needs metric_checkpoint")
        probe = Path(args.probe_root)
        summary = json.loads((probe / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((probe / "manifest.json").read_text(encoding="utf-8"))
        if not summary.get("complete") or not summary.get("probe"):
            raise ValueError("The short real-data probe did not finish")
        checks = dict(source_hash=source_hash(), front_sha256=digest(front),
                      a1_sha256=digest(args.a1_checkpoint), bank_sha256=cache["bank_sha256"],
                      metric_sha256=digest(args.metric_checkpoint))
        for key, value in checks.items():
            if manifest.get(key) != value:
                raise ValueError("Probe artifact mismatch: " + key)
        result["matching_probe_complete"] = True
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
