"""Validate a snapshot and only the downstream artifacts explicitly requested."""
import argparse
import json
from pathlib import Path

from .common import (
    digest, lists, load_front, load_metric, read_cache, source_hash, validate_front_payload,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "a1_checkpoint", "train_list", "val_list"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cache")
    parser.add_argument("--metric_checkpoint")
    parser.add_argument("--probe_root")
    args = parser.parse_args()
    lists(args.train_list, args.val_list)
    model, payload = load_front(args.checkpoint, "cpu", args.a1_checkpoint)
    del model
    validate_front_payload(payload, digest(args.train_list), digest(args.val_list))
    result = dict(snapshot_protocol=True, parent_run_complete=bool(payload.get("run_complete", False)),
                  parent_checkpoint_step=payload["step"], declared_parent_steps=payload["arguments"]["steps"],
                  snapshot_sha256=digest(args.checkpoint), test_accessed=False)
    cache = None
    if args.cache:
        _, cache, bank = read_cache(args.cache, args.checkpoint)
        if cache["smoke"] or len(cache["shards"]) != 2918:
            raise ValueError("Downstream fitting requires complete Train2918, not a smoke bank")
        if (cache["train_sha256"] != payload["train_sha256"]
                or cache["val_sha256"] != payload["val_sha256"]):
            raise ValueError("Snapshot / bank split mismatch")
        result.update(train_images=len(cache["shards"]), bank_tokens=len(bank["tokens"]))
    if args.metric_checkpoint:
        if cache is None:
            raise ValueError("Metric check requires --cache")
        _, metric = load_metric(args.metric_checkpoint, cache, args.checkpoint, "cpu")
        if metric["smoke"]:
            raise ValueError("Use the completed formal 2000-step metric")
        result["metric_step"] = metric["step"]
    if args.probe_root:
        if not args.metric_checkpoint:
            raise ValueError("Probe check requires --cache and --metric_checkpoint")
        root = Path(args.probe_root)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not summary.get("complete") or not summary.get("probe") or not summary.get("snapshot_protocol"):
            raise ValueError("Snapshot joint real-data probe did not complete")
        checks = dict(source_hash=source_hash(), front_sha256=digest(args.checkpoint),
                      a1_sha256=digest(args.a1_checkpoint), bank_sha256=cache["bank_sha256"],
                      metric_sha256=digest(args.metric_checkpoint))
        for name, expected in checks.items():
            if manifest.get(name) != expected:
                raise ValueError("Probe mismatch: " + name)
        result["matching_probe_complete"] = True
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
