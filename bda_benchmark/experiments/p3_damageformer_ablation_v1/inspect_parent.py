#!/usr/bin/env python3
"""Fail-fast inspection for a formal original-DamageFormer parent."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--expected_step", type=int, default=50000)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    summary_path = Path(args.training_summary)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload.get("model_state_dict")))
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no model state dictionary")
    step = int(payload.get("step", -1))
    variant = str(payload.get("variant", "")).lower()
    keys = [key[7:] if key.startswith("module.") else key for key in state]
    checks = {
        "variant_is_d0": variant == "d0",
        "training_completed_expected_step": int(summary.get("completed_step", -1)) == args.expected_step,
        "best_step_is_valid": 0 < step <= args.expected_step,
        "has_rgb_swin": any(key.startswith("rgb_encoder.features.") for key in keys),
        "has_sar_swin": any(key.startswith("sar_encoder.features.") for key in keys),
        "has_original_fusion": any(key.startswith("fusion_layer_3.") for key in keys),
        "has_no_micb": not any("micb" in key.lower() for key in keys),
        "has_no_roi_parameters": not any("roi" in key.lower() for key in keys),
    }
    metrics = payload.get("best_metrics", payload.get("metrics", {}))
    result = {
        "protocol": "ORIGINAL_DAMAGEFORMER_PARENT_V1",
        "checkpoint": str(path.resolve()), "sha256": sha256(path),
        "variant": variant, "step": step, "best_metrics": metrics,
        "training_summary": str(summary_path.resolve()),
        "completed_step": int(summary.get("completed_step", -1)),
        "checks": checks, "passed": all(checks.values()), "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(3)
    print(f"Verified formal DamageFormer parent: {path}")


if __name__ == "__main__":
    main()
