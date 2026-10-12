# -*- coding: utf-8 -*-
"""Strict independent validation/test evaluator for H14-UWDC."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H14_UWDC import (  # noqa: E402
    VALID_H14_VARIANTS,
    build_h14_uwdc_damageformer,
)
from bda_benchmark.script.standard_ML.eval_H13_SPER import (  # noqa: E402
    advancement_gate,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    metrics_line,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)

VALID_PROTOCOLS = ("H14_UWDC_best_weights_v1", "H14_UWDC_resume_v1")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate H14-UWDC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--baseline_metrics_json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol not in VALID_PROTOCOLS:
        raise RuntimeError(f"unexpected H14 checkpoint protocol: {protocol!r}")
    if variant not in VALID_H14_VARIANTS:
        raise RuntimeError(f"unexpected H14 checkpoint variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("H14 checkpoint has no model state dictionary")
    model = build_h14_uwdc_damageformer(variant, pretrained_rgb=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load unexpectedly failed: {incompatible}")
    model.to(device).eval()

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split=args.split,
        variant="h3",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    details: Dict[str, object] = {}
    observed = evaluate_h10(
        model, loader, device, args.ignore_index, max_batches=0, details_sink=details
    )
    baseline = {}
    baseline_events = {}
    if args.baseline_metrics_json:
        payload = json.loads(
            Path(args.baseline_metrics_json).read_text(encoding="utf-8")
        )
        baseline = payload.get("observed_metrics", payload.get("metrics", payload))
        baseline_events = payload.get("event_metrics", {})
    gate = None
    if isinstance(baseline, Mapping) and baseline:
        event_metrics = details.get("event_metrics", {})
        if not isinstance(event_metrics, Mapping):
            event_metrics = {}
        if not isinstance(baseline_events, Mapping):
            baseline_events = {}
        gate = advancement_gate(observed, baseline, event_metrics, baseline_events)

    output_payload = {
        "protocol": "H14_UWDC_independent_evaluation_v1",
        "split": args.split,
        "test_accessed": args.split == "test",
        "processed_images": int(observed.get("processed_images", 0)),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
            "protocol": protocol,
            "variant": variant,
            "selector": checkpoint.get("selector"),
            "step": checkpoint.get("step"),
            "strict_load": True,
        },
        "observed_metrics": observed,
        "stored_metrics": checkpoint.get("metrics", checkpoint.get("best_metrics", {})),
        "event_metrics": details.get("event_metrics", {}),
        "advancement_gate": gate,
        "metrics_line": metrics_line("H14 independent", observed),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(output_payload["metrics_line"], flush=True)
    if gate is not None:
        print(f"Advancement gate: {'PASSED' if gate['passed'] else 'FAILED'}", flush=True)
    print("Strict checkpoint load: True", flush=True)
    print(f"Processed images: {output_payload['processed_images']}", flush=True)
    print(f"Saved: {output}", flush=True)
    print(f"Test accessed: {args.split == 'test'}", flush=True)


if __name__ == "__main__":
    main()
