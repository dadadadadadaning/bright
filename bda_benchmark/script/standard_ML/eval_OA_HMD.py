# -*- coding: utf-8 -*-
"""Strict independent validation/test evaluator for OA-HMD."""
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
from bda_benchmark.model.DamageFormer_RGBSAR_OA_HMD import (  # noqa: E402
    VALID_OA_HMD_VARIANTS,
    build_oa_hmd_damageformer,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    metrics_line,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


VALID_PROTOCOLS = ("OA_HMD_best_weights_v1", "OA_HMD_resume_v1")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_metrics(payload: Mapping[str, object]) -> Mapping[str, float]:
    for key in ("observed_metrics", "metrics", "best_metrics"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value  # type: ignore[return-value]
    raise KeyError("baseline JSON has no observed_metrics/metrics/best_metrics")


def advancement_gate(
    observed: Mapping[str, float],
    baseline: Mapping[str, float],
    event_metrics: Mapping[str, Mapping[str, object]],
    baseline_events: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    keys = {
        "valid_miou": ("valid_miou", 0.30),
        "damaged_iou": ("damaged_iou", 0.50),
        "affected_recall": ("affected_recall", 0.50),
        "damaged_gate_miss": ("damaged_gate_miss", -0.50),
        "event_mean": ("event_mean_official_style_4class_miou", 0.0),
        "event_worst": ("event_worst_official_style_4class_miou", 0.0),
    }
    delta = {
        name: 100.0 * (float(observed[key]) - float(baseline[key]))
        for name, (key, _) in keys.items()
    }
    event_delta: Dict[str, float] = {}
    for event, record in event_metrics.items():
        old = baseline_events.get(event)
        if not isinstance(old, Mapping):
            continue
        event_delta[event] = 100.0 * (
            float(record["miou4"]) - float(old["miou4"])
        )
    nonnegative = sum(value >= 0.0 for value in event_delta.values())
    checks = {
        "valid_miou": delta["valid_miou"] >= 0.30,
        "damaged_iou": delta["damaged_iou"] >= 0.50,
        "affected_recall": delta["affected_recall"] >= 0.50,
        "damaged_gate_miss": delta["damaged_gate_miss"] <= -0.50,
        "event_mean_nondecreasing": delta["event_mean"] >= 0.0,
        "event_worst_nondecreasing": delta["event_worst"] >= 0.0,
        "nonnegative_events": nonnegative >= 10,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_pp": {
            **delta,
            "nonnegative_events": nonnegative,
            "comparable_events": len(event_delta),
        },
        "event_miou4_delta_pp": event_delta,
        "thresholds_pp": {
            name: threshold for name, (_, threshold) in keys.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate OA-HMD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--baseline_json", default="")
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
        raise RuntimeError(f"unexpected OA-HMD checkpoint protocol: {protocol!r}")
    if variant not in VALID_OA_HMD_VARIANTS:
        raise RuntimeError(f"unexpected OA-HMD checkpoint variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("OA-HMD checkpoint has no model state dictionary")

    model = build_oa_hmd_damageformer(variant, pretrained_rgb=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load unexpectedly failed: {incompatible}")
    model.to(device).eval()

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split=args.split,
        # h4 returns [normalised native SAR, raw display SAR]. The first
        # channel feeds H6-A2; the second reconstructs P3's exact old input.
        variant="h4",
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
        model,
        loader,
        device,
        args.ignore_index,
        max_batches=0,
        details_sink=details,
    )
    stored = checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
    deltas: Dict[str, float] = {}
    if args.split == "val" and isinstance(stored, Mapping):
        for key, value in observed.items():
            if key in stored and isinstance(stored[key], (int, float)):
                deltas[key] = float(value) - float(stored[key])

    gate = None
    if args.baseline_json:
        baseline_payload = json.loads(
            Path(args.baseline_json).read_text(encoding="utf-8")
        )
        baseline = _baseline_metrics(baseline_payload)
        baseline_events = baseline_payload.get("event_metrics", {})
        if not isinstance(baseline_events, Mapping):
            raise TypeError("baseline event_metrics must be a mapping")
        gate = advancement_gate(
            observed,
            baseline,
            details.get("event_metrics", {}),  # type: ignore[arg-type]
            baseline_events,  # type: ignore[arg-type]
        )

    payload = {
        "protocol": "OA_HMD_independent_evaluation_v1",
        "split": args.split,
        "test_accessed": args.split == "test",
        "processed_images": int(observed.get("processed_images", 0)),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
            "protocol": protocol,
            "variant": variant,
            "step": int(checkpoint.get("step", 0)),
            "selector": checkpoint.get("selector"),
            "strict_load": True,
        },
        "stored_metrics": dict(stored) if isinstance(stored, Mapping) else {},
        "observed_metrics": observed,
        "observed_minus_stored": deltas,
        "advancement_gate": gate,
        **details,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(metrics_line(f"OA-HMD independent {args.split}", observed), flush=True)
    if gate is not None:
        print(f"Advancement gate: {'PASSED' if gate['passed'] else 'FAILED'}")
    print("Strict checkpoint load: True", flush=True)
    print(f"Processed images: {payload['processed_images']}", flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
