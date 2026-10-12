# -*- coding: utf-8 -*-
"""Strict independent validation/test evaluator for H12-DGB."""
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
from bda_benchmark.model.DamageFormer_RGBSAR_H12_DGB import (  # noqa: E402
    VALID_H12_VARIANTS,
    build_h12_dgb_damageformer,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    metrics_line,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


VALID_PROTOCOLS = ("H12_DGB_best_weights_v1", "H12_DGB_resume_v1")
ADVANCE_THRESHOLDS = {
    "damaged_iou_pp": 0.30,
    "affected_recall_pp": 0.50,
    "damaged_gate_miss_pp": -0.50,
    "final_miou_drop_pp": 0.10,
    "min_nonnegative_events": 10,
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def advancement_gate(
    observed: Mapping[str, float],
    baseline: Mapping[str, float],
    event_metrics: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    def pp(key: str) -> float:
        return 100.0 * (float(observed[key]) - float(baseline[key]))

    damaged = pp("damaged_iou")
    recall = pp("affected_recall")
    gate_miss = pp("damaged_gate_miss")
    event_mean = pp("event_mean_official_style_4class_miou")
    event_worst = pp("event_worst_official_style_4class_miou")
    final_miou = pp("miou4")
    nonnegative = 0
    for record in event_metrics.values():
        if float(record.get("miou4", 0.0)) + 1e-12 >= 0.0:
            nonnegative += 1
    checks = {
        "damaged_iou": damaged >= ADVANCE_THRESHOLDS["damaged_iou_pp"],
        "affected_recall": recall >= ADVANCE_THRESHOLDS["affected_recall_pp"],
        "damaged_gate_miss": gate_miss <= ADVANCE_THRESHOLDS["damaged_gate_miss_pp"],
        "event_mean_nondecreasing": event_mean >= -1e-9,
        "event_worst_nondecreasing": event_worst >= -1e-9,
        "enough_nonnegative_events": nonnegative
        >= ADVANCE_THRESHOLDS["min_nonnegative_events"]
        or nonnegative >= max(len(event_metrics) - 4, 0),
        "final_miou_drop_limited": final_miou
        >= -ADVANCE_THRESHOLDS["final_miou_drop_pp"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_pp": {
            "damaged_iou": damaged,
            "affected_recall": recall,
            "damaged_gate_miss": gate_miss,
            "event_mean": event_mean,
            "event_worst": event_worst,
            "final_miou": final_miou,
            "nonnegative_events": nonnegative,
            "event_count": len(event_metrics),
        },
        "thresholds": dict(ADVANCE_THRESHOLDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate H12-DGB")
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
        raise RuntimeError(f"unexpected H12 checkpoint protocol: {protocol!r}")
    if variant not in VALID_H12_VARIANTS:
        raise RuntimeError(f"unexpected H12 checkpoint variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("H12 checkpoint has no model state dictionary")

    bypass = None
    provenance = checkpoint.get("provenance", {})
    if isinstance(provenance, Mapping):
        stages = provenance.get("bypass_stages") or provenance.get("missing_new_keys")
        if isinstance(provenance.get("bypass_stages"), list):
            bypass = tuple(str(value) for value in provenance["bypass_stages"])
    model = build_h12_dgb_damageformer(
        variant, pretrained_rgb=False, bypass_stages=bypass
    )
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

    baseline = {}
    if args.baseline_metrics_json:
        baseline_payload = json.loads(
            Path(args.baseline_metrics_json).read_text(encoding="utf-8")
        )
        baseline = baseline_payload.get(
            "observed_metrics", baseline_payload.get("metrics", baseline_payload)
        )
    gate = None
    if isinstance(baseline, Mapping) and baseline:
        event_metrics = details.get("event_metrics", {})
        assert isinstance(event_metrics, Mapping)
        gate = advancement_gate(observed, baseline, event_metrics)

    payload = {
        "protocol": "H12_DGB_independent_evaluation_v1",
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
        },
        "observed_metrics": observed,
        "stored_metrics": stored if isinstance(stored, Mapping) else {},
        "deltas_vs_stored": deltas,
        "event_metrics": details.get("event_metrics", {}),
        "advancement_gate": gate,
        "metrics_line": metrics_line("H12 independent", observed),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["metrics_line"], flush=True)
    if gate is not None:
        print(
            f"Advancement gate: {'PASSED' if gate['passed'] else 'FAILED'}",
            flush=True,
        )
    print(f"Saved: {output}", flush=True)
    print(f"Test accessed: {args.split == 'test'}", flush=True)


if __name__ == "__main__":
    main()
