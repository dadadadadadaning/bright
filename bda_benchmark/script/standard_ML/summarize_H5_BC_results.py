# -*- coding: utf-8 -*-
"""Summarize H5-B/C validation-only experiments against locked H4-A4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

import torch


METRICS = (
    "valid_miou",
    "miou4",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "loc_f1",
    "damaged_to_intact",
    "destroyed_to_intact",
    "event_mean_official_style_4class_miou",
    "event_worst_official_style_4class_miou",
)


def load_source(path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
    if not isinstance(metrics, Mapping):
        raise RuntimeError("source A4 checkpoint has no metrics")
    return {
        "name": "H4-A4-locked",
        "variant": "a4",
        "step": int(checkpoint.get("step", 0)),
        "metrics": {key: float(metrics[key]) for key in METRICS if key in metrics},
        "path": str(Path(path).resolve()),
    }


def collect(output_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(output_root.glob("H5_*/training_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        variant = str(summary.get("variant", "")).lower()
        if variant not in ("b1", "c1"):
            continue
        metrics = summary.get("best_metrics", {})
        if not isinstance(metrics, Mapping) or "valid_miou" not in metrics:
            continue
        records.append(
            {
                "name": path.parent.name,
                "variant": variant,
                "step": int(summary.get("completed_step", 0)),
                "metrics": {
                    key: float(metrics[key])
                    for key in METRICS
                    if key in metrics
                },
                "path": str(path.parent.resolve()),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H5-B/C validation results")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_a4_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    source = load_source(args.source_a4_checkpoint)
    records = collect(Path(args.output_root))
    baseline = source["metrics"]
    for record in records:
        metrics = record["metrics"]
        record["delta_vs_a4_pp"] = {
            key: 100.0 * (float(metrics[key]) - float(baseline[key]))
            for key in METRICS
            if key in metrics and key in baseline
        }
    best_by_variant = {}
    for variant in ("b1", "c1"):
        candidates = [
            record for record in records if record["variant"] == variant
        ]
        if candidates:
            best_by_variant[variant] = max(
                candidates,
                key=lambda item: item["metrics"]["valid_miou"],
            )

    payload = {
        "protocol": "H5_BC_validation_comparison_v1",
        "source_a4": source,
        "records": records,
        "best_by_variant": best_by_variant,
        "decision_rule": {
            "promote_if_valid_miou_delta_pp_at_least": 0.10,
            "must_not_reduce_event_mean": True,
            "must_not_reduce_damaged_or_destroyed_iou_materially": True,
            "test_accessed": False,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"{'Experiment':<42} {'Step':>6} {'Valid':>9} {'dValid':>9} "
        f"{'Damaged':>9} {'Destroy':>9} {'Event':>9}"
    )
    print("-" * 101)
    rows = [source, *records]
    for record in rows:
        metrics = record["metrics"]
        delta = 100.0 * (
            float(metrics["valid_miou"]) - float(baseline["valid_miou"])
        )
        print(
            f"{record['name']:<42} {record['step']:6d} "
            f"{100*metrics['valid_miou']:9.4f} {delta:9.4f} "
            f"{100*metrics['damaged_iou']:9.4f} "
            f"{100*metrics['destroyed_iou']:9.4f} "
            f"{100*metrics['event_mean_official_style_4class_miou']:9.4f}"
        )
    print(f"Saved H5-B/C comparison: {output}")


if __name__ == "__main__":
    main()
