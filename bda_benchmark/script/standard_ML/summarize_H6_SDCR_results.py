# -*- coding: utf-8 -*-
"""Summarize H6 validation-only results without accessing the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

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


def checkpoint_metrics(path: str) -> Dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    values = checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
    result = {
        key: (float(values[key]) / 100.0 if float(values[key]) > 1.0 else float(values[key]))
        for key in METRICS
        if key in values
    }
    iou = values.get("iou") if isinstance(values, Mapping) else None
    if isinstance(iou, (list, tuple)) and len(iou) == 4:
        normalized = [float(value) / 100.0 if float(value) > 1.0 else float(value) for value in iou]
        result.update(
            {
                "intact_iou": normalized[1],
                "damaged_iou": normalized[2],
                "destroyed_iou": normalized[3],
                "valid_miou": sum(normalized[1:]) / 3.0,
            }
        )
    localization = values.get("localization", {}) if isinstance(values, Mapping) else {}
    confusion = localization.get("confusion") if isinstance(localization, Mapping) else None
    if confusion:
        _, fp = map(float, confusion[0])
        fn, tp = map(float, confusion[1])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        result["loc_f1"] = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H6-SDCR validation results")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_a4_checkpoint", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    rows: Dict[str, Mapping[str, object]] = {
        "P3": {"metrics": checkpoint_metrics(args.p3_checkpoint)},
        "H4-A4": {"metrics": checkpoint_metrics(args.source_a4_checkpoint)},
    }
    root = Path(args.output_root)
    for path in sorted(root.glob("H6_*/training_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("protocol") != "H6_SDCR_training_summary_v1":
            continue
        rows[path.parent.name] = {
            "variant": summary.get("variant"),
            "step": summary.get("completed_step"),
            "metrics": summary.get("best_metrics", {}),
            "path": str(path.parent.resolve()),
        }

    payload = {
        "protocol": "H6_SDCR_validation_comparison_v1",
        "test_accessed": False,
        "rows": rows,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    header = (
        f"{'Model':<38} {'Valid':>8} {'mIoU4':>8} {'Dmg':>8} "
        f"{'Des':>8} {'LocF1':>8} {'Event':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        def pct(key: str) -> float:
            return 100.0 * float(metrics.get(key, float("nan")))
        print(
            f"{name:<38} {pct('valid_miou'):8.4f} {pct('miou4'):8.4f} "
            f"{pct('damaged_iou'):8.4f} {pct('destroyed_iou'):8.4f} "
            f"{pct('loc_f1'):8.4f} "
            f"{pct('event_mean_official_style_4class_miou'):8.4f}"
        )
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
