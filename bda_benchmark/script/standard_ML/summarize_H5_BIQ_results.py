# -*- coding: utf-8 -*-
"""Summarize validation-only H5 BIQ experiments against the Q0 A4 guard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


THRESHOLDS = {
    "valid_miou_delta": 0.0015,
    "damaged_iou_delta": 0.0030,
    "event_mean_delta": -0.0010,
    "loc_f1_delta": -0.0001,
}


def _read(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H5 BIQ validation experiments")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--baseline_json", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
    else:
        candidates = sorted(
            root.glob("H5_Q0_guard_seed*/initial_validation.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("run the H5 Q0 guard before summarizing")
        baseline_path = candidates[0]
    baseline = {
        key: float(value) for key, value in _read(baseline_path).items()
    }

    records_by_variant: Dict[str, Dict[str, object]] = {}
    for path in sorted(root.glob("H5_Q[12]_*/training_summary.json")):
        if "smoke" in path.parent.name.lower():
            continue
        summary = _read(path)
        metrics = summary.get("best_metrics", {})
        if not isinstance(metrics, Mapping) or not metrics:
            continue
        variant = str(summary.get("variant", "")).lower()
        deltas = {
            "valid_miou_delta": float(metrics["valid_miou"])
            - baseline["valid_miou"],
            "damaged_iou_delta": float(metrics["damaged_iou"])
            - baseline["damaged_iou"],
            "event_mean_delta": float(
                metrics["event_mean_official_style_4class_miou"]
            )
            - baseline["event_mean_official_style_4class_miou"],
            "loc_f1_delta": float(metrics["loc_f1"]) - baseline["loc_f1"],
        }
        checks = {
            key: value >= THRESHOLDS[key] for key, value in deltas.items()
        }
        record = {
            "variant": variant,
            "experiment_dir": str(path.parent.resolve()),
            "completed_step": int(summary.get("completed_step", -1)),
            "best_metrics": dict(metrics),
            "deltas": deltas,
            "checks": checks,
            "decision": "PASS" if all(checks.values()) else "REJECT",
        }
        previous = records_by_variant.get(variant)
        if previous is None or record["completed_step"] >= previous["completed_step"]:
            records_by_variant[variant] = record

    records = [records_by_variant[key] for key in sorted(records_by_variant)]
    payload = {
        "protocol": "H5_BIQ_validation_comparison_v1",
        "baseline_json": str(baseline_path.resolve()),
        "baseline_metrics": baseline,
        "thresholds": THRESHOLDS,
        "candidates": records,
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{'Variant':<8} {'Step':>6} {'Valid':>8} {'DmgIoU':>8} "
        f"{'DesIoU':>8} {'Event':>8} {'Worst':>8} {'dValid':>8} "
        f"{'dDmg':>8} {'Decision':>9}"
    )
    for record in records:
        metrics = record["best_metrics"]
        deltas = record["deltas"]
        print(
            f"{record['variant'].upper():<8} "
            f"{record['completed_step']:6d} "
            f"{100*metrics['valid_miou']:8.4f} "
            f"{100*metrics['damaged_iou']:8.4f} "
            f"{100*metrics['destroyed_iou']:8.4f} "
            f"{100*metrics['event_mean_official_style_4class_miou']:8.4f} "
            f"{100*metrics['event_worst_official_style_4class_miou']:8.4f} "
            f"{100*deltas['valid_miou_delta']:8.4f} "
            f"{100*deltas['damaged_iou_delta']:8.4f} "
            f"{record['decision']:>9}"
        )
    print(f"Saved H5 BIQ validation comparison: {destination}", flush=True)


if __name__ == "__main__":
    main()
