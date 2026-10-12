# -*- coding: utf-8 -*-
"""Summarize H4 validation-only results without accessing the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


THRESHOLDS = {
    "valid_miou_delta": 0.003,
    "damaged_iou_delta": 0.005,
    "event_mean_delta": -0.002,
    "loc_f1_delta": -0.002,
}


def read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_baseline(root: Path, explicit: str) -> tuple[Path, Mapping[str, float]]:
    candidates = (
        [Path(explicit)]
        if explicit
        else sorted(
            root.glob("H4_A0_guard_seed*/initial_validation.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    if not candidates or not candidates[0].is_file():
        raise FileNotFoundError(
            "A0 initial_validation.json was not found; run the A0 guard first"
        )
    path = candidates[0]
    payload = read_json(path)
    return path, {key: float(value) for key, value in payload.items()}


def candidate_record(
    summary_path: Path,
    baseline: Mapping[str, float],
) -> Dict[str, object]:
    summary = read_json(summary_path)
    metrics = summary.get("best_metrics", {})
    if not isinstance(metrics, Mapping) or not metrics:
        raise RuntimeError(f"summary contains no best_metrics: {summary_path}")
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
    return {
        "variant": variant,
        "experiment_dir": str(summary_path.parent.resolve()),
        "completed_step": int(summary.get("completed_step", -1)),
        "best_metrics": dict(metrics),
        "deltas": deltas,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "REJECT",
    }


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H4 validation experiments")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--baseline_json", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    baseline_path, baseline = find_baseline(root, args.baseline_json)
    records_by_variant: Dict[str, Dict[str, object]] = {}
    for path in sorted(root.glob("H4_A[1-4]_*/training_summary.json")):
        if "smoke" in path.parent.name.lower():
            continue
        try:
            record = candidate_record(path, baseline)
            variant = str(record["variant"])
            previous = records_by_variant.get(variant)
            if previous is None or int(record["completed_step"]) >= int(
                previous["completed_step"]
            ):
                records_by_variant[variant] = record
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            print(f"SKIP {path}: {error}", flush=True)
    records = [
        records_by_variant[key] for key in sorted(records_by_variant)
    ]

    payload = {
        "protocol": "H4_validation_only_comparison_v1",
        "baseline_json": str(baseline_path.resolve()),
        "baseline_metrics": baseline,
        "thresholds": THRESHOLDS,
        "candidates": records,
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)

    print(
        f"{'Variant':<8} {'Step':>6} {'Valid':>8} {'DmgIoU':>8} "
        f"{'Event':>8} {'LocF1':>8} {'dValid':>8} {'dDmg':>8} "
        f"{'dEvent':>8} {'Decision':>9}"
    )
    for record in records:
        metrics = record["best_metrics"]
        deltas = record["deltas"]
        print(
            f"{record['variant'].upper():<8} "
            f"{record['completed_step']:6d} "
            f"{100*metrics['valid_miou']:8.4f} "
            f"{100*metrics['damaged_iou']:8.4f} "
            f"{100*metrics['event_mean_official_style_4class_miou']:8.4f} "
            f"{100*metrics['loc_f1']:8.4f} "
            f"{100*deltas['valid_miou_delta']:8.4f} "
            f"{100*deltas['damaged_iou_delta']:8.4f} "
            f"{100*deltas['event_mean_delta']:8.4f} "
            f"{record['decision']:>9}"
        )
    print(f"Saved H4 validation comparison: {destination}", flush=True)


if __name__ == "__main__":
    main()
