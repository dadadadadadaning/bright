#!/usr/bin/env python3
"""Validation-only admission gate for SR1PAA damage affinity evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


REQUIRED = (
    "srac_affinity_loss",
    "srac_affinity_same_mean",
    "srac_affinity_different_mean",
    "srac_affinity_margin",
    "srac_affinity_balanced_accuracy",
    "srac_affinity_damage_boundary_recall",
    "srac_affinity_valid_pairs",
    "srac_affinity_same_count",
    "srac_affinity_different_count",
    "srac_affinity_damage_boundary_count",
    "srac_affinity_negative_fraction",
    "processed_images",
)


def read_metrics(path: str | Path) -> Dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = value.get("metrics", value)
    missing = [key for key in REQUIRED if key not in metrics]
    if missing:
        raise KeyError(f"affinity audit metrics are missing: {missing}")
    return dict(metrics)


def build_report(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> Dict[str, object]:
    if int(baseline["processed_images"]) != int(candidate["processed_images"]):
        raise ValueError("baseline and candidate processed different image counts")
    delta = {
        key: float(candidate[key]) - float(baseline[key])
        for key in REQUIRED
        if key != "processed_images"
    }
    checks = {
        "affinity_loss_decreased": delta["srac_affinity_loss"] <= -0.02,
        "balanced_accuracy_at_least_0_65": (
            float(candidate["srac_affinity_balanced_accuracy"]) >= 0.65
        ),
        "damage_boundary_recall_at_least_0_60": (
            float(candidate["srac_affinity_damage_boundary_recall"]) >= 0.60
        ),
        "same_different_margin_at_least_0_15": (
            float(candidate["srac_affinity_margin"]) >= 0.15
        ),
        "valid_pair_support_present": (
            float(candidate["srac_affinity_valid_pairs"]) > 0.0
        ),
        "same_and_different_pairs_present": (
            float(candidate["srac_affinity_same_count"]) > 0.0
            and float(candidate["srac_affinity_different_count"]) > 0.0
        ),
        "damaged_boundary_support_present": (
            float(candidate["srac_affinity_damage_boundary_count"]) > 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": "P3_SRAC_damage_affinity_gate_v1",
        "test_accessed": False,
        "processed_images": int(candidate["processed_images"]),
        "baseline": {key: baseline[key] for key in REQUIRED},
        "candidate": {key: candidate[key] for key in REQUIRED},
        "delta": delta,
        "checks": checks,
        "passed": passed,
        "next_step": (
            "PROCEED_TO_AFFINITY_REGION_CONSTRUCTION_VALIDATION_ONLY"
            if passed
            else "STOP_OR_REDESIGN_AFFINITY_DO_NOT_CHANGE_P3"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(read_metrics(args.baseline), read_metrics(args.candidate))
    report["baseline_path"] = str(Path(args.baseline).resolve())
    report["candidate_path"] = str(Path(args.candidate).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
