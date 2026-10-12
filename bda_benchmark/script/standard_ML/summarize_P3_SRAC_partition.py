#!/usr/bin/env python3
"""Validation-only gate for SR1PA boundary-aware partition pretraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


REQUIRED = (
    "srac_assignment_iterations",
    "srac_partition_entropy",
    "srac_effective_candidates",
    "srac_background_leakage_fraction",
    "srac_candidate_node_count",
    "srac_building_iou",
    "srac_boundary_f1",
    "srac_partition_temperature",
    "processed_images",
)


def read_metrics(path: str | Path) -> Dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = value.get("metrics", value)
    missing = [key for key in REQUIRED if key not in observed]
    if missing:
        raise KeyError(f"partition audit metrics are missing: {missing}")
    return dict(observed)


def build_report(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> Dict[str, object]:
    if int(baseline["processed_images"]) != int(candidate["processed_images"]):
        raise ValueError(
            "baseline/candidate processed different image counts: "
            f"{baseline['processed_images']} vs {candidate['processed_images']}"
        )
    baseline_iterations = int(round(float(baseline["srac_assignment_iterations"])))
    candidate_iterations = int(round(float(candidate["srac_assignment_iterations"])))
    if baseline_iterations != candidate_iterations:
        raise ValueError(
            "baseline/candidate use different assignment iteration counts: "
            f"{baseline_iterations} vs {candidate_iterations}"
        )
    delta = {
        "partition_entropy": float(candidate["srac_partition_entropy"])
        - float(baseline["srac_partition_entropy"]),
        "effective_candidates": float(candidate["srac_effective_candidates"])
        - float(baseline["srac_effective_candidates"]),
        "background_leakage_fraction": float(
            candidate["srac_background_leakage_fraction"]
        )
        - float(baseline["srac_background_leakage_fraction"]),
        "building_iou": float(candidate["srac_building_iou"])
        - float(baseline["srac_building_iou"]),
        "boundary_f1": float(candidate["srac_boundary_f1"])
        - float(baseline["srac_boundary_f1"]),
    }
    checks = {
        # Absolute compactness checks prevent a tiny relative gain from
        # admitting another high-entropy fuzzy grid.
        "partition_entropy_below_0_65": (
            float(candidate["srac_partition_entropy"]) <= 0.65
        ),
        "effective_candidates_below_4_2": (
            float(candidate["srac_effective_candidates"]) <= 4.20
        ),
        "background_leakage_below_0_25": (
            float(candidate["srac_background_leakage_fraction"]) <= 0.25
        ),
        # Relative geometry checks prove that supervision changed the
        # partition, rather than merely benefiting from temperature annealing.
        "building_iou_plus_0_02": delta["building_iou"] >= 0.02,
        "boundary_f1_plus_0_03": delta["boundary_f1"] >= 0.03,
        # The iterative experiment must not trade away the geometry already
        # achieved by the completed one-shot SR1PA validation run.
        "building_iou_at_least_sr1pa_0_6656": (
            float(candidate["srac_building_iou"]) >= 0.6656
        ),
        "boundary_f1_at_least_sr1pa_0_5789": (
            float(candidate["srac_boundary_f1"]) >= 0.5789
        ),
        "temperature_schedule_completed": (
            float(candidate["srac_partition_temperature"]) <= 0.16
        ),
    }
    next_stage = "SR1PIB" if candidate_iterations > 1 else "SR1PB"
    return {
        "protocol": "P3_SRAC_partition_gate_v2",
        "test_accessed": False,
        "processed_images": int(candidate["processed_images"]),
        "baseline": {key: baseline[key] for key in REQUIRED},
        "candidate": {key: candidate[key] for key in REQUIRED},
        "delta": delta,
        "checks": checks,
        "passed": all(checks.values()),
        "next_step": (
            f"PROCEED_TO_{next_stage}_SAFE_PROJECTION"
            if all(checks.values())
            else "STOP_OR_REDESIGN_PARTITION_DO_NOT_OPEN_PROJECTION"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        read_metrics(args.baseline),
        read_metrics(args.candidate),
    )
    report["baseline_path"] = str(Path(args.baseline).resolve())
    report["candidate_path"] = str(Path(args.candidate).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
