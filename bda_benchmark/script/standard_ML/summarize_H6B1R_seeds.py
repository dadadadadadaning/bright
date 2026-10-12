# -*- coding: utf-8 -*-
"""Validation-only decision report for H6-B1R multi-seed pilots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


P3_VALIDATION = {
    "valid_miou": 0.605019,
    "damaged_iou": 0.430309,
    "destroyed_iou": 0.628957,
    "event_mean_official_style_4class_miou": 0.527280,
    "event_worst_official_style_4class_miou": 0.407300,
}


def percent(value: object) -> float:
    return 100.0 * float(value)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize validation-only H6-B1R seeds")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 2027, 7777])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    root = Path(args.output_root)
    records: List[Dict[str, object]] = []
    header = (
        f"{'Seed':>6} {'Step':>6} {'Valid':>8} {'Dmg':>8} {'Des':>8} "
        f"{'Event':>8} {'Worst':>8} {'Gate-D':>8} {'Gate-X':>8} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))

    for seed in args.seeds:
        directory = root / f"H6_B1R_pilot{args.steps}_seed{seed}"
        summary_path = directory / "training_summary.json"
        if not summary_path.is_file():
            print(f"{seed:6d} MISSING {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("variant", "")).lower() != "b1r":
            raise SystemExit(f"unexpected variant in {summary_path}")
        if bool(summary.get("test_accessed", False)):
            raise SystemExit(f"test leakage detected in {summary_path}")
        metrics = summary["best_metrics"]
        passed = all(
            float(metrics.get(key, float("-inf"))) >= threshold
            for key, threshold in P3_VALIDATION.items()
        )
        selected_step = (
            summary.get("selection_records", {})
            .get("valid", {})
            .get("step", summary.get("completed_step"))
        )
        print(
            f"{seed:6d} {int(selected_step):6d} "
            f"{percent(metrics['valid_miou']):8.4f} "
            f"{percent(metrics['damaged_iou']):8.4f} "
            f"{percent(metrics['destroyed_iou']):8.4f} "
            f"{percent(metrics['event_mean_official_style_4class_miou']):8.4f} "
            f"{percent(metrics['event_worst_official_style_4class_miou']):8.4f} "
            f"{percent(metrics.get('safe_residual_sar_gate_damaged', 0.0)):8.4f} "
            f"{percent(metrics.get('safe_residual_sar_gate_destroyed', 0.0)):8.4f} "
            f"{('YES' if passed else 'NO'):>6}"
        )
        records.append(
            {
                "seed": seed,
                "experiment_dir": str(directory.resolve()),
                "selected_step": int(selected_step),
                "passed_all_validation_gates": passed,
                "metrics": metrics,
            }
        )

    passing = [record for record in records if record["passed_all_validation_gates"]]
    passing.sort(
        key=lambda record: float(record["metrics"]["valid_miou"]),
        reverse=True,
    )
    candidate = passing[0] if len(passing) >= 2 else None
    payload = {
        "protocol": "H6_B1R_validation_only_multiseed_v1",
        "test_accessed": False,
        "checkpoint_selector": "validation_only",
        "p3_validation_thresholds": P3_VALIDATION,
        "minimum_passing_seeds_for_test": 2,
        "eligible_for_one_final_test": candidate is not None,
        "locked_candidate": candidate,
        "records": records,
    }
    output = (
        Path(args.output_json)
        if args.output_json
        else root / "H6_B1R_validation_only_summary.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    if candidate is None:
        print("Decision: STOP. Fewer than two seeds passed every validation gate.")
    else:
        print(
            "Decision: one final test is allowed for the validation-locked candidate: "
            f"{candidate['experiment_dir']}"
        )


if __name__ == "__main__":
    main()
