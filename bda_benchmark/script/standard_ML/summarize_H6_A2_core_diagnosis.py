# -*- coding: utf-8 -*-
"""Summarize the H6 native-SAR stage probe and F0/F1/F2 freeze ablation.

This is a validation-only decision report.  It never reads a test result and
does not select a checkpoint using test performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


RUNS = (
    ("F0", "f0_shallow_frozen"),
    ("F1", "f1_stage2_to_4"),
    ("F2", "f2_all_sar"),
)


def load_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        raise KeyError(f"required metric {key!r} is absent")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--probe_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    probe = load_json(args.probe_json)
    if probe.get("test_accessed") is not False:
        raise RuntimeError("SAR stage probe must explicitly declare test_accessed=False")
    stages = probe.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RuntimeError("SAR stage probe has no stage records")

    stage_records = []
    for item in stages:
        if not isinstance(item, Mapping):
            continue
        stage_records.append(
            {
                "stage": int(item["stage"]),
                "damage_3class_accuracy": item.get("damage_3class_accuracy"),
                "affected_auroc": item.get("affected_auroc"),
                "affected_auprc": item.get("affected_auprc"),
                "severity_auroc": item.get("severity_auroc"),
                "severity_auprc": item.get("severity_auprc"),
                "event_macro_accuracy": item.get("event_macro_accuracy"),
                "sampled_building_cells": int(item["sampled_building_cells"]),
            }
        )
    valid_auc = [
        item for item in stage_records
        if isinstance(item["affected_auroc"], (int, float))
    ]
    best_stage = max(valid_auc, key=lambda item: item["affected_auroc"]) if valid_auc else None

    records: Dict[str, Dict[str, object]] = {}
    for label, mode in RUNS:
        directory = args.output_root / f"H6_A2_{label}_freeze{args.steps}_seed{args.seed}"
        summary = load_json(directory / "training_summary.json")
        if summary.get("test_accessed") not in (False, None):
            raise RuntimeError(f"{label} training summary accessed test data")
        independent_path = directory / "independent_val.json"
        independent = load_json(independent_path) if independent_path.is_file() else None
        if independent is not None:
            if independent.get("split") != "val" or independent.get("test_accessed") is not False:
                raise RuntimeError(f"{label} independent result is not validation-only")
            metrics = independent.get("observed_metrics")
        else:
            metrics = summary.get("best_metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"{label} has no best_metrics")
        selected = summary.get("selection_records", {})
        selected_valid = selected.get("valid", {}) if isinstance(selected, Mapping) else {}
        selected_step = (
            int(independent.get("checkpoint", {}).get("step", 0))
            if independent is not None
            else int(selected_valid.get("step", 0))
        )
        records[label] = {
            "mode": mode,
            "directory": str(directory),
            "completed_step": int(summary.get("completed_step", 0)),
            "selected_step": selected_step,
            "metrics_source": str(independent_path if independent is not None else directory / "training_summary.json"),
            "valid_miou": metric(metrics, "valid_miou"),
            "miou4": metric(metrics, "miou4"),
            "damaged_iou": metric(metrics, "damaged_iou"),
            "destroyed_iou": metric(metrics, "destroyed_iou"),
            "affected_recall": metric(metrics, "affected_recall"),
            "damaged_gate_miss": metric(metrics, "damaged_gate_miss"),
            "event_mean": metric(metrics, "event_mean_official_style_4class_miou"),
            "event_worst": metric(metrics, "event_worst_official_style_4class_miou"),
        }

    baseline = records["F0"]
    thresholds_pp = {
        "damaged_iou": 0.30,
        "affected_recall": 0.50,
        "damaged_gate_miss": -0.50,
        "event_mean": 0.0,
        "event_worst": 0.0,
    }
    for label in ("F1", "F2"):
        current = records[label]
        deltas_pp = {
            key: 100.0 * (float(current[key]) - float(baseline[key]))
            for key in (
                "valid_miou", "miou4", "damaged_iou", "destroyed_iou",
                "affected_recall", "damaged_gate_miss", "event_mean", "event_worst",
            )
        }
        checks = {
            "damaged_iou": deltas_pp["damaged_iou"] >= thresholds_pp["damaged_iou"],
            "affected_recall": deltas_pp["affected_recall"] >= thresholds_pp["affected_recall"],
            "damaged_gate_miss": deltas_pp["damaged_gate_miss"] <= thresholds_pp["damaged_gate_miss"],
            "event_mean": deltas_pp["event_mean"] >= 0.0,
            "event_worst": deltas_pp["event_worst"] >= 0.0,
        }
        current["delta_vs_f0_pp"] = deltas_pp
        current["advancement_checks"] = checks
        current["advancement_passed"] = all(checks.values())

    passing = [label for label in ("F1", "F2") if records[label]["advancement_passed"]]
    if passing:
        winner = max(
            passing,
            key=lambda label: (
                records[label]["event_worst"], records[label]["event_mean"],
                records[label]["damaged_iou"], records[label]["valid_miou"],
            ),
        )
        decision = f"FREEZING_IS_A_BOTTLENECK__ADVANCE_{winner}"
    else:
        winner = None
        decision = "NO_FREEZE_CANDIDATE_PASSED__DO_NOT_ACCESS_TEST"

    payload = {
        "protocol": "H6_A2_native_SAR_core_diagnosis_v1",
        "test_accessed": False,
        "steps": args.steps,
        "seed": args.seed,
        "probe": {
            "path": str(args.probe_json),
            "best_affected_stage": best_stage,
            "stages": stage_records,
        },
        "thresholds_pp": thresholds_pp,
        "freeze_ablation": records,
        "winner": winner,
        "decision": decision,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"{'Run':<4} {'Mode':<20} {'Step':>6} {'Valid':>9} {'Dmg':>9} "
        f"{'Recall':>9} {'GateMiss':>10} {'Event':>9} {'Worst':>9} {'Pass':>6}"
    )
    print("-" * 106)
    for label, _ in RUNS:
        item = records[label]
        passed = "BASE" if label == "F0" else ("YES" if item["advancement_passed"] else "NO")
        print(
            f"{label:<4} {item['mode']:<20} {item['selected_step']:6d} "
            f"{100*item['valid_miou']:9.4f} {100*item['damaged_iou']:9.4f} "
            f"{100*item['affected_recall']:9.4f} {100*item['damaged_gate_miss']:10.4f} "
            f"{100*item['event_mean']:9.4f} {100*item['event_worst']:9.4f} {passed:>6}"
        )
    if best_stage is not None:
        print(
            "Best frozen-SAR affected stage: "
            f"Stage{best_stage['stage']} (AUROC={100*best_stage['affected_auroc']:.3f})"
        )
    print(f"Decision: {decision}")
    print(f"Saved: {args.output_json}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
