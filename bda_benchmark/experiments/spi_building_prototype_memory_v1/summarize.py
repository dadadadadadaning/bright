#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .core import event_bootstrap, write_json
except ImportError:
    from core import event_bootstrap, write_json


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(left, right):
    lm, rm = left["metrics"], right["metrics"]
    left_events, right_events = lm["event_miou4_by_name"], rm["event_miou4_by_name"]
    common = sorted(set(left_events) & set(right_events))
    deltas = {name: float(left_events[name] - right_events[name]) for name in common}
    return {
        "gain_pp": 100.0 * (lm["valid_miou"] - rm["valid_miou"]),
        "damaged_f1_gain_pp": 100.0 * (lm["damaged_f1"] - rm["damaged_f1"]),
        "destroyed_iou_gain_pp": 100.0 * (lm["destroyed_iou"] - rm["destroyed_iou"]),
        "affected_recall_gain_pp": 100.0 * (lm["affected_recall"] - rm["affected_recall"]),
        "damaged_gate_miss_change_pp": 100.0 * (lm["damaged_gate_miss"] - rm["damaged_gate_miss"]),
        "intact_to_affected_change_pp": 100.0 * (lm["intact_to_affected"] - rm["intact_to_affected"]),
        "worst_event_gain_pp": 100.0 * (lm["event_worst_miou4"] - rm["event_worst_miou4"]),
        "event_nonnegative": sum(value >= 0 for value in deltas.values()),
        "event_total": len(deltas), "event_deltas_pp": {k: 100.0 * v for k, v in deltas.items()},
        "event_bootstrap": event_bootstrap(deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tag", default="probe5000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    variants = ("mm0_frozen_a0", "mm1_no_memory", "mm2_memory", "mm4_event_excluded", "mm5_shuffled_labels")
    rows = {name: load(root / f"{name}_{args.tag}.json") for name in variants}
    comparisons = {
        "mm2_minus_mm1_memory_value": compare(rows["mm2_memory"], rows["mm1_no_memory"]),
        "mm4_minus_mm2_event_exclusion": compare(rows["mm4_event_excluded"], rows["mm2_memory"]),
        "mm4_minus_mm5_correct_semantics": compare(rows["mm4_event_excluded"], rows["mm5_shuffled_labels"]),
        "mm4_minus_mm0_final": compare(rows["mm4_event_excluded"], rows["mm0_frozen_a0"]),
    }
    final = comparisons["mm4_minus_mm0_final"]
    semantics = comparisons["mm4_minus_mm5_correct_semantics"]
    memory_value = comparisons["mm2_minus_mm1_memory_value"]
    high_conf_modified = rows["mm4_event_excluded"]["diagnostics"]["high_confidence_modified_fraction"]
    checks = {
        "memory_beats_parameter_control": memory_value["gain_pp"] > 0.0,
        "gain_ge_0_30pp": final["gain_pp"] >= 0.30,
        "damaged_f1_nonnegative": final["damaged_f1_gain_pp"] >= 0.0,
        "destroyed_iou_drop_le_0_10pp": final["destroyed_iou_gain_pp"] >= -0.10,
        "affected_recall_improved": final["affected_recall_gain_pp"] > 0.0,
        "damaged_gate_miss_reduced": final["damaged_gate_miss_change_pp"] < 0.0,
        "intact_to_affected_increase_le_0_10pp": final["intact_to_affected_change_pp"] <= 0.10,
        "worst_event_drop_le_0_30pp": final["worst_event_gain_pp"] >= -0.30,
        "events_nonnegative_ge_10": final["event_nonnegative"] >= 10,
        "event_bootstrap_low_gt_zero": final["event_bootstrap"]["ci95_low_pp"] > 0.0,
        "correct_memory_beats_shuffled": semantics["gain_pp"] > 0.0,
        "high_confidence_modified_fraction_le_5pct": high_conf_modified <= 0.05,
    }
    payload = {
        "protocol": "SPI_BUILDING_PROTOTYPE_MEMORY_V1", "tag": args.tag,
        "rows": {name: row["metrics"] for name, row in rows.items()},
        "comparisons": comparisons, "checks": checks, "pass": all(checks.values()),
        "test_accessed": False,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
