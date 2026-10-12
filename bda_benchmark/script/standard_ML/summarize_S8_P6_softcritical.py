#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation-only advancement gate for S8-P6 Soft-Critical MICB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


def _load_metrics(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    metrics = payload.get("best_metrics", payload)
    if not isinstance(metrics, dict):
        raise TypeError(f"cannot find metrics mapping in {path}")
    return metrics


def _number(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        raise KeyError(f"required metric {key!r} is missing")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize S8-P6 validation advancement")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = _load_metrics(args.baseline)
    candidate = _load_metrics(args.candidate)
    keys = (
        "valid_miou",
        "miou4",
        "damaged_iou",
        "destroyed_iou",
        "affected_recall",
        "damaged_gate_miss",
        "event_mean_official_style_4class_miou",
        "event_worst_official_style_4class_miou",
    )
    delta_pp = {
        key: 100.0 * (_number(candidate, key) - _number(baseline, key))
        for key in keys
    }
    base_events = baseline.get("event_miou_by_name", {})
    candidate_events = candidate.get("event_miou_by_name", {})
    if not isinstance(base_events, dict) or not isinstance(candidate_events, dict):
        raise TypeError("event_miou_by_name must be a mapping")
    common_events = sorted(set(base_events) & set(candidate_events))
    event_delta_pp = {
        event: 100.0 * (float(candidate_events[event]) - float(base_events[event]))
        for event in common_events
    }
    nonnegative_events = sum(delta >= 0.0 for delta in event_delta_pp.values())

    checks = {
        "valid_miou_nondecreasing": delta_pp["valid_miou"] >= 0.0,
        "damaged_iou_plus_0_30pp": delta_pp["damaged_iou"] >= 0.30,
        "affected_recall_plus_0_50pp": delta_pp["affected_recall"] >= 0.50,
        "damaged_gate_miss_minus_0_50pp": delta_pp["damaged_gate_miss"] <= -0.50,
        "event_mean_nondecreasing": (
            delta_pp["event_mean_official_style_4class_miou"] >= 0.0
        ),
        "event_worst_nondecreasing": (
            delta_pp["event_worst_official_style_4class_miou"] >= 0.0
        ),
        "ten_events_nondecreasing": (
            len(common_events) >= 14 and nonnegative_events >= 10
        ),
    }
    passed = all(checks.values())
    result = {
        "protocol": "S8_P6_softcritical_advancement_gate_v1",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "delta_pp": delta_pp,
        "event_delta_pp": event_delta_pp,
        "nonnegative_events": nonnegative_events,
        "comparable_events": len(common_events),
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_TO_REPLICATION_BEFORE_TEST"
            if passed
            else "STOP_P6_DO_NOT_ADD_PIXEL_OCCURRENCE_OR_TEST"
        ),
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
