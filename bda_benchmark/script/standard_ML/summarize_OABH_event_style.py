# -*- coding: utf-8 -*-
"""Locked validation gate for OA-HSDNet event-style experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _metrics(path: str) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload.get("best_metrics")
    if not isinstance(metrics, Mapping):
        raise KeyError(f"{path} has no best_metrics mapping")
    return metrics


def _bottom_mean(events: Mapping[str, object], count: int = 3) -> float:
    values = sorted(float(value) for value in events.values())
    if len(values) < count:
        raise ValueError(f"need at least {count} event metrics, received {len(values)}")
    return sum(values[:count]) / count


def main() -> None:
    parser = argparse.ArgumentParser("OABH event-style validation gate")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = _metrics(args.baseline)
    cand = _metrics(args.candidate)
    metric_keys = {
        "valid_miou": "valid_miou",
        "damaged_iou": "damaged_iou",
        "event_mean": "event_mean_official_style_4class_miou",
        "event_worst": "event_worst_official_style_4class_miou",
    }
    deltas = {
        name: 100.0 * (float(cand[key]) - float(base[key]))
        for name, key in metric_keys.items()
    }
    base_events = base.get("event_miou4", {})
    cand_events = cand.get("event_miou4", {})
    if not isinstance(base_events, Mapping) or not isinstance(cand_events, Mapping):
        raise KeyError("both summaries must expose per-event event_miou4")
    comparable = sorted(set(base_events) & set(cand_events))
    event_deltas = {
        event: 100.0 * (float(cand_events[event]) - float(base_events[event]))
        for event in comparable
    }
    nonnegative = sum(delta >= 0.0 for delta in event_deltas.values())
    bottom3_delta = 100.0 * (_bottom_mean(cand_events) - _bottom_mean(base_events))
    diagnostics = {}
    for key in (
        "damaged_to_intact",
        "damaged_severity_to_destroyed",
        "intact_to_damaged",
        "destroyed_severity_to_damaged",
        "damaged_gate_miss",
    ):
        if key in base and key in cand:
            diagnostics[key] = 100.0 * (float(cand[key]) - float(base[key]))
    checks = {
        "event_worst_plus_0_20pp": deltas["event_worst"] >= 0.20,
        "bottom3_mean_nondecreasing": bottom3_delta >= 0.0,
        "event_mean_nondecreasing": deltas["event_mean"] >= 0.0,
        "damaged_iou_nondecreasing": deltas["damaged_iou"] >= 0.0,
        "valid_drop_limited": deltas["valid_miou"] >= -0.10,
        "at_least_10_nonnegative_events": len(comparable) == 14 and nonnegative >= 10,
    }
    report = {
        "protocol": "OABH_SAR_event_style_gate_v1",
        "test_accessed": False,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "observed_pp": {
            **deltas,
            "bottom3_mean": bottom3_delta,
            "nonnegative_events": nonnegative,
            "comparable_events": len(comparable),
        },
        "error_diagnostic_delta_pp": diagnostics,
        "event_miou4_delta_pp": event_deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
