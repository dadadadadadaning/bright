# -*- coding: utf-8 -*-
"""Compare two validation-selected OABH runs and enforce the advancement gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _summary(path: str) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload.get("best_metrics")
    if not isinstance(metrics, Mapping):
        raise KeyError(f"{path} has no best_metrics mapping")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("OABH validation-only advancement report")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    baseline = _summary(args.baseline)
    candidate = _summary(args.candidate)
    base = baseline["best_metrics"]
    cand = candidate["best_metrics"]

    names = {
        "valid_miou": "valid_miou",
        "damaged_iou": "damaged_iou",
        "damaged_gate_miss": "damaged_gate_miss",
        "event_mean": "event_mean_official_style_4class_miou",
        "event_worst": "event_worst_official_style_4class_miou",
    }
    deltas = {
        name: 100.0 * (float(cand[key]) - float(base[key]))
        for name, key in names.items()
    }
    base_events = base.get("event_miou4", {})
    cand_events = cand.get("event_miou4", {})
    comparable = sorted(set(base_events) & set(cand_events))
    event_deltas = {
        event: 100.0 * (float(cand_events[event]) - float(base_events[event]))
        for event in comparable
    }
    nonnegative = sum(delta >= 0.0 for delta in event_deltas.values())
    checks = {
        "valid_miou": deltas["valid_miou"] >= 0.20,
        "damaged_iou": deltas["damaged_iou"] >= 0.30,
        "damaged_gate_miss_not_increasing": deltas["damaged_gate_miss"] <= 0.0,
        "event_mean_nondecreasing": deltas["event_mean"] >= 0.0,
        "event_worst_nondecreasing": deltas["event_worst"] >= 0.0,
        "nonnegative_events": len(comparable) > 0 and nonnegative >= 10,
    }
    report = {
        "protocol": "OABH_advancement_gate_v1",
        "test_accessed": False,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "observed_pp": {
            **deltas,
            "nonnegative_events": nonnegative,
            "comparable_events": len(comparable),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "event_miou4_delta_pp": event_deltas,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
