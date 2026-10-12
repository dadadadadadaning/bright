#!/usr/bin/env python3
"""Validation-only advancement decision for controlled SR1/SR2."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


KEYS = (
    "valid_miou",
    "miou4",
    "damaged_iou",
    "destroyed_iou",
    "affected_recall",
    "damaged_gate_miss",
    "event_mean_official_style_4class_miou",
    "event_worst_official_style_4class_miou",
)


def metrics(path: str):
    source = Path(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
    else:
        value = torch.load(path, map_location="cpu", weights_only=False)
    observed = value.get("best_metrics", value.get("metrics", value))
    return value, observed


def effective_structure_transition(checkpoint) -> float | None:
    """Read SR1P's bounded, non-negative feature projection strength."""
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        return None
    matches = [
        value
        for key, value in state.items()
        if key.endswith("structure_transition") and torch.is_tensor(value)
    ]
    if len(matches) != 1:
        return None
    raw = float(matches[0].detach().float().cpu())
    return math.tanh(max(raw, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_ckpt, baseline = metrics(args.baseline)
    candidate_ckpt, candidate = metrics(args.candidate)
    gate_required = (
        "valid_miou",
        "damaged_iou",
        "damaged_gate_miss",
        "event_mean_official_style_4class_miou",
        "event_worst_official_style_4class_miou",
    )
    missing_baseline = [key for key in gate_required if key not in baseline]
    missing_candidate = [key for key in gate_required if key not in candidate]
    if missing_baseline or missing_candidate:
        raise KeyError(
            "advancement metrics are incomplete; "
            f"baseline_missing={missing_baseline}, "
            f"candidate_missing={missing_candidate}"
        )
    delta = {
        key: 100.0 * (float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)))
        for key in KEYS
    }
    baseline_events = baseline.get("event_miou4", {})
    candidate_events = candidate.get("event_miou4", {})
    comparable = sorted(set(baseline_events) & set(candidate_events))
    event_delta = {
        event: 100.0 * (candidate_events[event] - baseline_events[event])
        for event in comparable
    }
    nonnegative = sum(value >= 0.0 for value in event_delta.values())
    transition = effective_structure_transition(candidate_ckpt)
    checks = {
        # SR1P E1 is a safety-first representation test.  It must preserve
        # P3 globally while producing a measurable Damaged improvement.
        "valid_miou_nondecreasing": delta["valid_miou"] >= 0.0,
        "damaged_iou_plus_0_30pp": delta["damaged_iou"] >= 0.30,
        "damaged_gate_miss_nonincreasing": delta["damaged_gate_miss"] <= 0.0,
        "event_mean_nondecreasing": delta[
            "event_mean_official_style_4class_miou"
        ] >= 0.0,
        "event_worst_nondecreasing": delta[
            "event_worst_official_style_4class_miou"
        ] >= 0.0,
        "ten_of_fourteen_events_nondecreasing": (
            len(comparable) == 14 and nonnegative >= 10
        ),
        "structure_projection_opened": (
            transition is not None and transition > 1.0e-4
        ),
    }
    report = {
        "protocol": "P3_SRAC_advancement_gate_v1",
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "variant": candidate_ckpt.get("variant"),
        "test_accessed": False,
        "delta_pp": delta,
        "event_delta_pp": event_delta,
        "nonnegative_events": nonnegative,
        "comparable_events": len(comparable),
        "effective_structure_transition": transition,
        "checks": checks,
        "passed": all(checks.values()),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
