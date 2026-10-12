#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser("Locked D0/A0 Test comparison")
    parser.add_argument("--d0", required=True)
    parser.add_argument("--a0", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    args = parser.parse_args()
    d0, a0 = load(args.d0), load(args.a0)
    if not d0.get("test_accessed") or not a0.get("test_accessed"):
        raise RuntimeError("both inputs must be frozen Test evaluations")
    if d0.get("processed_images") != a0.get("processed_images"):
        raise RuntimeError("D0 and A0 did not evaluate the same number of images")
    dm, am = d0["metrics"], a0["metrics"]
    keys = (
        "valid_miou", "miou4", "damaged_f1", "intact_iou", "damaged_iou",
        "destroyed_iou", "affected_recall", "damaged_gate_miss",
        "event_mean_miou4", "event_worst_miou4",
    )
    metric_delta_pp = {
        key: 100.0 * (float(am[key]) - float(dm[key]))
        for key in keys if key in am and key in dm
    }
    d0_events = dm["event_miou4_by_name"]
    a0_events = am["event_miou4_by_name"]
    common = sorted(set(d0_events) & set(a0_events))
    event_delta = {name: float(a0_events[name]) - float(d0_events[name]) for name in common}
    values = np.asarray(list(event_delta.values()), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    samples = values[rng.integers(0, values.size, size=(args.bootstrap_iterations, values.size))].mean(axis=1)
    payload = {
        "protocol": "D0_A0_TEST878_LOCKED_V1",
        "d0_checkpoint": d0["checkpoint"], "d0_checkpoint_sha256": d0["checkpoint_sha256"],
        "a0_checkpoint": a0["checkpoint"], "a0_checkpoint_sha256": a0["checkpoint_sha256"],
        "processed_images": d0["processed_images"],
        "d0_metrics": dm, "a0_metrics": am,
        "a0_minus_d0_pp": metric_delta_pp,
        "events": {
            "total": len(event_delta),
            "improved": sum(value > 0 for value in event_delta.values()),
            "nonnegative": sum(value >= 0 for value in event_delta.values()),
            "delta_pp": {key: 100.0 * value for key, value in event_delta.items()},
        },
        "equal_event_bootstrap": {
            "iterations": args.bootstrap_iterations, "seed": args.seed,
            "mean_pp": float(100.0 * values.mean()),
            "ci95_low_pp": float(100.0 * np.quantile(samples, 0.025)),
            "ci95_high_pp": float(100.0 * np.quantile(samples, 0.975)),
        },
        "test_accessed": True,
        "warning": "Frozen Test result: do not tune D0, A0, thresholds, or architecture from this output.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("# LOCKED Test | D0 vs A0")
    print(f"D0 mIoU={100*dm['valid_miou']:.4f} D-F1={100*dm['damaged_f1']:.4f}")
    print(f"A0 mIoU={100*am['valid_miou']:.4f} D-F1={100*am['damaged_f1']:.4f}")
    print(f"A0-D0 mIoU={metric_delta_pp['valid_miou']:+.4f}pp")
    print(f"Event wins={payload['events']}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
