#!/usr/bin/env python3
"""Compare frozen D0, A0 and A1 on the one-shot labelled Test878 split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = (
    "valid_miou",
    "miou4",
    "damaged_f1",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "affected_recall",
    "damaged_gate_miss",
    "event_mean_miou4",
    "event_worst_miou4",
)


def load(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("test_accessed") is not True:
        raise RuntimeError(f"not a locked Test evaluation: {path}")
    return payload


def compare(left: dict, right: dict, seed: int, repeats: int) -> dict:
    """Return right-minus-left deltas (candidate minus reference)."""
    lm, rm = left["metrics"], right["metrics"]
    metric_delta_pp = {
        key: 100.0 * (float(rm[key]) - float(lm[key]))
        for key in METRIC_KEYS
        if key in lm and key in rm
    }
    left_events = lm["event_miou4_by_name"]
    right_events = rm["event_miou4_by_name"]
    common = sorted(set(left_events) & set(right_events))
    event_delta_pp = {
        name: 100.0 * (float(right_events[name]) - float(left_events[name]))
        for name in common
    }
    values = np.asarray(list(event_delta_pp.values()), dtype=np.float64)
    if values.size == 0:
        raise RuntimeError("no common event metrics were found")
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, values.size, size=(repeats, values.size))
    ].mean(axis=1)
    return {
        "metric_delta_pp": metric_delta_pp,
        "events": {
            "total": int(values.size),
            "improved": int(np.sum(values > 0.0)),
            "nonnegative": int(np.sum(values >= 0.0)),
            "delta_pp": event_delta_pp,
        },
        "equal_event_bootstrap": {
            "iterations": repeats,
            "seed": seed,
            "mean_pp": float(values.mean()),
            "ci95_low_pp": float(np.quantile(samples, 0.025)),
            "ci95_high_pp": float(np.quantile(samples, 0.975)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser("Locked D0/A0/A1 Test878 comparison")
    parser.add_argument("--d0", required=True)
    parser.add_argument("--a0", required=True)
    parser.add_argument("--a1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    args = parser.parse_args()

    models = {"d0": load(args.d0), "a0": load(args.a0), "a1": load(args.a1)}
    image_counts = {item.get("processed_images") for item in models.values()}
    if len(image_counts) != 1:
        raise RuntimeError(f"models evaluated different image counts: {image_counts}")

    comparisons = {
        "a0_minus_d0": compare(
            models["d0"], models["a0"], args.seed, args.bootstrap_iterations
        ),
        "a1_minus_a0": compare(
            models["a0"], models["a1"], args.seed + 1, args.bootstrap_iterations
        ),
        "a1_minus_d0": compare(
            models["d0"], models["a1"], args.seed + 2, args.bootstrap_iterations
        ),
    }
    payload = {
        "protocol": "D0_A0_A1_TEST878_LOCKED_V1",
        "processed_images": int(next(iter(image_counts))),
        "models": {
            name: {
                "checkpoint": item["checkpoint"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "metrics": item["metrics"],
            }
            for name, item in models.items()
        },
        "comparisons": comparisons,
        "test_accessed": True,
        "warning": (
            "Frozen one-shot Test878 result. Do not tune D0, A0, A1, thresholds, "
            "training schedules, or architecture from this output."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("# LOCKED Test878 | D0 vs A0 vs A1")
    for name, item in models.items():
        metrics = item["metrics"]
        print(
            f"{name.upper():2s} mIoU={100*metrics['valid_miou']:.4f} "
            f"D-F1={100*metrics['damaged_f1']:.4f} "
            f"I/D/X IoU={100*metrics['intact_iou']:.4f}/"
            f"{100*metrics['damaged_iou']:.4f}/"
            f"{100*metrics['destroyed_iou']:.4f}"
        )
    for name, result in comparisons.items():
        delta = result["metric_delta_pp"]
        events = result["events"]
        ci = result["equal_event_bootstrap"]
        print(
            f"{name}: mIoU={delta.get('valid_miou', float('nan')):+.4f}pp, "
            f"D-F1={delta.get('damaged_f1', float('nan')):+.4f}pp, "
            f"events={events['nonnegative']}/{events['total']}, "
            f"event-CI=[{ci['ci95_low_pp']:+.4f}, {ci['ci95_high_pp']:+.4f}]pp"
        )
    print(f"Saved: {destination}")
    print("Test accessed: YES. These results are frozen and must not be used for tuning.")


if __name__ == "__main__":
    main()
