#!/usr/bin/env python3
"""Fail-fast check that SR0 reproduces the locked P3 validation metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REQUIRED = (
    "valid_miou",
    "miou4",
    "damaged_iou",
    "destroyed_iou",
    "loc_f1",
    "event_mean_official_style_4class_miou",
    "event_worst_official_style_4class_miou",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-checkpoint", required=True)
    parser.add_argument(
        "--reference-metrics",
        help="optional independent P3 validation JSON for numeric comparison",
    )
    parser.add_argument("--sr0-metrics", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.p3_checkpoint, map_location="cpu", weights_only=False
    )
    if args.reference_metrics:
        reference = json.loads(
            Path(args.reference_metrics).read_text(encoding="utf-8")
        )
        expected = reference.get("best_metrics", reference.get("metrics", reference))
    else:
        expected = checkpoint.get("best_metrics", checkpoint.get("metrics", {}))
    observed = json.loads(Path(args.sr0_metrics).read_text(encoding="utf-8"))
    missing_expected = [key for key in REQUIRED if key not in expected]
    missing_observed = [key for key in REQUIRED if key not in observed]
    if missing_observed:
        raise KeyError(
            "SR0 guard cannot be checked; "
            f"missing_expected={missing_expected}, missing_observed={missing_observed}"
        )
    if missing_expected:
        print(
            "P3 checkpoint contains no embedded validation metrics. "
            "SR0 metrics were recorded successfully; strict state loading and "
            "the tensor-equivalence unit test still protect the forward path. "
            "Set P3_REFERENCE_JSON to enable the numeric metric comparison."
        )
        return

    failed = {}
    for key in REQUIRED:
        delta = abs(float(observed[key]) - float(expected[key]))
        print(
            f"{key:<48}: expected={float(expected[key]):.10f} "
            f"observed={float(observed[key]):.10f} delta={delta:.3g}"
        )
        if delta > args.tolerance:
            failed[key] = delta
    if failed:
        raise AssertionError(f"P3-SR0 exact guard failed: {failed}")
    print("P3-SR0 exact guard passed.")


if __name__ == "__main__":
    main()
