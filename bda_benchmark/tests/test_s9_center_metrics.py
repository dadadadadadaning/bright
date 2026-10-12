#!/usr/bin/env python3
"""Synthetic center-count, P/R, localization, and scale-recall test."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.metrics import (
    evaluate_center_mechanism,
)


def main():
    masks = []
    small = np.zeros((64, 64), dtype=bool)
    small[4:20, 4:20] = True
    medium = np.zeros((64, 64), dtype=bool)
    medium[22:54, 4:40] = True
    large = np.zeros((128, 128), dtype=bool)
    large[40:112, 48:120] = True
    # Use a common image size.
    padded_small = np.zeros_like(large)
    padded_medium = np.zeros_like(large)
    padded_small[:64, :64] = small
    padded_medium[:64, :64] = medium
    masks = [padded_small, padded_medium, large]

    center = np.zeros((32, 32), dtype=np.float32)
    center[3, 3] = 0.95
    center[9, 5] = 0.92
    center[20, 21] = 0.90
    center[24, 25] = 0.85  # extra peak in the large building
    report = evaluate_center_mechanism(
        center,
        masks,
        min_peak_distance_native=2,
        match_tolerance_native=4.0,
    )
    assert report["building_count"] == 3
    assert report["building_peak_distribution"]["1"] >= 2
    assert report["building_peak_distribution"][">1"] == 1
    assert 0.0 <= report["center_precision"] <= 1.0
    assert 0.0 <= report["center_recall"] <= 1.0
    assert set(report["scale_recall"]) == {
        "small_lt1024",
        "medium_1024_4095",
        "large_ge4096",
    }
    print("S9 center metrics: PASS")


if __name__ == "__main__":
    main()
