# -*- coding: utf-8 -*-
"""Compute exact train-only 8-bit SAR statistics with a bounded histogram."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    read_data_list,
    to_single_channel_sar,
)


def _quantile_from_histogram(histogram: np.ndarray, probability: float) -> float:
    cumulative = np.cumsum(histogram, dtype=np.int64)
    target = int(np.floor(probability * max(int(cumulative[-1]) - 1, 0)))
    return float(np.searchsorted(cumulative, target + 1))


def main() -> None:
    parser = argparse.ArgumentParser("Compute train-only SAR statistics")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset_path)
    list_path = Path(args.train_data_list_path)
    names = read_data_list(list_path)
    histogram = np.zeros(256, dtype=np.int64)
    nonfinite = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    for index, name in enumerate(names, start=1):
        path = dataset / "post-event" / f"{name}_post_disaster.tif"
        sar = to_single_channel_sar(np.asarray(imageio.imread(path), dtype=np.float32))
        finite = np.isfinite(sar)
        nonfinite += int((~finite).sum())
        values = sar[finite]
        if values.size:
            raw_min = min(raw_min, float(values.min()))
            raw_max = max(raw_max, float(values.max()))
            if values.min() < 0 or values.max() > 255:
                raise ValueError(
                    f"{path} is not the declared 8-bit display SAR representation"
                )
            bins = np.rint(values).astype(np.uint8)
            histogram += np.bincount(bins, minlength=256)
        if index % 100 == 0 or index == len(names):
            print(f"histogram pass: {index}/{len(names)}", flush=True)

    p01 = _quantile_from_histogram(histogram, 0.01)
    p99 = _quantile_from_histogram(histogram, 0.99)
    levels = np.arange(256, dtype=np.float64)
    clipped_levels = np.clip(levels, p01, p99)
    count = int(histogram.sum())
    mean = float((clipped_levels * histogram).sum() / max(count, 1))
    variance = float(
        (((clipped_levels - mean) ** 2) * histogram).sum() / max(count, 1)
    )
    output = {
        "protocol": "HeterogeneousDamageFormer_train_only_SAR_stats_v1",
        "dataset_path": str(dataset),
        "train_data_list_path": str(list_path),
        "train_list_sha256": hashlib.sha256(list_path.read_bytes()).hexdigest(),
        "train_image_count": len(names),
        "sar_representation": "single_channel_display_uint8",
        "finite_pixel_count": count,
        "nonfinite_pixel_count": nonfinite,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "histogram": histogram.tolist(),
        "normalization": {
            "clip": "p01_p99_exact_histogram",
            "p01": p01,
            "p99": p99,
            "mean": mean,
            "std": max(variance**0.5, 1e-6),
        },
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved frozen SAR stats: {destination}", flush=True)


if __name__ == "__main__":
    main()
