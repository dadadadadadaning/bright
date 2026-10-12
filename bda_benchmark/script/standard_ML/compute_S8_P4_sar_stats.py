#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute frozen train-only SAR statistics for S8-P4 A1/A2/A3/A4."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def list_sha256(items: List[str]) -> str:
    canonical = "".join(item + "\n" for item in items).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_scalar_sar(path: str) -> np.ndarray:
    image = np.asarray(imageio.imread(path), dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        channels = image[..., :3]
        if np.allclose(channels[..., 0], channels[..., 1]) and np.allclose(
            channels[..., 0], channels[..., 2]
        ):
            return channels[..., 0]
    raise ValueError(f"SAR file is not scalar or repeated-scalar: {path} {image.shape}")


class UniformReservoir:
    """Chunk-wise exact reservoir update without a Python loop per pixel."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.values = np.empty((self.capacity,), dtype=np.float32)
        self.size = 0
        self.seen = 0
        self.large_population_fallback_draws = 0

    def _draw_new_count(self, count: int, old_seen: int) -> int:
        """Draw how many reservoir slots come from the incoming chunk.

        NumPy's exact hypergeometric sampler rejects either population count
        at one billion. BRIGHT contains roughly three billion train pixels, so
        above that implementation limit we use the standard binomial large-
        population approximation with the same expected inclusion rate.
        """

        if count < 1_000_000_000 and old_seen < 1_000_000_000:
            value = self.rng.hypergeometric(
                ngood=count,
                nbad=old_seen,
                nsample=self.capacity,
            )
        else:
            probability = float(count) / float(old_seen + count)
            value = self.rng.binomial(self.capacity, probability)
            self.large_population_fallback_draws += 1
        return min(int(value), count, self.capacity)

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        count = int(flat.size)
        if count == 0:
            return
        old_seen = self.seen
        self.seen += count
        if self.size < self.capacity:
            take = min(self.capacity - self.size, count)
            self.values[self.size : self.size + take] = flat[:take]
            self.size += take
            flat = flat[take:]
            count -= take
            old_seen += take
            if count == 0:
                return

        # Given a uniform reservoir of old_seen elements, the number of items
        # drawn from the new chunk in a uniform capacity-sized sample follows
        # a hypergeometric distribution.
        new_count = self._draw_new_count(count, old_seen)
        if new_count <= 0:
            return
        incoming = self.rng.choice(count, size=new_count, replace=False)
        outgoing = self.rng.choice(self.capacity, size=new_count, replace=False)
        self.values[outgoing] = flat[incoming]

    def sample(self) -> np.ndarray:
        return self.values[: self.size].copy()


def main() -> None:
    parser = argparse.ArgumentParser("Compute S8-P4 train-only SAR statistics")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--suffix", default=".tif")
    parser.add_argument("--reservoir_pixels", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--sar_representation",
        choices=("amplitude", "power", "db", "display", "unknown"),
        default="unknown",
        help="Metadata only: P4 never applies an implicit log transform.",
    )
    args = parser.parse_args()

    names = read_list(args.train_data_list_path)
    if not names:
        raise RuntimeError("training list is empty")
    paths = [
        os.path.join(
            args.dataset_path,
            "post-event",
            name + "_post_disaster" + args.suffix,
        )
        for name in names
    ]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} SAR files; first={missing[0]}")

    reservoir = UniformReservoir(args.reservoir_pixels, args.seed)
    raw_min = float("inf")
    raw_max = float("-inf")
    finite_pixels = 0
    nonfinite_pixels = 0
    for index, path in enumerate(paths, start=1):
        sar = load_scalar_sar(path)
        finite = np.isfinite(sar)
        nonfinite_pixels += int((~finite).sum())
        values = sar[finite]
        if values.size:
            raw_min = min(raw_min, float(values.min()))
            raw_max = max(raw_max, float(values.max()))
            finite_pixels += int(values.size)
            reservoir.update(values)
        if index % 100 == 0 or index == len(paths):
            print(f"quantile pass: {index}/{len(paths)}", flush=True)
    if finite_pixels == 0:
        raise RuntimeError("no finite SAR pixels found")

    sample = reservoir.sample()
    p01, p99 = np.quantile(sample.astype(np.float64), [0.01, 0.99]).tolist()
    if not p99 > p01:
        raise RuntimeError(f"invalid SAR quantiles: p01={p01}, p99={p99}")

    clipped_sum = 0.0
    clipped_square_sum = 0.0
    clipped_count = 0
    clipped_low = 0
    clipped_high = 0
    for index, path in enumerate(paths, start=1):
        sar = load_scalar_sar(path)
        values = sar[np.isfinite(sar)].astype(np.float64)
        clipped_low += int((values < p01).sum())
        clipped_high += int((values > p99).sum())
        values = np.clip(values, p01, p99)
        clipped_sum += float(values.sum(dtype=np.float64))
        clipped_square_sum += float(np.square(values).sum(dtype=np.float64))
        clipped_count += int(values.size)
        if index % 100 == 0 or index == len(paths):
            print(f"moment pass: {index}/{len(paths)}", flush=True)

    mean = clipped_sum / clipped_count
    variance = max(clipped_square_sum / clipped_count - mean * mean, 0.0)
    std = variance**0.5
    if not std > 1e-12:
        raise RuntimeError(f"clipped SAR standard deviation is invalid: {std}")

    payload = {
        "protocol": "S8-P4_train_only_SAR_stats_v1",
        "dataset_path": os.path.abspath(args.dataset_path),
        "train_data_list_path": os.path.abspath(args.train_data_list_path),
        "train_list_sha256": list_sha256(names),
        "train_image_count": len(names),
        "sar_representation": args.sar_representation,
        "implicit_log_transform": False,
        "finite_pixel_count": finite_pixels,
        "nonfinite_pixel_count": nonfinite_pixels,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "quantile_estimator": {
            "name": "chunkwise_uniform_reservoir",
            "capacity": args.reservoir_pixels,
            "actual_sample_count": int(sample.size),
            "seed": args.seed,
            "large_population_fallback": (
                "binomial approximation when a population count reaches "
                "NumPy's 1e9 hypergeometric limit"
            ),
            "large_population_fallback_draws": (
                reservoir.large_population_fallback_draws
            ),
        },
        "normalization": {
            "clip": "p01_p99",
            "p01": p01,
            "p99": p99,
            "mean": mean,
            "std": std,
            "clipped_low_count": clipped_low,
            "clipped_high_count": clipped_high,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved frozen SAR stats: {output}", flush=True)


if __name__ == "__main__":
    main()
