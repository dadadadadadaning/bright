#!/usr/bin/env python3
"""Audit label and SAR-domain differences among BRIGHT split lists.

This is deliberately model-free.  It separates three possible causes of a
validation/test gap: class-prior shift, SAR radiometric shift, and event/AOI
composition.  Test labels are read only for retrospective diagnosis; never
use this report to select a checkpoint or tune a model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


EVENT_RE = re.compile(r"_\d+$")
CLASSES = (0, 1, 2, 3, 255)


def event_name(sample: str) -> str:
    return EVENT_RE.sub("", sample.strip())


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left /= max(left.sum(), 1.0)
    right /= max(right.sum(), 1.0)
    middle = 0.5 * (left + right)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        keep = p > 0
        return float(np.sum(p[keep] * np.log2(p[keep] / q[keep])))

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def summarize(label_hist: np.ndarray, sar_hist: np.ndarray, images: int) -> dict:
    valid = float(label_hist[:4].sum())
    building = float(label_hist[1:4].sum())
    affected = float(label_hist[2:4].sum())
    sar_total = float(sar_hist.sum())
    values = np.arange(256, dtype=np.float64)
    sar_mean = float((values * sar_hist).sum() / max(sar_total, 1.0))
    sar_var = float((((values - sar_mean) ** 2) * sar_hist).sum() / max(sar_total, 1.0))

    def percentile(q: float) -> int:
        if sar_total == 0:
            return 0
        return int(np.searchsorted(np.cumsum(sar_hist), q * sar_total))

    return {
        "images": images,
        "label_pixels": {str(c): int(label_hist[i]) for i, c in enumerate(CLASSES)},
        "class_fraction_valid": {
            str(c): float(label_hist[i] / max(valid, 1.0)) for i, c in enumerate(CLASSES[:4])
        },
        "building_fraction": building / max(valid, 1.0),
        "affected_fraction_of_buildings": affected / max(building, 1.0),
        "damaged_fraction_of_buildings": float(label_hist[2]) / max(building, 1.0),
        "destroyed_fraction_of_buildings": float(label_hist[3]) / max(building, 1.0),
        "sar": {
            "sampled_pixels": int(sar_total),
            "mean": sar_mean,
            "std": math.sqrt(max(sar_var, 0.0)),
            "p01": percentile(0.01),
            "p50": percentile(0.50),
            "p99": percentile(0.99),
            "histogram_0_255": sar_hist.astype(int).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-dir", type=Path,
        default=Path("bda_benchmark/dataset/splitname/standard_ML"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--sar-stride", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    target_dir = args.dataset_root / "target"
    sar_dir = args.dataset_root / "post-event"
    aggregate: dict[str, dict[str, object]] = {}

    for split in args.splits:
        ids = read_ids(args.split_dir / f"{split}_set.txt")
        total_labels = np.zeros(len(CLASSES), dtype=np.int64)
        total_sar = np.zeros(256, dtype=np.int64)
        per_event = defaultdict(
            lambda: {
                "labels": np.zeros(len(CLASSES), dtype=np.int64),
                "sar": np.zeros(256, dtype=np.int64),
                "images": 0,
            }
        )

        for index, sample in enumerate(ids, 1):
            label_path = target_dir / f"{sample}_building_damage.tif"
            sar_path = sar_dir / f"{sample}_post_disaster.tif"
            if not label_path.is_file() or not sar_path.is_file():
                raise FileNotFoundError(f"missing pair for {sample}: {label_path}, {sar_path}")

            label = np.asarray(Image.open(label_path))
            sar = np.asarray(Image.open(sar_path))
            if sar.ndim == 3:
                sar = sar[..., 0]
            if sar.dtype != np.uint8:
                sar = np.clip(sar, 0, 255).astype(np.uint8)
            sampled_sar = sar[:: args.sar_stride, :: args.sar_stride]
            label_counts = np.array([(label == c).sum() for c in CLASSES], dtype=np.int64)
            sar_counts = np.bincount(sampled_sar.reshape(-1), minlength=256).astype(np.int64)

            total_labels += label_counts
            total_sar += sar_counts
            record = per_event[event_name(sample)]
            record["labels"] += label_counts
            record["sar"] += sar_counts
            record["images"] += 1
            if index % 250 == 0:
                print(f"{split}: {index}/{len(ids)}", flush=True)

        aggregate[split] = summarize(total_labels, total_sar, len(ids))
        aggregate[split]["events"] = {
            event: summarize(data["labels"], data["sar"], int(data["images"]))
            for event, data in sorted(per_event.items())
        }

    pairwise = {}
    for i, left_name in enumerate(args.splits):
        for right_name in args.splits[i + 1 :]:
            left = aggregate[left_name]
            right = aggregate[right_name]
            left_label = np.array([left["label_pixels"][str(c)] for c in CLASSES[:4]])
            right_label = np.array([right["label_pixels"][str(c)] for c in CLASSES[:4]])
            left_sar = np.array(left["sar"]["histogram_0_255"])
            right_sar = np.array(right["sar"]["histogram_0_255"])
            pairwise[f"{left_name}_vs_{right_name}"] = {
                "label_js_bits": js_divergence(left_label, right_label),
                "sar_js_bits": js_divergence(left_sar, right_sar),
            }

    report = {
        "protocol": "BRIGHT_split_domain_audit_v1",
        "warning": "diagnostic only; test must never select checkpoints or hyperparameters",
        "sar_sampling_stride": args.sar_stride,
        "splits": aggregate,
        "pairwise": pairwise,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'Split':<8} {'Images':>7} {'Build%':>9} {'Dmg/B%':>9} {'Des/B%':>9} "
          f"{'SARmean':>9} {'SARstd':>9}")
    print("-" * 72)
    for name in args.splits:
        item = aggregate[name]
        print(f"{name:<8} {item['images']:7d} {100*item['building_fraction']:9.3f} "
              f"{100*item['damaged_fraction_of_buildings']:9.3f} "
              f"{100*item['destroyed_fraction_of_buildings']:9.3f} "
              f"{item['sar']['mean']:9.3f} {item['sar']['std']:9.3f}")
    print("Pairwise JS divergence (bits):")
    for name, item in pairwise.items():
        print(f"  {name:<20} labels={item['label_js_bits']:.6f} sar={item['sar_js_bits']:.6f}")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
