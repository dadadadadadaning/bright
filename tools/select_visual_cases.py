#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select and visualize BRIGHT success/failure cases from prediction masks.

This script compares three prediction directories against GT labels and saves
panel images:
  pre RGB | post SAR | GT | baseline pred | A11 pred | improved pred | error map

Prediction files are searched as:
  <pred_dir>/<name>.png|.tif|.tiff|.npy

Dataset files follow BRIGHT naming:
  pre-event/<name>_pre_disaster.tif
  post-event/<name>_post_disaster.tif
  target/<name>_building_damage.tif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


PALETTE = {
    0: (0, 0, 0),
    1: (0, 180, 0),
    2: (255, 180, 0),
    3: (220, 0, 0),
    255: (120, 120, 120),
}


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_image(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    return arr


def read_mask_from_dir(pred_dir: str, name: str) -> np.ndarray:
    for suffix in (".png", ".tif", ".tiff", ".npy"):
        path = os.path.join(pred_dir, name + suffix)
        if os.path.exists(path):
            if suffix == ".npy":
                return np.load(path).astype(np.uint8)
            return np.array(Image.open(path)).astype(np.uint8)
    raise FileNotFoundError(f"No prediction mask found for {name} in {pred_dir}")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, [2, 98])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def color_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for cls, color in PALETTE.items():
        out[mask == cls] = color
    return out


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt.shape[:2], 3), dtype=np.uint8)
    valid = gt != 255
    out[(pred == gt) & valid] = (40, 40, 40)
    out[(gt == 2) & (pred == 1)] = (255, 0, 255)      # damaged -> intact
    out[(gt == 3) & (pred == 1)] = (0, 255, 255)      # destroyed -> intact
    out[(gt == 3) & (pred == 2)] = (255, 255, 0)      # destroyed -> damaged
    out[(gt == 1) & ((pred == 2) | (pred == 3))] = (0, 0, 255)  # intact over-damage
    out[(pred != gt) & valid & (out.sum(axis=-1) == 0)] = (255, 255, 255)
    return out


def resize_panel(arr: np.ndarray, size: int = 256) -> Image.Image:
    return Image.fromarray(normalize_to_uint8(arr)).resize((size, size), Image.Resampling.NEAREST)


def make_panel(items: List[Tuple[str, np.ndarray]], out_path: str, size: int = 256) -> None:
    title_h = 24
    canvas = Image.new("RGB", (size * len(items), size + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (title, arr) in enumerate(items):
        panel = resize_panel(arr, size)
        x = i * size
        canvas.paste(panel, (x, title_h))
        draw.text((x + 4, 4), title, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def score_cases(gt: np.ndarray, base: np.ndarray, a11: np.ndarray, imp: np.ndarray) -> Dict[str, float]:
    destroyed = gt == 3
    damaged = gt == 2
    intact = gt == 1
    return {
        "destroyed_fix": ratio(destroyed & (base == 1) & ((a11 == 3) | (imp == 3))),
        "damaged_fix": ratio(damaged & (base == 1) & ((a11 == 2) | (imp == 2))),
        "intact_fail": ratio(intact & (((a11 == 2) | (a11 == 3)) | ((imp == 2) | (imp == 3)))),
        "typical_success": ratio(((gt == 2) | (gt == 3)) & (base == 1) & (imp == gt)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--baseline_pred_dir", required=True)
    parser.add_argument("--a11_pred_dir", required=True)
    parser.add_argument("--improved_pred_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--panel_size", type=int, default=256)
    args = parser.parse_args()

    names = read_list(args.data_list_path)
    buckets: Dict[str, List[Tuple[float, str]]] = {
        "destroyed_to_intact_fixed": [],
        "damaged_to_intact_fixed": [],
        "intact_overdamage_failure": [],
        "typical_success": [],
    }

    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in names:
        gt = read_image(os.path.join(args.dataset_path, "target", name + "_building_damage.tif")).astype(np.uint8)
        base = read_mask_from_dir(args.baseline_pred_dir, name)
        a11 = read_mask_from_dir(args.a11_pred_dir, name)
        imp = read_mask_from_dir(args.improved_pred_dir, name)
        scores = score_cases(gt, base, a11, imp)
        buckets["destroyed_to_intact_fixed"].append((scores["destroyed_fix"], name))
        buckets["damaged_to_intact_fixed"].append((scores["damaged_fix"], name))
        buckets["intact_overdamage_failure"].append((scores["intact_fail"], name))
        buckets["typical_success"].append((scores["typical_success"], name))
        cache[name] = (gt, base, a11, imp)

    selected = set()
    for bucket, rows in buckets.items():
        rows.sort(reverse=True)
        for score, name in rows[: args.topk]:
            if score <= 0:
                continue
            gt, base, a11, imp = cache[name]
            pre = read_image(os.path.join(args.dataset_path, "pre-event", name + "_pre_disaster.tif"))[..., :3]
            post = read_image(os.path.join(args.dataset_path, "post-event", name + "_post_disaster.tif"))
            err = error_map(gt, imp)
            out_path = os.path.join(args.out_dir, bucket, f"{name}_score{score:.4f}.png")
            make_panel(
                [
                    ("pre RGB", pre),
                    ("post SAR", post),
                    ("GT", color_mask(gt)),
                    ("baseline", color_mask(base)),
                    ("A11", color_mask(a11)),
                    ("improved", color_mask(imp)),
                    ("error", err),
                ],
                out_path,
                size=args.panel_size,
            )
            selected.add((bucket, name, score))

    index_path = os.path.join(args.out_dir, "selected_cases.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("case_type,name,score\n")
        for bucket, name, score in sorted(selected):
            f.write(f"{bucket},{name},{score:.6f}\n")
    print(f"Saved visual cases to {args.out_dir}")


if __name__ == "__main__":
    main()
