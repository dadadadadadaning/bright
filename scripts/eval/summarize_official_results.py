#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize all official-style BRIGHT evaluation results without pandas."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-master/eval_results_official"
DEFAULT_OUTPUT = "/root/shared-nvme/BRIGHT-master/eval_results_official/all_model_official_summary.csv"

FIELDS = [
    "model",
    "checkpoint",
    "official_style_4class_miou",
    "target_3class_miou",
    "background_iou",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_recall",
    "destroyed_recall",
    "damaged_to_intact",
    "destroyed_to_intact",
    "loc_iou",
    "loc_f1",
    "loc_head_iou",
    "loc_head_f1",
    "damage_building_iou",
    "damage_building_f1",
    "save_dir",
]


def as_sort_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def iter_summary_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        if "summary_official.json" in filenames:
            yield os.path.join(dirpath, "summary_official.json")


def infer_model_checkpoint(root: str, summary_path: str) -> Tuple[str, str]:
    save_dir = os.path.dirname(summary_path)
    rel = os.path.relpath(save_dir, root)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[0], parts[-1]
    if len(parts) == 1:
        return parts[0], ""
    return "", ""


def read_summary(root: str, summary_path: str) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model, checkpoint = infer_model_checkpoint(root, summary_path)
    save_dir = os.path.dirname(summary_path)
    return {
        "model": model,
        "checkpoint": checkpoint,
        "official_style_4class_miou": data.get("official_style_4class_miou", ""),
        "target_3class_miou": data.get("target_3class_miou", ""),
        "background_iou": data.get("background_iou", ""),
        "intact_iou": data.get("intact_iou", ""),
        "damaged_iou": data.get("damaged_iou", ""),
        "destroyed_iou": data.get("destroyed_iou", ""),
        "damaged_recall": data.get("damaged_recall", ""),
        "destroyed_recall": data.get("destroyed_recall", ""),
        "damaged_to_intact": data.get("damaged_to_intact", ""),
        "destroyed_to_intact": data.get("destroyed_to_intact", ""),
        "loc_iou": data.get("loc_iou", ""),
        "loc_f1": data.get("loc_f1", ""),
        "loc_head_iou": data.get("loc_head_iou", data.get("loc_iou", "")),
        "loc_head_f1": data.get("loc_head_f1", data.get("loc_f1", "")),
        "damage_building_iou": data.get("damage_building_iou", ""),
        "damage_building_f1": data.get("damage_building_f1", ""),
        "save_dir": save_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official-style BRIGHT evaluation results.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    if os.path.isdir(args.root):
        rows = [read_summary(args.root, path) for path in iter_summary_files(args.root)]

    rows.sort(key=lambda row: as_sort_float(row.get("target_3class_miou")), reverse=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {args.output}")
    print(f"Found {len(rows)} official summaries under: {args.root}")
    print(f"Top {min(args.top_k, len(rows))} by target_3class_miou:")
    for rank, row in enumerate(rows[: args.top_k], start=1):
        print(
            f"{rank:02d}. {row['model']} / {row['checkpoint']} | "
            f"target_3class_miou={row['target_3class_miou']} | "
            f"official_style_4class_miou={row['official_style_4class_miou']} | "
            f"damaged_iou={row['damaged_iou']} | destroyed_iou={row['destroyed_iou']}"
        )


if __name__ == "__main__":
    main()
