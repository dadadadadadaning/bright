#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a compact, reproducible comparison of two official S8 evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("official_style_4class_miou", "Final mIoU"),
    ("target_3class_miou", "Valid 3-class mIoU"),
    ("background_iou", "Background IoU"),
    ("intact_iou", "Intact IoU"),
    ("damaged_iou", "Damaged IoU"),
    ("destroyed_iou", "Destroyed IoU"),
    ("loc_precision", "Localization Precision"),
    ("loc_recall", "Localization Recall"),
    ("loc_f1", "Localization F1"),
    ("loc_iou", "Localization IoU"),
    ("damaged_f1", "Damaged F1"),
    ("destroyed_f1", "Destroyed F1"),
    ("damaged_to_intact", "Damaged -> Intact"),
    ("destroyed_to_intact", "Destroyed -> Intact"),
    ("event_mean_official_style_4class_miou", "Event-mean mIoU"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize paired S8 official evaluations")
    parser.add_argument("--p3_json", required=True)
    parser.add_argument("--ia_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    return parser.parse_args()


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return obj


def finite_metric(payload: dict[str, Any], key: str, source: str) -> float:
    value = payload.get(key)
    if value is None:
        raise KeyError(f"Metric {key!r} is missing from {source}")
    return float(value)


def main() -> None:
    args = parse_args()
    p3 = read_json(args.p3_json)
    ia = read_json(args.ia_json)

    rows: list[dict[str, Any]] = []
    for key, label in METRICS:
        p3_value = finite_metric(p3, key, args.p3_json)
        ia_value = finite_metric(ia, key, args.ia_json)
        higher_is_better = key not in {"damaged_to_intact", "destroyed_to_intact"}
        raw_delta = ia_value - p3_value
        preferred = (
            "S8_IA_MLA"
            if (raw_delta > 0) == higher_is_better and raw_delta != 0
            else "S8_MICB_DSMIL_P3"
            if raw_delta != 0
            else "tie"
        )
        rows.append(
            {
                "metric": key,
                "label": label,
                "higher_is_better": higher_is_better,
                "S8_MICB_DSMIL_P3_percent": 100.0 * p3_value,
                "S8_IA_MLA_percent": 100.0 * ia_value,
                "IA_minus_P3_pp": 100.0 * raw_delta,
                "preferred": preferred,
            }
        )

    output = {
        "protocol": "official",
        "unit": "percent; deltas are percentage points",
        "p3_source": str(Path(args.p3_json)),
        "ia_source": str(Path(args.ia_json)),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_tsv = Path(args.output_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'Metric':32s} {'P3 (%)':>10s} {'IA-MLA (%)':>12s} "
        f"{'IA-P3 (pp)':>12s} {'Preferred':>14s}"
    )
    for row in rows:
        print(
            f"{row['label']:32s} "
            f"{row['S8_MICB_DSMIL_P3_percent']:10.4f} "
            f"{row['S8_IA_MLA_percent']:12.4f} "
            f"{row['IA_minus_P3_pp']:12.4f} "
            f"{row['preferred']:>14s}"
        )
    print(f"Saved comparison JSON: {output_json}")
    print(f"Saved comparison TSV:  {output_tsv}")


if __name__ == "__main__":
    main()
