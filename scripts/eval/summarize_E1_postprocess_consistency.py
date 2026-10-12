#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize E1 instance/superpixel consistency postprocess grid results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


FIELDS = [
    "config",
    "mode",
    "dominant_ratio",
    "min_area",
    "damage_keep_ratio",
    "n_segments",
    "compactness",
    "mIoU4",
    "target3",
    "background_iou",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_to_intact",
    "destroyed_to_intact",
    "damaged_to_destroyed",
    "destroyed_to_damaged",
    "damaged_to_background",
    "destroyed_to_background",
    "before_target3",
    "before_damaged_iou",
    "before_destroyed_iou",
    "load_missing_keys_count",
    "load_unexpected_keys_count",
]


def to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def fmt(value: Any) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.6f}"


def read_summary(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary_official.json")):
        data = read_summary(summary_path)
        if not data:
            continue
        row = {key: data.get(key, "") for key in FIELDS}
        row["config"] = summary_path.parent.name
        rows.append(row)
    return rows


def sort_key(row: Dict[str, Any]) -> tuple:
    return (
        to_float(row.get("target3")) or -1.0,
        to_float(row.get("damaged_iou")) or -1.0,
        to_float(row.get("destroyed_iou")) or -1.0,
        -(to_float(row.get("damaged_to_intact")) or 99.0),
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) if key not in {"config", "mode"} else row.get(key, "") for key in FIELDS})


def write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    best = rows[0] if rows else {}
    lines = [
        "# E1 Postprocess Consistency Report",
        "",
        f"Total configs: {len(rows)}",
        "",
    ]
    if best:
        lines += [
            "## Best Config",
            "",
            f"- config: {best.get('config')}",
            f"- target3: {fmt(best.get('target3'))}",
            f"- damaged_iou: {fmt(best.get('damaged_iou'))}",
            f"- destroyed_iou: {fmt(best.get('destroyed_iou'))}",
            f"- damaged_to_intact: {fmt(best.get('damaged_to_intact'))}",
            f"- destroyed_to_intact: {fmt(best.get('destroyed_to_intact'))}",
            "",
            "## Auto Conclusion",
            "",
        ]
        before = to_float(best.get("before_target3"))
        after = to_float(best.get("target3"))
        d2i = to_float(best.get("damaged_to_intact"))
        if before is not None and after is not None and after > before and (d2i is None or d2i <= 0.269005):
            lines.append("Postprocess improved target3 without increasing damaged-to-intact beyond the E1 reference.")
        elif before is not None and after is not None and after > before:
            lines.append("Postprocess improved target3, but damaged-to-intact should be inspected for over-smoothing.")
        else:
            lines.append("Postprocess did not clearly improve target3; keep it as an analysis tool unless a safer config appears.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = sorted(collect(root), key=sort_key, reverse=True)
    write_csv(root / "E1_postprocess_all_configs.csv", rows)
    write_csv(root / "E1_postprocess_best_config.csv", rows[:1])
    write_report(root / "E1_postprocess_report.md", rows)
    print(f"Summarized {len(rows)} postprocess configs under {root}")


if __name__ == "__main__":
    main()
