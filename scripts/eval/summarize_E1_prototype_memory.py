#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize E1 prototype memory grid results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


FIELDS = [
    "config",
    "alpha",
    "topk",
    "apply_region",
    "uncertainty_threshold",
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


def collect(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/summary_official.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = {key: data.get(key, "") for key in FIELDS}
        row["config"] = path.parent.name
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
            writer.writerow({key: fmt(row.get(key)) if key not in {"config", "apply_region"} else row.get(key, "") for key in FIELDS})


def write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    best = rows[0] if rows else {}
    lines = ["# E1 Prototype Memory Report", "", f"Total configs: {len(rows)}", ""]
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
        if (to_float(best.get("target3")) or 0.0) > 0.525102:
            lines.append("Prototype memory improved over the E1 target3 reference; inspect class trade-offs before adopting.")
        elif (to_float(best.get("damaged_iou")) or 0.0) > 0.388059:
            lines.append("Prototype memory improved damaged IoU but did not clearly improve target3.")
        else:
            lines.append("Prototype memory did not clearly beat the E1 reference in this grid.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = sorted(collect(root), key=sort_key, reverse=True)
    write_csv(root / "prototype_all_configs.csv", rows)
    write_csv(root / "prototype_best_config.csv", rows[:1])
    write_report(root / "prototype_report.md", rows)
    print(f"Summarized {len(rows)} prototype configs under {root}")


if __name__ == "__main__":
    main()
