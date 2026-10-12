#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize official DamageFormer reproduce results without pandas."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(os.environ.get("ROOT", "/root/shared-nvme/BRIGHT-master"))
EVAL_ROOT = ROOT / "eval_results_official" / "DamageFormer_Reproduce"
OUT_CSV = EVAL_ROOT / "DamageFormer_Reproduce_summary.csv"

FIELDS = [
    "model",
    "checkpoint",
    "target3",
    "DmgIoU",
    "DesIoU",
    "Dmg_to_Int",
    "Des_to_Int",
    "Dmg_to_Des",
    "Des_to_Dmg",
    "Dmg_to_Bg",
    "Des_to_Bg",
]


def to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def fmt(value: Any) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.6f}"


def collect_one(model: str, checkpoint: str, ckpt_dir: Path) -> Dict[str, Any]:
    summary = read_json(ckpt_dir / "summary_official.json")
    return {
        "model": model,
        "checkpoint": checkpoint,
        "target3": to_float(summary.get("target_3class_miou")),
        "DmgIoU": to_float(summary.get("damaged_iou")),
        "DesIoU": to_float(summary.get("destroyed_iou")),
        "Dmg_to_Int": to_float(summary.get("damaged_to_intact")),
        "Des_to_Int": to_float(summary.get("destroyed_to_intact")),
        "Dmg_to_Des": to_float(summary.get("damaged_to_destroyed")),
        "Des_to_Dmg": to_float(summary.get("destroyed_to_damaged")),
        "Dmg_to_Bg": to_float(summary.get("damaged_to_background")),
        "Des_to_Bg": to_float(summary.get("destroyed_to_background")),
    }


def rank_key(row: Dict[str, Any]) -> float:
    val = to_float(row.get("target3"))
    return val if val is not None else -1.0


def print_row(prefix: str, row: Dict[str, Any]) -> None:
    print(
        f"{prefix}{row['model']} / {row['checkpoint']} | target3={fmt(row.get('target3'))} | "
        f"DmgIoU={fmt(row.get('DmgIoU'))} | DesIoU={fmt(row.get('DesIoU'))} | "
        f"Dmg->Int={fmt(row.get('Dmg_to_Int'))} | Des->Int={fmt(row.get('Des_to_Int'))} | "
        f"Dmg->Des={fmt(row.get('Dmg_to_Des'))} | Des->Dmg={fmt(row.get('Des_to_Dmg'))} | "
        f"Dmg->Bg={fmt(row.get('Dmg_to_Bg'))} | Des->Bg={fmt(row.get('Des_to_Bg'))}"
    )


def main() -> None:
    if not EVAL_ROOT.exists():
        raise SystemExit(f"Evaluation root does not exist: {EVAL_ROOT}")
    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(EVAL_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for ckpt_dir in sorted(model_dir.iterdir()):
            if ckpt_dir.is_dir():
                row = collect_one(model_dir.name, ckpt_dir.name, ckpt_dir)
                if any(row.get(k) is not None for k in FIELDS[2:]):
                    rows.append(row)
    rows.sort(key=rank_key, reverse=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if k in {"model", "checkpoint"} else fmt(row.get(k)) for k in FIELDS})
    print(f"Saved summary: {OUT_CSV}")
    print("\nDamageFormer reproduce checkpoints:")
    for row in rows:
        print_row("  ", row)


if __name__ == "__main__":
    main()
