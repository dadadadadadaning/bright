#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize official A11 SH/Under experiments without pandas."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(os.environ.get("ROOT", "/root/shared-nvme/BRIGHT-master"))
EVAL_ROOT = ROOT / "eval_results_official" / "A11_SH_Under"
OUT_CSV = EVAL_ROOT / "A11_SH_Under_summary.csv"

BASELINES = {
    "A11_Cons005_w20000": {
        "target3": 0.510170,
        "damaged_iou": 0.358117,
        "destroyed_iou": 0.456557,
        "damaged_to_intact": 0.251663,
        "destroyed_to_intact": 0.215569,
        "damaged_to_destroyed": 0.039612,
        "destroyed_to_damaged": 0.093389,
    },
    "A11_Cons005_w20000_SH": {
        "target3": 0.534248,
        "damaged_iou": 0.358225,
        "destroyed_iou": 0.522405,
        "damaged_to_intact": 0.372050,
        "destroyed_to_intact": 0.277166,
        "damaged_to_destroyed": 0.028136,
        "destroyed_to_damaged": 0.037503,
    },
}

FIELDS = [
    "model",
    "checkpoint",
    "official4",
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
    "damage_score",
    "balanced_score",
]
METRIC_KEYS = FIELDS[2:]


def to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def nanmean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_confusion_csv(path: Path) -> Optional[List[List[float]]]:
    if not path.exists():
        return None
    rows: List[List[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.reader(f):
            vals = [to_float(cell) for cell in raw[1:]]
            nums = [v for v in vals if v is not None]
            if len(nums) >= 4:
                rows.append(nums[:4])
    return rows[:4] if len(rows) >= 4 else None


def metrics_from_confusion(conf: List[List[float]]) -> Dict[str, float]:
    ious: List[float] = []
    for cls in range(4):
        tp = conf[cls][cls]
        gt = sum(conf[cls])
        pred = sum(conf[row][cls] for row in range(4))
        ious.append(safe_div(tp, gt + pred - tp))
    gt_damaged = sum(conf[2])
    gt_destroyed = sum(conf[3])
    return {
        "background_iou": ious[0],
        "intact_iou": ious[1],
        "damaged_iou": ious[2],
        "destroyed_iou": ious[3],
        "official4": nanmean(ious),
        "target3": nanmean(ious[1:4]),
        "damaged_to_background": safe_div(conf[2][0], gt_damaged),
        "damaged_to_intact": safe_div(conf[2][1], gt_damaged),
        "damaged_to_destroyed": safe_div(conf[2][3], gt_damaged),
        "destroyed_to_background": safe_div(conf[3][0], gt_destroyed),
        "destroyed_to_intact": safe_div(conf[3][1], gt_destroyed),
        "destroyed_to_damaged": safe_div(conf[3][2], gt_destroyed),
    }


def add_scores(row: Dict[str, Any]) -> None:
    target3 = to_float(row.get("target3"))
    intact = to_float(row.get("intact_iou"))
    damaged = to_float(row.get("damaged_iou"))
    destroyed = to_float(row.get("destroyed_iou"))
    d2i = to_float(row.get("damaged_to_intact"))
    x2i = to_float(row.get("destroyed_to_intact"))
    if None not in (target3, damaged, destroyed, d2i, x2i):
        row["damage_score"] = target3 + 0.3 * damaged + 0.3 * destroyed - 0.2 * d2i - 0.2 * x2i
    if None not in (target3, intact, damaged, destroyed):
        row["balanced_score"] = 0.5 * target3 + 0.5 * min(intact, damaged, destroyed)


def collect_one(model: str, checkpoint: str, ckpt_dir: Path) -> Dict[str, Any]:
    summary = read_json(ckpt_dir / "summary_official.json")
    row: Dict[str, Any] = {
        "model": model,
        "checkpoint": checkpoint,
        "official4": to_float(summary.get("official_style_4class_miou")),
        "target3": to_float(summary.get("target_3class_miou")),
    }
    for key in FIELDS[4:14]:
        row[key] = to_float(summary.get(key))
    conf = read_confusion_csv(ckpt_dir / "confusion_4class_official.csv")
    if conf is not None:
        fallback = metrics_from_confusion(conf)
        for key, value in fallback.items():
            if row.get(key) is None:
                row[key] = value
    add_scores(row)
    return row


def fmt(value: Any) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.6f}"


def rank_key(row: Dict[str, Any], key: str) -> float:
    val = to_float(row.get(key))
    return val if val is not None else -1.0


def print_row(prefix: str, row: Dict[str, Any]) -> None:
    print(
        f"{prefix}{row['model']} / {row['checkpoint']} | "
        f"target3={fmt(row.get('target3'))} | DmgIoU={fmt(row.get('damaged_iou'))} | "
        f"DesIoU={fmt(row.get('destroyed_iou'))} | Dmg->Int={fmt(row.get('damaged_to_intact'))} | "
        f"Des->Int={fmt(row.get('destroyed_to_intact'))} | Dmg->Des={fmt(row.get('damaged_to_destroyed'))} | "
        f"Des->Dmg={fmt(row.get('destroyed_to_damaged'))} | balanced={fmt(row.get('balanced_score'))}"
    )


def best_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: rank_key(r, key), reverse=True):
        best.setdefault(str(row["model"]), row)
    return best


def delta_text(row: Dict[str, Any], baseline_name: str) -> str:
    base = BASELINES[baseline_name]
    parts = []
    for label, key in (
        ("target3", "target3"),
        ("DmgIoU", "damaged_iou"),
        ("DesIoU", "destroyed_iou"),
        ("Dmg->Int", "damaged_to_intact"),
        ("Des->Int", "destroyed_to_intact"),
        ("Dmg->Des", "damaged_to_destroyed"),
        ("Des->Dmg", "destroyed_to_damaged"),
    ):
        val = to_float(row.get(key))
        parts.append(f"d_{label}={'' if val is None else f'{val - base[key]:+.6f}'}")
    return " | ".join(parts)


def labels(row: Dict[str, Any]) -> List[str]:
    checks = [
        (rank_key(row, "target3") >= 0.525, "target3 reached"),
        (rank_key(row, "destroyed_iou") >= 0.500, "Destroyed IoU reached"),
        (rank_key(row, "damaged_iou") >= 0.350, "Damaged IoU did not clearly drop"),
        (rank_key(row, "damaged_to_intact") <= 0.30, "damaged underestimation suppressed"),
        (rank_key(row, "destroyed_to_intact") <= 0.25, "destroyed underestimation suppressed"),
        (
            rank_key(row, "damaged_to_destroyed") <= 0.04
            and rank_key(row, "destroyed_to_damaged") <= 0.05,
            "damaged/destroyed confusion is low",
        ),
    ]
    return [text for ok, text in checks if ok] or ["needs inspection"]


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
                if any(row.get(key) is not None for key in METRIC_KEYS):
                    rows.append(row)
    rows.sort(key=lambda r: rank_key(r, "target3"), reverse=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if k in {"model", "checkpoint"} else fmt(row.get(k)) for k in FIELDS})
    print(f"Saved summary: {OUT_CSV}")

    print("\nTop checkpoints by target3:")
    for row in rows[:10]:
        print_row("  ", row)

    print("\nBest checkpoint per experiment by target3:")
    best_target3 = best_by(rows, "target3")
    for model in sorted(best_target3):
        print_row("  ", best_target3[model])

    print("\nBest checkpoint per experiment by balanced_score:")
    best_balanced = best_by(rows, "balanced_score")
    for model in sorted(best_balanced):
        print_row("  ", best_balanced[model])

    print("\nStability per experiment:")
    for model in sorted(best_target3):
        vals = [rank_key(r, "target3") for r in rows if r["model"] == model and rank_key(r, "target3") >= 0]
        if vals:
            print(f"  {model}: n={len(vals)} | min={min(vals):.6f} | max={max(vals):.6f} | spread={max(vals)-min(vals):.6f}")

    print("\nDelta vs A11 baseline:")
    for model in sorted(best_target3):
        print(f"  {model}: {delta_text(best_target3[model], 'A11_Cons005_w20000')}")

    print("\nDelta vs A11+SH baseline:")
    for model in sorted(best_target3):
        print(f"  {model}: {delta_text(best_target3[model], 'A11_Cons005_w20000_SH')}")

    print("\nRecommendation:")
    candidates = sorted(rows, key=lambda r: (rank_key(r, "target3"), rank_key(r, "destroyed_iou"), -rank_key(r, "damaged_to_intact")), reverse=True)
    for row in candidates[:5]:
        print_row("  ", row)
        print(f"    labels: {', '.join(labels(row))}")


if __name__ == "__main__":
    main()
