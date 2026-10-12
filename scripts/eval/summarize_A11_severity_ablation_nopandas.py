#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize official A11 severity-ablation experiments without pandas."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(os.environ.get("ROOT", "/root/shared-nvme/BRIGHT-master"))
EVAL_ROOT = ROOT / "eval_results_official" / "A11_severity_ablation"
OUT_CSV = EVAL_ROOT / "A11_severity_ablation_summary.csv"

BASELINE = {
    "official4": 0.624483,
    "target3": 0.510170,
    "intact_iou": 0.715835,
    "damaged_iou": 0.358117,
    "destroyed_iou": 0.456557,
    "damaged_to_intact": 0.251663,
    "destroyed_to_intact": 0.215569,
    "damaged_to_destroyed": 0.039612,
    "destroyed_to_damaged": 0.093389,
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
    "delta_target3",
    "delta_damaged_iou",
    "delta_destroyed_iou",
    "delta_damaged_to_destroyed",
    "delta_destroyed_to_damaged",
]

METRIC_KEYS = FIELDS[2:14]


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
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
        reader = csv.reader(f)
        for raw in reader:
            values: List[float] = []
            for cell in raw[1:] if raw else []:
                val = to_float(cell)
                if val is not None:
                    values.append(val)
            if len(values) >= 4:
                rows.append(values[:4])
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
        "damaged_to_intact": safe_div(conf[2][1], gt_damaged),
        "destroyed_to_intact": safe_div(conf[3][1], gt_destroyed),
        "damaged_to_destroyed": safe_div(conf[2][3], gt_damaged),
        "destroyed_to_damaged": safe_div(conf[3][2], gt_destroyed),
        "damaged_to_background": safe_div(conf[2][0], gt_damaged),
        "destroyed_to_background": safe_div(conf[3][0], gt_destroyed),
    }


def add_deltas(row: Dict[str, Any]) -> None:
    for key in (
        "target3",
        "damaged_iou",
        "destroyed_iou",
        "damaged_to_destroyed",
        "destroyed_to_damaged",
    ):
        val = to_float(row.get(key))
        base = BASELINE.get(key)
        row[f"delta_{key}"] = None if val is None or base is None else val - base


def collect_one(model: str, checkpoint: str, ckpt_dir: Path) -> Dict[str, Any]:
    summary = read_json(ckpt_dir / "summary_official.json")
    row: Dict[str, Any] = {
        "model": model,
        "checkpoint": checkpoint,
        "official4": to_float(summary.get("official_style_4class_miou")),
        "target3": to_float(summary.get("target_3class_miou")),
        "background_iou": to_float(summary.get("background_iou")),
        "intact_iou": to_float(summary.get("intact_iou")),
        "damaged_iou": to_float(summary.get("damaged_iou")),
        "destroyed_iou": to_float(summary.get("destroyed_iou")),
        "damaged_to_intact": to_float(summary.get("damaged_to_intact")),
        "destroyed_to_intact": to_float(summary.get("destroyed_to_intact")),
        "damaged_to_destroyed": to_float(summary.get("damaged_to_destroyed")),
        "destroyed_to_damaged": to_float(summary.get("destroyed_to_damaged")),
        "damaged_to_background": to_float(summary.get("damaged_to_background")),
        "destroyed_to_background": to_float(summary.get("destroyed_to_background")),
    }

    conf = read_confusion_csv(ckpt_dir / "confusion_4class_official.csv")
    if conf is not None:
        fallback = metrics_from_confusion(conf)
        for key, value in fallback.items():
            if row.get(key) is None:
                row[key] = value
    add_deltas(row)
    return row


def fmt(value: Any) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.6f}"


def score_key(row: Dict[str, Any]) -> float:
    val = to_float(row.get("target3"))
    return val if val is not None else -1.0


def print_row(prefix: str, row: Dict[str, Any]) -> None:
    print(
        f"{prefix}{row['model']} / {row['checkpoint']} | "
        f"target3={fmt(row.get('target3'))} | "
        f"DmgIoU={fmt(row.get('damaged_iou'))} | "
        f"DesIoU={fmt(row.get('destroyed_iou'))} | "
        f"Dmg->Des={fmt(row.get('damaged_to_destroyed'))} | "
        f"Des->Dmg={fmt(row.get('destroyed_to_damaged'))}"
    )


def judgement_labels(row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    if (to_float(row.get("target3")) or 0.0) >= 0.520:
        labels.append("target3 improvement is clear")
    if (to_float(row.get("destroyed_iou")) or 0.0) >= 0.480:
        labels.append("destroyed improvement is clear")
    des_to_dmg = to_float(row.get("destroyed_to_damaged"))
    if des_to_dmg is not None and des_to_dmg < BASELINE["destroyed_to_damaged"]:
        labels.append("destroyed/damaged confusion decreased")
    dmg_to_des = to_float(row.get("damaged_to_destroyed"))
    if dmg_to_des is not None and dmg_to_des < BASELINE["damaged_to_destroyed"]:
        labels.append("damaged/destroyed confusion decreased")
    dmg_to_int = to_float(row.get("damaged_to_intact"))
    des_to_int = to_float(row.get("destroyed_to_intact"))
    if (
        dmg_to_int is not None
        and des_to_int is not None
        and dmg_to_int <= 0.28
        and des_to_int <= 0.25
    ):
        labels.append("damage underestimation did not rebound clearly")
    return labels or ["needs inspection"]


def main() -> None:
    if not EVAL_ROOT.exists():
        raise SystemExit(f"Evaluation root does not exist: {EVAL_ROOT}")

    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(EVAL_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for ckpt_dir in sorted(model_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            row = collect_one(model_dir.name, ckpt_dir.name, ckpt_dir)
            if any(row.get(key) is not None for key in METRIC_KEYS):
                rows.append(row)

    rows.sort(key=score_key, reverse=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, "") if key in {"model", "checkpoint"} else fmt(row.get(key))
                    for key in FIELDS
                }
            )

    print(f"Saved summary: {OUT_CSV}")

    print("")
    print("Top checkpoints by target3:")
    for row in rows[:10]:
        print_row("  ", row)

    print("")
    print("Best checkpoint per model:")
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        best_by_model.setdefault(str(row["model"]), row)
    for model in sorted(best_by_model):
        print_row("  ", best_by_model[model])

    print("")
    print("Checkpoint stability per model:")
    for model in sorted(best_by_model):
        model_rows = [r for r in rows if r["model"] == model]
        target_vals = [to_float(r.get("target3")) for r in model_rows]
        target_vals = [v for v in target_vals if v is not None]
        latest = next((r for r in model_rows if r["checkpoint"] == "latest_resume"), None)
        if target_vals:
            print(
                f"  {model}: n={len(target_vals)} | min={min(target_vals):.6f} | "
                f"max={max(target_vals):.6f} | spread={max(target_vals) - min(target_vals):.6f} | "
                f"latest_resume_target3={fmt(latest.get('target3')) if latest else 'missing'}"
            )
        else:
            print(f"  {model}: no valid target3 values")

    print("")
    print("Delta vs baseline:")
    for model in sorted(best_by_model):
        row = best_by_model[model]
        print(
            f"  {model}: d_target3={fmt(row.get('delta_target3'))} | "
            f"d_DmgIoU={fmt(row.get('delta_damaged_iou'))} | "
            f"d_DesIoU={fmt(row.get('delta_destroyed_iou'))} | "
            f"d_Dmg->Des={fmt(row.get('delta_damaged_to_destroyed'))} | "
            f"d_Des->Dmg={fmt(row.get('delta_destroyed_to_damaged'))}"
        )

    print("")
    print("Quick judgement:")
    for model in sorted(best_by_model):
        row = best_by_model[model]
        print_row(f"  {model}: ", row)
        for label in judgement_labels(row):
            print(f"    - {label}")


if __name__ == "__main__":
    main()
