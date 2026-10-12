#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize next-direction official experiments without pandas."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(os.environ.get("ROOT", "/root/shared-nvme/BRIGHT-master"))
EVAL_ROOT = ROOT / "eval_results_official" / "next_directions"
BASELINE_ROOT = ROOT / "eval_results_official" / "A11_SH_Under"
OUT_CSV = EVAL_ROOT / "next_directions_summary.csv"

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
    "event_wise_miou",
    "worst_event",
    "damage_score",
    "balanced_score",
]


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


def worst_event_name(summary: Dict[str, Any]) -> str:
    worst = summary.get("event_worst")
    if isinstance(worst, dict):
        return str(worst.get("event", ""))
    return str(worst or "")


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
        "event_wise_miou": to_float(summary.get("event_wise_miou") or summary.get("event_mean_official_style_4class_miou")),
        "worst_event": worst_event_name(summary),
    }
    for key in FIELDS[4:14]:
        row[key] = to_float(summary.get(key))
    conf = read_confusion_csv(ckpt_dir / "confusion_4class_official.csv")
    if conf is not None:
        for key, value in metrics_from_confusion(conf).items():
            if row.get(key) is None:
                row[key] = value
    add_scores(row)
    return row


def collect_root(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for ckpt_dir in sorted(model_dir.iterdir()):
            if ckpt_dir.is_dir():
                row = collect_one(model_dir.name, ckpt_dir.name, ckpt_dir)
                if any(row.get(k) is not None for k in FIELDS[2:] if k != "worst_event"):
                    rows.append(row)
    return rows


def fmt(value: Any) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.6f}"


def rank_key(row: Dict[str, Any], key: str) -> float:
    val = to_float(row.get(key))
    return val if val is not None else -1.0


def print_row(prefix: str, row: Dict[str, Any]) -> None:
    print(
        f"{prefix}{row['model']} / {row['checkpoint']} | target3={fmt(row.get('target3'))} | "
        f"DmgIoU={fmt(row.get('damaged_iou'))} | DesIoU={fmt(row.get('destroyed_iou'))} | "
        f"Dmg->Int={fmt(row.get('damaged_to_intact'))} | Des->Int={fmt(row.get('destroyed_to_intact'))} | "
        f"Dmg->Des={fmt(row.get('damaged_to_destroyed'))} | Des->Dmg={fmt(row.get('destroyed_to_damaged'))} | "
        f"Dmg->Bg={fmt(row.get('damaged_to_background'))} | Des->Bg={fmt(row.get('destroyed_to_background'))} | "
        f"event={fmt(row.get('event_wise_miou'))} | worst={row.get('worst_event', '')}"
    )


def best_by_model(rows: List[Dict[str, Any]], key: str = "target3") -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: rank_key(r, key), reverse=True):
        best.setdefault(str(row["model"]), row)
    return best


def delta(row: Dict[str, Any], base: Dict[str, Any], key: str) -> str:
    val = to_float(row.get(key))
    ref = to_float(base.get(key))
    return "" if val is None or ref is None else f"{val - ref:+.6f}"


def direction_name(model: str) -> str:
    if "DamageAwareSampler" in model:
        return "sampler"
    if "BDPCDE" in model:
        return "bdpcde"
    if "SFDG" in model:
        return "sfdg"
    return model


def main() -> None:
    if not EVAL_ROOT.exists():
        raise SystemExit(f"Evaluation root does not exist: {EVAL_ROOT}")
    rows = collect_root(EVAL_ROOT)
    rows.sort(key=lambda r: rank_key(r, "target3"), reverse=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if k in {"model", "checkpoint", "worst_event"} else fmt(row.get(k)) for k in FIELDS})
    print(f"Saved summary: {OUT_CSV}")

    baseline_rows = collect_root(BASELINE_ROOT)
    baseline = max(baseline_rows, key=lambda r: rank_key(r, "target3")) if baseline_rows else None
    if baseline:
        print("\nA11+SH+Under baseline:")
        print_row("  ", baseline)
    else:
        print("\nA11+SH+Under baseline not found under eval_results_official/A11_SH_Under")

    print("\nBest checkpoint per experiment:")
    best = best_by_model(rows, "target3")
    for model in sorted(best):
        print_row("  ", best[model])
        if baseline:
            print(
                f"    delta target3={delta(best[model], baseline, 'target3')} | "
                f"DmgIoU={delta(best[model], baseline, 'damaged_iou')} | "
                f"DesIoU={delta(best[model], baseline, 'destroyed_iou')} | "
                f"Dmg->Int={delta(best[model], baseline, 'damaged_to_intact')} | "
                f"Des->Int={delta(best[model], baseline, 'destroyed_to_intact')}"
            )

    print("\nRanking by target3:")
    for row in rows[:10]:
        print_row("  ", row)

    print("\nRanking by damaged IoU:")
    for row in sorted(rows, key=lambda r: rank_key(r, "damaged_iou"), reverse=True)[:10]:
        print_row("  ", row)

    print("\nRanking by destroyed IoU:")
    for row in sorted(rows, key=lambda r: rank_key(r, "destroyed_iou"), reverse=True)[:10]:
        print_row("  ", row)

    print("\nRanking by underestimation suppression:")
    for row in sorted(rows, key=lambda r: (rank_key(r, "damaged_to_intact") + rank_key(r, "destroyed_to_intact"))):
        print_row("  ", row)

    print("\nRecommendation for next combination:")
    improved = []
    if baseline:
        for model, row in best.items():
            if rank_key(row, "target3") > rank_key(baseline, "target3"):
                improved.append(direction_name(model))
    improved = sorted(set(improved))
    if not improved:
        print("  No direction clearly beats the A11+SH+Under baseline by target3; inspect protocol, checkpoints, stability, seeds, and sampling statistics before adding modules.")
    elif "sampler" in improved and "bdpcde" in improved:
        print("  Recommended: A11_Cons005_w20000_SH_Under005_BDPCDE_BD_DamageAwareSampler_seed3407_C640_I120k_BS2")
    elif "sampler" in improved and "sfdg" in improved:
        print("  Recommended: A11_Cons005_w20000_SH_Under005_SFDG_DamageAwareSampler_seed3407_C640_I120k_BS2")
    elif "bdpcde" in improved and "sfdg" in improved:
        print("  Possible but higher risk: A11_Cons005_w20000_SH_Under005_BDPCDE_BD_SFDG_seed3407_C640_I120k_BS2")
    else:
        print(f"  Only {', '.join(improved)} improved; continue that direction before combining.")


if __name__ == "__main__":
    main()
