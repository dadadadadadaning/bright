#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize A11_next official evaluation results.

Scans:
  /root/shared-nvme/BRIGHT-master/eval_results_official/A11_next/

Writes:
  A11_next_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(os.environ.get("ROOT", "/root/shared-nvme/BRIGHT-master"))
EVAL_ROOT = ROOT / "eval_results_official" / "A11_next"
OUT_CSV = EVAL_ROOT / "A11_next_summary.csv"

FIELDS = [
    "model",
    "checkpoint",
    "official4",
    "target3",
    "DmgIoU",
    "DesIoU",
    "Dmg_to_Int",
    "Des_to_Int",
    "IntactIoU",
    "BackgroundIoU",
]


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_confusion_csv(path: Path) -> Optional[List[List[float]]]:
    if not path.exists():
        return None

    rows: List[List[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            vals: List[float] = []
            for cell in row:
                try:
                    vals.append(float(cell))
                except Exception:
                    pass
            if len(vals) >= 4:
                rows.append(vals[-4:])

    if len(rows) >= 4:
        return rows[-4:]
    return None


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def compute_from_confusion(conf: List[List[float]]) -> Dict[str, float]:
    ious = []
    for c in range(4):
        tp = conf[c][c]
        row_sum = sum(conf[c])
        col_sum = sum(conf[r][c] for r in range(4))
        ious.append(safe_div(tp, row_sum + col_sum - tp))

    gt_damaged = sum(conf[2])
    gt_destroyed = sum(conf[3])
    return {
        "BackgroundIoU": ious[0],
        "IntactIoU": ious[1],
        "DmgIoU": ious[2],
        "DesIoU": ious[3],
        "official4": sum(ious) / 4.0,
        "target3": sum(ious[1:4]) / 3.0,
        "Dmg_to_Int": safe_div(conf[2][1], gt_damaged),
        "Des_to_Int": safe_div(conf[3][1], gt_destroyed),
    }


def parse_metrics_txt(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    mapping = {
        "BackgroundIoU": r"background_iou:\s*([0-9.eE+-]+)",
        "IntactIoU": r"intact_iou:\s*([0-9.eE+-]+)",
        "DmgIoU": r"damaged_iou:\s*([0-9.eE+-]+)",
        "DesIoU": r"destroyed_iou:\s*([0-9.eE+-]+)",
        "official4": r"official_style_4class_miou:\s*([0-9.eE+-]+)",
        "target3": r"target_3class_miou:\s*([0-9.eE+-]+)",
        "Dmg_to_Int": r"damaged_to_intact:\s*([0-9.eE+-]+)",
        "Des_to_Int": r"destroyed_to_intact:\s*([0-9.eE+-]+)",
    }
    out: Dict[str, float] = {}
    for key, pattern in mapping.items():
        match = re.search(pattern, text)
        if match:
            val = to_float(match.group(1))
            if val is not None:
                out[key] = val
    return out


def parse_per_class_csv(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    class_to_key = {
        "0": "BackgroundIoU",
        "1": "IntactIoU",
        "2": "DmgIoU",
        "3": "DesIoU",
        "background": "BackgroundIoU",
        "intact": "IntactIoU",
        "damaged": "DmgIoU",
        "destroyed": "DesIoU",
    }
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_id = str(row.get("class_id", "")).strip().lower()
            cls_name = str(row.get("class_name", "")).strip().lower()
            key = class_to_key.get(cls_id) or class_to_key.get(cls_name)
            val = to_float(row.get("iou"))
            if key and val is not None:
                out[key] = val
    if all(k in out for k in ("BackgroundIoU", "IntactIoU", "DmgIoU", "DesIoU")):
        out["official4"] = (
            out["BackgroundIoU"] + out["IntactIoU"] + out["DmgIoU"] + out["DesIoU"]
        ) / 4.0
        out["target3"] = (out["IntactIoU"] + out["DmgIoU"] + out["DesIoU"]) / 3.0
    return out


def collect_one(model: str, checkpoint: str, ckpt_dir: Path) -> Dict[str, Any]:
    summary = read_json(ckpt_dir / "summary_official.json")
    row: Dict[str, Any] = {
        "model": model,
        "checkpoint": checkpoint,
        "official4": to_float(summary.get("official_style_4class_miou")),
        "target3": to_float(summary.get("target_3class_miou")),
        "DmgIoU": to_float(summary.get("damaged_iou")),
        "DesIoU": to_float(summary.get("destroyed_iou")),
        "Dmg_to_Int": to_float(summary.get("damaged_to_intact")),
        "Des_to_Int": to_float(summary.get("destroyed_to_intact")),
        "IntactIoU": to_float(summary.get("intact_iou")),
        "BackgroundIoU": to_float(summary.get("background_iou")),
    }

    txt_metrics = parse_metrics_txt(ckpt_dir / "official_metrics.txt")
    for key, value in txt_metrics.items():
        if row.get(key) is None:
            row[key] = value

    per_class_metrics = parse_per_class_csv(ckpt_dir / "per_class_metrics_official.csv")
    for key, value in per_class_metrics.items():
        if row.get(key) is None:
            row[key] = value

    conf = read_confusion_csv(ckpt_dir / "confusion_4class_official.csv")
    if conf is not None:
        conf_metrics = compute_from_confusion(conf)
        for key, value in conf_metrics.items():
            if row.get(key) is None:
                row[key] = value

    return row


def fmt(value: Any) -> Any:
    val = to_float(value)
    if val is None:
        return ""
    return f"{val:.6f}"


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
            if any(row.get(k) is not None for k in FIELDS[2:]):
                rows.append(row)

    rows.sort(key=lambda r: (to_float(r.get("target3")) is not None, to_float(r.get("target3")) or -1.0), reverse=True)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) if key in ("model", "checkpoint") else fmt(row.get(key)) for key in FIELDS})

    print(f"Saved summary: {OUT_CSV}")
    print("")
    print("Best checkpoint per experiment:")
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        if model not in best_by_model:
            best_by_model[model] = row
    for model in sorted(best_by_model):
        row = best_by_model[model]
        print(
            f"  {model}: {row['checkpoint']} | "
            f"target3={fmt(row.get('target3'))} | "
            f"official4={fmt(row.get('official4'))} | "
            f"DmgIoU={fmt(row.get('DmgIoU'))} | "
            f"DesIoU={fmt(row.get('DesIoU'))} | "
            f"Dmg->Int={fmt(row.get('Dmg_to_Int'))} | "
            f"Des->Int={fmt(row.get('Des_to_Int'))}"
        )

    print("")
    print("Automatic checks:")
    for model in ("A11_ExistOnly_seed42", "A11_ExistOnly_seed2026"):
        row = best_by_model.get(model)
        ok = row is not None and (to_float(row.get("target3")) or 0.0) >= 0.52
        print(f"  {model} reproduces high target3>=0.52: {ok}")

    baseline_ref = 0.441744
    for model in ("A11_Cons005_seed3407", "A11_Cons01_seed3407"):
        model_rows = [r for r in rows if r["model"] == model]
        primary = [r for r in model_rows if r["checkpoint"] in {"best_valid_miou", "best_balanced_score", "latest_resume"}]
        vals = [to_float(r.get("target3")) for r in primary]
        vals = [v for v in vals if v is not None]
        stable = bool(vals) and min(vals) > baseline_ref
        print(f"  {model} more stable than A11_ExistOnly best_valid_miou target3={baseline_ref:.6f}: {stable}")

    checks = [
        ("Any target3 >= 0.52", lambda r: (to_float(r.get("target3")) or 0.0) >= 0.52),
        ("Any DesIoU >= 0.48", lambda r: (to_float(r.get("DesIoU")) or 0.0) >= 0.48),
        ("Any Dmg_to_Int <= 0.30", lambda r: (to_float(r.get("Dmg_to_Int")) is not None and to_float(r.get("Dmg_to_Int")) <= 0.30)),
        ("Any Des_to_Int <= 0.30", lambda r: (to_float(r.get("Des_to_Int")) is not None and to_float(r.get("Des_to_Int")) <= 0.30)),
    ]
    for label, fn in checks:
        hits = [r for r in rows if fn(r)]
        print(f"  {label}: {bool(hits)}")
        if hits:
            best = hits[0]
            print(f"    best hit: {best['model']} / {best['checkpoint']} target3={fmt(best.get('target3'))}")


if __name__ == "__main__":
    main()
