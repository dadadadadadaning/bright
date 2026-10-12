#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize unified official BRIGHT evaluation results for paper tables.

Expected directory layout:
  eval_root/model_name/seed3407/best_damage_score/summary_official.json
  eval_root/model_name/seed3407/best_damage_score/event_metrics_official.csv
  eval_root/model_name/seed3407/best_damage_score/event_destroyed_diagnosis.csv

Outputs:
  overall_summary.csv
  seed_mean_std_summary.csv
  event_wise_summary.csv
  error_transfer_summary.csv
  worst_event_top10.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


METRICS = [
    "official_style_4class_miou",
    "target_3class_miou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_to_intact",
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "destroyed_to_background",
    "intact_iou",
]

PRIMARY_CHECKPOINTS = {"best_damage_score", "best_valid_miou"}


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_runs(eval_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in sorted(os.listdir(eval_root)):
        model_dir = os.path.join(eval_root, model)
        if not os.path.isdir(model_dir):
            continue
        for seed_name in sorted(os.listdir(model_dir)):
            seed_dir = os.path.join(model_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            seed = seed_name.replace("seed", "")
            for checkpoint in sorted(os.listdir(seed_dir)):
                ckpt_dir = os.path.join(seed_dir, checkpoint)
                summary_path = os.path.join(ckpt_dir, "summary_official.json")
                if not os.path.exists(summary_path):
                    continue
                summary = read_json(summary_path)
                row: Dict[str, Any] = {
                    "model": model,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "summary_path": summary_path,
                }
                for metric in METRICS:
                    row[metric] = to_float(summary.get(metric))
                rows.append(row)
    return rows


def mean_std(values: List[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0


def build_seed_summary(overall_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in overall_rows:
        if row["checkpoint"] in PRIMARY_CHECKPOINTS:
            grouped[(row["model"], row["checkpoint"])].append(row)

    out: List[Dict[str, Any]] = []
    for (model, checkpoint), rows in sorted(grouped.items()):
        result: Dict[str, Any] = {
            "model": model,
            "checkpoint": checkpoint,
            "num_seeds": len(rows),
        }
        for metric in METRICS:
            m, s = mean_std([to_float(r.get(metric)) for r in rows])
            result[f"{metric}_mean"] = m
            result[f"{metric}_std"] = s
            result[f"{metric}_mean_std"] = f"{m:.6f} ± {s:.6f}" if not math.isnan(m) else "nan"
        out.append(result)
    return out


def build_event_and_error_rows(eval_root: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    event_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    worst_rows: List[Dict[str, Any]] = []

    for model in sorted(os.listdir(eval_root)):
        model_dir = os.path.join(eval_root, model)
        if not os.path.isdir(model_dir):
            continue
        for seed_name in sorted(os.listdir(model_dir)):
            seed_dir = os.path.join(model_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            seed = seed_name.replace("seed", "")
            for checkpoint in sorted(os.listdir(seed_dir)):
                ckpt_dir = os.path.join(seed_dir, checkpoint)
                event_path = os.path.join(ckpt_dir, "event_metrics_official.csv")
                diag_path = os.path.join(ckpt_dir, "event_destroyed_diagnosis.csv")
                for row in read_csv(event_path):
                    event = row.get("event", "")
                    out = {
                        "model": model,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "event": event,
                        "event_target3": to_float(row.get("target_3class_miou")),
                        "event_DmgIoU": to_float(row.get("damaged_iou")),
                        "event_DesIoU": to_float(row.get("destroyed_iou")),
                        "event_Dmg_to_Int": to_float(row.get("damaged_to_intact")),
                        "event_Des_to_Int": to_float(row.get("destroyed_to_intact")),
                        "event_Des_to_Dmg": to_float(row.get("destroyed_to_damaged")),
                        "event_Des_to_Bg": to_float(row.get("destroyed_to_background")),
                    }
                    event_rows.append(out)
                    if checkpoint in PRIMARY_CHECKPOINTS:
                        worst_score = (
                            (1.0 - out["event_target3"] if not math.isnan(out["event_target3"]) else 0.0)
                            + out["event_Dmg_to_Int"]
                            + out["event_Des_to_Int"]
                        )
                        worst_rows.append({**out, "worst_score": worst_score})

                for row in read_csv(diag_path):
                    error_rows.append({
                        "model": model,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "event": row.get("event", ""),
                        "diagnosis_code": row.get("diagnosis_code", row.get("code", "")),
                        "target_3class_miou": to_float(row.get("target_3class_miou")),
                        "damaged_iou": to_float(row.get("damaged_iou")),
                        "destroyed_iou": to_float(row.get("destroyed_iou")),
                        "damaged_to_intact": to_float(row.get("damaged_to_intact")),
                        "destroyed_to_intact": to_float(row.get("destroyed_to_intact")),
                        "destroyed_to_damaged": to_float(row.get("destroyed_to_damaged")),
                        "destroyed_to_background": to_float(row.get("destroyed_to_background")),
                    })

    worst_rows.sort(key=lambda r: to_float(r.get("worst_score")), reverse=True)
    return event_rows, error_rows, worst_rows[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    overall_rows = discover_runs(args.eval_root)
    write_csv(
        os.path.join(args.out_dir, "overall_summary.csv"),
        overall_rows,
        ["model", "seed", "checkpoint", *METRICS, "summary_path"],
    )

    seed_rows = build_seed_summary(overall_rows)
    seed_fields = ["model", "checkpoint", "num_seeds"]
    for metric in METRICS:
        seed_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    write_csv(os.path.join(args.out_dir, "seed_mean_std_summary.csv"), seed_rows, seed_fields)

    event_rows, error_rows, worst_rows = build_event_and_error_rows(args.eval_root)
    write_csv(
        os.path.join(args.out_dir, "event_wise_summary.csv"),
        event_rows,
        [
            "model", "seed", "checkpoint", "event", "event_target3", "event_DmgIoU",
            "event_DesIoU", "event_Dmg_to_Int", "event_Des_to_Int",
            "event_Des_to_Dmg", "event_Des_to_Bg",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "error_transfer_summary.csv"),
        error_rows,
        [
            "model", "seed", "checkpoint", "event", "diagnosis_code",
            "target_3class_miou", "damaged_iou", "destroyed_iou",
            "damaged_to_intact", "destroyed_to_intact",
            "destroyed_to_damaged", "destroyed_to_background",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "worst_event_top10.csv"),
        worst_rows,
        [
            "model", "seed", "checkpoint", "event", "worst_score",
            "event_target3", "event_DmgIoU", "event_DesIoU",
            "event_Dmg_to_Int", "event_Des_to_Int",
        ],
    )
    print(f"Wrote unified summaries to: {args.out_dir}")


if __name__ == "__main__":
    main()
