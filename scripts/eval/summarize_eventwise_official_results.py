#!/usr/bin/env python3
"""Build cross-model official and event-wise BRIGHT evaluation tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-master/eval_results_official"
DEFAULT_OUTPUT_DIR = "/root/shared-nvme/BRIGHT-master/eval_analysis"

CHECKPOINT_PRIORITY = {
    "best_valid_miou": 0,
    "best_balanced_score": 1,
    "best_model": 2,
    "best": 3,
    "best_destroyed_iou": 4,
}

ALL_MODEL_FIELDS = [
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
    "destroyed_to_damaged",
    "destroyed_to_background",
    "loc_head_iou",
    "loc_head_f1",
    "damage_building_iou",
    "damage_building_f1",
    "load_missing_keys_count",
    "load_unexpected_keys_count",
    "save_dir",
]

EVENT_FIELDS = [
    "model",
    "checkpoint",
    "event",
    "num_images",
    "num_pixels",
    "num_valid_pixels",
    "num_building_pixels",
    "num_intact_pixels",
    "num_damaged_pixels",
    "num_destroyed_pixels",
    "target_3class_miou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_recall",
    "destroyed_recall",
    "damaged_precision",
    "destroyed_precision",
    "damaged_to_intact",
    "damaged_to_destroyed",
    "damaged_to_background",
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "destroyed_to_background",
    "loc_head_iou",
    "damage_building_iou",
    "impact_score",
    "destroyed_impact_score",
]

DIAGNOSIS_FIELDS = [
    "model",
    "checkpoint",
    "event",
    "num_destroyed_pixels",
    "destroyed_iou",
    "destroyed_precision",
    "destroyed_recall",
    "destroyed_to_background",
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "destroyed_fp_from_background_count",
    "destroyed_fp_from_intact_count",
    "destroyed_fp_from_damaged_count",
    "destroyed_fp_from_background_share",
    "destroyed_fp_from_intact_share",
    "destroyed_fp_from_damaged_share",
    "intact_to_destroyed",
    "damaged_to_destroyed",
    "damage_building_iou",
    "destroyed_impact_score",
    "diagnosis_code",
    "diagnosis_text",
    "recommended_direction",
    "support_warning",
]

WORST_FIELDS = [
    "model",
    "checkpoint",
    "ranking_type",
    "rank",
    "event",
    "metric_value",
    "num_images",
    "num_building_pixels",
    "num_destroyed_pixels",
    "target_3class_miou",
    "destroyed_iou",
    "destroyed_to_intact",
    "diagnosis_code",
    "recommended_direction",
]

COMPARISON_FIELDS = [
    "event",
    "model",
    "checkpoint",
    "metric",
    "value",
    "delta_vs_M03_A4",
    "rank_within_event",
]

COMPARISON_METRICS = [
    "target_3class_miou",
    "damaged_iou",
    "destroyed_iou",
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "destroyed_to_background",
    "destroyed_recall",
    "destroyed_precision",
    "damage_building_iou",
]

LOWER_IS_BETTER = {
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "destroyed_to_background",
}


def iter_summary_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        if "summary_official.json" in filenames:
            yield os.path.join(dirpath, "summary_official.json")


def infer_model_checkpoint(root: str, summary_path: str) -> Tuple[str, str]:
    save_dir = os.path.dirname(summary_path)
    parts = os.path.relpath(save_dir, root).split(os.sep)
    model = parts[0] if parts else ""
    checkpoint = parts[-1] if len(parts) >= 2 else ""
    return model, checkpoint


def read_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def is_finite(value: Any) -> bool:
    return math.isfinite(as_float(value))


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def write_rows(path: str, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def checkpoint_sort_key(checkpoint: str) -> Tuple[int, str]:
    return CHECKPOINT_PRIORITY.get(checkpoint, 99), checkpoint


def collect(root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    overall_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    diagnosis_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for summary_path in sorted(iter_summary_files(root)):
        save_dir = os.path.dirname(summary_path)
        model, checkpoint = infer_model_checkpoint(root, summary_path)
        summary = read_json(summary_path)
        overall_rows.append({
            "model": model,
            "checkpoint": checkpoint,
            **summary,
            "loc_head_iou": summary.get("loc_head_iou", summary.get("loc_iou", "")),
            "loc_head_f1": summary.get("loc_head_f1", summary.get("loc_f1", "")),
            "save_dir": save_dir,
        })

        event_path = os.path.join(save_dir, "event_metrics_official.csv")
        diagnosis_path = os.path.join(save_dir, "event_destroyed_diagnosis.csv")
        confusion_path = os.path.join(save_dir, "event_confusion_4class.csv")
        if not os.path.isfile(event_path):
            warnings.append(f"missing event metrics: {event_path}")
            continue
        if not os.path.isfile(confusion_path):
            warnings.append(f"missing event confusion: {confusion_path}")

        current_events = read_csv(event_path)
        for row in current_events:
            event_rows.append({"model": model, "checkpoint": checkpoint, **row})

        current_diagnosis = read_csv(diagnosis_path)
        if current_diagnosis:
            for row in current_diagnosis:
                diagnosis_rows.append({"model": model, "checkpoint": checkpoint, **row})
        else:
            warnings.append(f"missing destroyed diagnosis: {diagnosis_path}")
            for row in current_events:
                diagnosis_rows.append({"model": model, "checkpoint": checkpoint, **row})

    overall_rows.sort(
        key=lambda row: (
            -as_float(row.get("target_3class_miou"))
            if is_finite(row.get("target_3class_miou"))
            else float("inf"),
            row.get("model", ""),
            checkpoint_sort_key(str(row.get("checkpoint", ""))),
        )
    )
    event_rows.sort(key=lambda row: (row["model"], checkpoint_sort_key(row["checkpoint"]), row["event"]))
    diagnosis_rows.sort(
        key=lambda row: (
            -as_float(row.get("destroyed_impact_score"))
            if is_finite(row.get("destroyed_impact_score"))
            else float("inf"),
            row["model"],
            row["event"],
        )
    )
    return overall_rows, event_rows, diagnosis_rows, warnings


def build_worst_rows(event_rows: Sequence[Mapping[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(str(row["model"]), str(row["checkpoint"]))].append(row)

    rankings = [
        ("overall_impact", "impact_score", True),
        ("destroyed_impact", "destroyed_impact_score", True),
        ("lowest_target_3class_miou", "target_3class_miou", False),
        ("lowest_destroyed_iou", "destroyed_iou", False),
        ("highest_destroyed_to_intact", "destroyed_to_intact", True),
    ]
    output: List[Dict[str, Any]] = []
    for (model, checkpoint), rows in sorted(grouped.items()):
        for ranking_type, metric, descending in rankings:
            valid = [row for row in rows if is_finite(row.get(metric))]
            valid.sort(key=lambda row: as_float(row[metric]), reverse=descending)
            for rank, row in enumerate(valid[:top_k], start=1):
                output.append({
                    "model": model,
                    "checkpoint": checkpoint,
                    "ranking_type": ranking_type,
                    "rank": rank,
                    "event": row.get("event", ""),
                    "metric_value": row.get(metric, ""),
                    **row,
                })
    return output


def choose_baselines(event_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    candidates: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if row.get("model") == "M03_A4":
            candidates[str(row["event"])].append(row)
    selected: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for event, rows in candidates.items():
        rows = sorted(rows, key=lambda row: checkpoint_sort_key(str(row["checkpoint"])))
        for metric in COMPARISON_METRICS:
            baseline = next((row for row in rows if is_finite(row.get(metric))), rows[0])
            selected[(event, metric)] = baseline
    return selected


def build_comparison_rows(event_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baselines = choose_baselines(event_rows)
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        for metric in COMPARISON_METRICS:
            if is_finite(row.get(metric)):
                groups[(str(row["event"]), metric)].append(row)

    output: List[Dict[str, Any]] = []
    for (event, metric), rows in sorted(groups.items()):
        reverse = metric not in LOWER_IS_BETTER
        ranked = sorted(rows, key=lambda row: as_float(row[metric]), reverse=reverse)
        baseline = baselines.get((event, metric))
        baseline_value = as_float(baseline.get(metric)) if baseline is not None else float("nan")
        for rank, row in enumerate(ranked, start=1):
            value = as_float(row[metric])
            output.append({
                "event": event,
                "model": row["model"],
                "checkpoint": row["checkpoint"],
                "metric": metric,
                "value": value,
                "delta_vs_M03_A4": value - baseline_value if math.isfinite(baseline_value) else float("nan"),
                "rank_within_event": rank,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--eval_root", dest="root", default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    overall, events, diagnosis, warnings = collect(args.root)
    worst = build_worst_rows(events, args.top_k)
    comparison = build_comparison_rows(events)

    write_rows(os.path.join(args.output_dir, "all_model_official_summary.csv"), ALL_MODEL_FIELDS, overall)
    write_rows(os.path.join(args.output_dir, "event_wise_summary.csv"), EVENT_FIELDS, events)
    write_rows(
        os.path.join(args.output_dir, "event_wise_destroyed_diagnosis.csv"),
        DIAGNOSIS_FIELDS,
        diagnosis,
    )
    write_rows(os.path.join(args.output_dir, "worst_events_top10.csv"), WORST_FIELDS, worst)
    write_rows(os.path.join(args.output_dir, "model_event_comparison.csv"), COMPARISON_FIELDS, comparison)

    print(f"Official summaries: {len(overall)}")
    print(f"Event rows: {len(events)}")
    print(f"Destroyed diagnosis rows: {len(diagnosis)}")
    print(f"Worst-event rows: {len(worst)}")
    print(f"Model-event comparison rows: {len(comparison)}")
    print(f"Output directory: {args.output_dir}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
