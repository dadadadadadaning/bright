#!/usr/bin/env python3
"""Create a readable, reproducible P3 event-wise evaluation report.

This script is deliberately inference-free.  It consumes the files produced by
``full_evaluate_unified.py --eval_protocol official`` so event metrics use the
same confusion matrices as the official aggregate result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


CLASS_IDS = {"background": 0, "intact": 1, "damaged": 2, "destroyed": 3}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _f1(precision: float, recall: float) -> float:
    if not (math.isfinite(precision) and math.isfinite(recall)):
        return float("nan")
    return _safe_div(2.0 * precision * recall, precision + recall)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_confusions(path: Path) -> Dict[str, List[List[int]]]:
    rows = _read_csv(path)
    result: Dict[str, List[List[int]]] = {}
    for row in rows:
        result[row["event"]] = [
            [int(row[f"conf_{gt}{pred}"]) for pred in range(4)]
            for gt in range(4)
        ]
    return result


def derive_event_metrics(confusion: List[List[int]]) -> Dict[str, float]:
    """Derive occurrence and error-flow metrics from a 4x4 GT x prediction matrix."""
    building_gt = sum(confusion[gt][pred] for gt in (1, 2, 3) for pred in range(4))
    building_pred = sum(confusion[gt][pred] for gt in range(4) for pred in (1, 2, 3))
    building_tp = sum(confusion[gt][pred] for gt in (1, 2, 3) for pred in (1, 2, 3))
    loc_precision = _safe_div(building_tp, building_pred)
    loc_recall = _safe_div(building_tp, building_gt)

    affected_gt = sum(confusion[gt][pred] for gt in (2, 3) for pred in range(4))
    affected_tp = sum(confusion[gt][pred] for gt in (2, 3) for pred in (2, 3))
    damaged_gt = sum(confusion[2])

    return {
        "loc_precision": loc_precision,
        "loc_recall": loc_recall,
        "loc_f1_derived": _f1(loc_precision, loc_recall),
        "affected_recall": _safe_div(affected_tp, affected_gt),
        "damaged_gate_miss": _safe_div(confusion[2][0] + confusion[2][1], damaged_gt),
        "damaged_to_background": _safe_div(confusion[2][0], damaged_gt),
        "damaged_to_intact": _safe_div(confusion[2][1], damaged_gt),
        "damaged_to_destroyed": _safe_div(confusion[2][3], damaged_gt),
    }


def _as_float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(row: Mapping[str, str], key: str) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _percent(value: float) -> float:
    return 100.0 * value if math.isfinite(value) else value


def build_rows(
    official_rows: Iterable[Mapping[str, str]],
    confusions: Mapping[str, List[List[int]]],
    baseline_rows: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for official in official_rows:
        event = official["event"]
        if event not in confusions:
            raise KeyError(f"event {event!r} is missing from event confusion CSV")
        derived = derive_event_metrics(confusions[event])
        row: Dict[str, object] = {
            "event": event,
            "num_images": _as_int(official, "num_images"),
            "num_valid_pixels": _as_int(official, "num_valid_pixels"),
            "num_building_pixels": _as_int(official, "num_building_pixels"),
            "num_intact_pixels": _as_int(official, "num_intact_pixels"),
            "num_damaged_pixels": _as_int(official, "num_damaged_pixels"),
            "num_destroyed_pixels": _as_int(official, "num_destroyed_pixels"),
            "valid_miou": _percent(_as_float(official, "target_3class_miou")),
            "miou4": _percent(_as_float(official, "official_style_4class_miou")),
            # P3's reported localisation is the independent RGB localisation
            # head.  Do not silently substitute the building mask derived from
            # the final four-class damage prediction.
            "loc_precision": _percent(_as_float(official, "loc_head_precision")),
            "loc_recall": _percent(_as_float(official, "loc_head_recall")),
            "loc_f1": _percent(_as_float(official, "loc_head_f1")),
            "damage_output_loc_precision": _percent(derived["loc_precision"]),
            "damage_output_loc_recall": _percent(derived["loc_recall"]),
            "damage_output_loc_f1": _percent(derived["loc_f1_derived"]),
            "background_iou": _percent(_as_float(official, "background_iou")),
            "intact_iou": _percent(_as_float(official, "intact_iou")),
            "damaged_iou": _percent(_as_float(official, "damaged_iou")),
            "destroyed_iou": _percent(_as_float(official, "destroyed_iou")),
            "affected_recall": _percent(derived["affected_recall"]),
            "damaged_gate_miss": _percent(derived["damaged_gate_miss"]),
            "damaged_to_background": _percent(derived["damaged_to_background"]),
            "damaged_to_intact": _percent(derived["damaged_to_intact"]),
            "damaged_to_destroyed": _percent(derived["damaged_to_destroyed"]),
        }
        if baseline_rows is not None and event in baseline_rows:
            baseline = baseline_rows[event]
            row["delta_valid_miou_pp"] = row["valid_miou"] - _percent(
                _as_float(baseline, "target_3class_miou")
            )
            row["delta_miou4_pp"] = row["miou4"] - _percent(
                _as_float(baseline, "official_style_4class_miou")
            )
            row["delta_damaged_iou_pp"] = row["damaged_iou"] - _percent(
                _as_float(baseline, "damaged_iou")
            )
        result.append(row)
    return sorted(result, key=lambda item: str(item["event"]))


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "NA"


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, object]], summary: Mapping[str, object]) -> None:
    has_delta = bool(rows and "delta_miou4_pp" in rows[0])
    headers = ["Event", "Tiles", "Valid", "mIoU4", "Loc F1", "Intact", "Damaged", "Destroyed", "Affected R", "D gate-miss"]
    if has_delta:
        headers.append("Delta mIoU4")
    lines = [
        "# P3 event-wise official evaluation",
        "",
        f"- Split: `{summary['split']}`",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Event mean mIoU4: **{_fmt(summary['event_mean_miou4'])}**",
        f"- Event worst: **{summary['event_worst_name']} ({_fmt(summary['event_worst_miou4'])})**",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        values = [
            str(row["event"]),
            str(row["num_images"]),
            _fmt(row["valid_miou"]),
            _fmt(row["miou4"]),
            _fmt(row["loc_f1"]),
            _fmt(row["intact_iou"]),
            _fmt(row["damaged_iou"]),
            _fmt(row["destroyed_iou"]),
            _fmt(row["affected_recall"]),
            _fmt(row["damaged_gate_miss"]),
        ]
        if has_delta:
            values.append(_fmt(row["delta_miou4_pp"]))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--baseline_event_csv", type=Path)
    parser.add_argument("--expected_event_mean", type=float)
    parser.add_argument("--expected_event_worst", type=float)
    parser.add_argument("--expected_tolerance", type=float, default=0.02)
    parser.add_argument("--expected_num_events", type=int, default=14)
    args = parser.parse_args()

    official_csv = args.eval_dir / "event_metrics_official.csv"
    confusion_csv = args.eval_dir / "event_confusion_4class.csv"
    overall_json = args.eval_dir / "summary_official.json"
    for required in (official_csv, confusion_csv, overall_json):
        if not required.is_file():
            raise FileNotFoundError(required)

    official_rows = _read_csv(official_csv)
    confusions = _read_confusions(confusion_csv)
    baseline_rows = None
    if args.baseline_event_csv:
        baseline_rows = {row["event"]: row for row in _read_csv(args.baseline_event_csv)}
    rows = build_rows(official_rows, confusions, baseline_rows)
    if not rows:
        raise RuntimeError("official event CSV contains no event rows")

    mean_miou4 = sum(float(row["miou4"]) for row in rows) / len(rows)
    worst = min(rows, key=lambda row: float(row["miou4"]))
    overall = json.loads(overall_json.read_text(encoding="utf-8"))
    official_mean = 100.0 * float(overall["event_mean_official_style_4class_miou"])
    if abs(mean_miou4 - official_mean) > 1e-6:
        raise AssertionError(
            f"event mean mismatch: rows={mean_miou4:.8f}, official={official_mean:.8f}"
        )
    consistency_checks = {
        "expected_num_events": len(rows) == args.expected_num_events,
        "rows_reproduce_official_event_mean": True,
        "expected_event_mean": (
            args.expected_event_mean is None
            or abs(mean_miou4 - args.expected_event_mean) <= args.expected_tolerance
        ),
        "expected_event_worst": (
            args.expected_event_worst is None
            or abs(float(worst["miou4"]) - args.expected_event_worst) <= args.expected_tolerance
        ),
    }
    if not all(consistency_checks.values()):
        raise AssertionError(
            "event-wise result does not reproduce the locked P3 validation reference: "
            f"mean={mean_miou4:.4f}, worst={float(worst['miou4']):.4f}, "
            f"checks={consistency_checks}"
        )

    summary: Dict[str, object] = {
        "protocol": "P3_eventwise_official_v1",
        "split": args.split,
        "checkpoint_sha256": args.checkpoint_sha256,
        "num_events": len(rows),
        "event_mean_miou4": mean_miou4,
        "event_worst_name": worst["event"],
        "event_worst_miou4": worst["miou4"],
        "aggregate_valid_miou": 100.0 * float(overall["target_3class_miou"]),
        "aggregate_miou4": 100.0 * float(overall["official_style_4class_miou"]),
        "test_accessed": args.split == "test",
        "consistency_checks": consistency_checks,
        "events": rows,
    }
    (args.eval_dir / "eventwise_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_tsv(args.eval_dir / "eventwise_summary.tsv", rows)
    write_markdown(args.eval_dir / "eventwise_summary.md", rows, summary)

    print("=" * 104)
    print(f"Split/Event count : {args.split} / {len(rows)}")
    print(f"Event mean mIoU4  : {mean_miou4:.4f}")
    print(f"Event worst       : {worst['event']} / {float(worst['miou4']):.4f}")
    print(f"Saved JSON/TSV/MD : {args.eval_dir}")
    print(f"Test accessed     : {args.split == 'test'}")


if __name__ == "__main__":
    main()
