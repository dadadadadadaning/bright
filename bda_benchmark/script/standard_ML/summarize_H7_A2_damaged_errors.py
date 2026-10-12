# -*- coding: utf-8 -*-
"""Summarize A2 damaged-class errors from the validation-only H7 audit.

The H7 P3/A2 oracle JSON already contains one 4x4 target-by-prediction
confusion matrix per event.  This script performs no inference and never
accesses the test split; it only turns those stored validation counts into a
class-conditional error report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


def disaster_type(event: str) -> str:
    match = re.search(r"-([^-]+)$", str(event))
    return match.group(1) if match else "unknown"


def _matrix(value: object) -> list[list[int]]:
    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("expected a 4x4 confusion matrix")
    matrix = []
    for row in value:
        if not isinstance(row, Sequence) or len(row) != 4:
            raise ValueError("expected a 4x4 confusion matrix")
        matrix.append([int(cell) for cell in row])
    return matrix


def add_matrix(left: list[list[int]], right: Sequence[Sequence[int]]) -> None:
    for row in range(4):
        for column in range(4):
            left[row][column] += int(right[row][column])


def damaged_row(event: str, matrix: Sequence[Sequence[int]]) -> Dict[str, object]:
    # H7 uses target rows and prediction columns.  Class 2 is Damaged.
    row = [int(value) for value in matrix[2]]
    total = sum(row)
    percentages = [100.0 * value / total if total else math.nan for value in row]
    errors = total - row[2]
    return {
        "event": event,
        "disaster_type": disaster_type(event),
        "gt_damaged_pixels": total,
        "error_pixels": errors,
        "error_rate_pct": 100.0 * errors / total if total else math.nan,
        "pred_background_pixels": row[0],
        "pred_intact_pixels": row[1],
        "pred_damaged_pixels": row[2],
        "pred_destroyed_pixels": row[3],
        "pred_background_pct": percentages[0],
        "pred_intact_pct": percentages[1],
        "pred_damaged_pct": percentages[2],
        "pred_destroyed_pct": percentages[3],
        "underestimate_pct": percentages[0] + percentages[1],
        "overestimate_pct": percentages[3],
    }


def rows_from_payload(payload: Mapping[str, object]) -> list[Dict[str, object]]:
    if bool(payload.get("test_accessed", False)):
        raise ValueError("refusing a test-accessed H7 report")
    raw_events = payload.get("events")
    if not isinstance(raw_events, Sequence):
        raise ValueError("H7 report has no event records")
    rows = []
    for record in raw_events:
        if not isinstance(record, Mapping):
            continue
        confusion = record.get("confusion")
        if not isinstance(confusion, Mapping) or "a2" not in confusion:
            raise ValueError("event record has no A2 confusion matrix")
        rows.append(damaged_row(str(record["event"]), _matrix(confusion["a2"])))
    if not rows:
        raise ValueError("H7 report contains no usable event records")
    return rows


def aggregate_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    totals = [0, 0, 0, 0]
    for record in rows:
        for index, key in enumerate(
            (
                "pred_background_pixels",
                "pred_intact_pixels",
                "pred_damaged_pixels",
                "pred_destroyed_pixels",
            )
        ):
            totals[index] += int(record[key])
    grand_total = sum(totals)
    return {
        f"pred_{name}_pct": 100.0 * value / grand_total if grand_total else math.nan
        for name, value in zip(CLASS_NAMES, totals)
    } | {
        "gt_damaged_pixels": float(grand_total),
        "underestimate_pct": (
            100.0 * (totals[0] + totals[1]) / grand_total if grand_total else math.nan
        ),
        "overestimate_pct": (
            100.0 * totals[3] / grand_total if grand_total else math.nan
        ),
    }


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "event",
        "disaster_type",
        "gt_damaged_pixels",
        "error_pixels",
        "error_rate_pct",
        "pred_background_pixels",
        "pred_intact_pixels",
        "pred_damaged_pixels",
        "pred_destroyed_pixels",
        "pred_background_pct",
        "pred_intact_pct",
        "pred_damaged_pct",
        "pred_destroyed_pct",
        "underestimate_pct",
        "overestimate_pct",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(title: str, rows: Sequence[Mapping[str, object]]) -> None:
    print(f"\n{title}")
    print(
        f"{'Event':<28} {'Type':<12} {'GT-Dmg':>10} {'ErrPix':>10} "
        f"{'Err%':>8} {'D->I':>8} {'D->X':>8} {'D->BG':>8}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{str(row['event']):<28} {str(row['disaster_type']):<12} "
            f"{int(row['gt_damaged_pixels']):10d} {int(row['error_pixels']):10d} "
            f"{float(row['error_rate_pct']):8.2f} "
            f"{float(row['pred_intact_pct']):8.2f} "
            f"{float(row['pred_destroyed_pct']):8.2f} "
            f"{float(row['pred_background_pct']):8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H7 A2 damaged errors")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_tsv", default="")
    parser.add_argument("--top_k", type=int, default=14)
    args = parser.parse_args()

    source = Path(args.input_json)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = rows_from_payload(payload)
    overall = aggregate_rows(rows)

    print("=" * 82)
    print("H7 A2 damaged-error audit | validation only | test accessed: False")
    print(f"GT Damaged pixels : {int(overall['gt_damaged_pixels']):,}")
    print(f"Damaged -> BG     : {overall['pred_background_pct']:.2f}%")
    print(f"Damaged -> Intact : {overall['pred_intact_pct']:.2f}%")
    print(f"Damaged -> Correct: {overall['pred_damaged_pct']:.2f}%")
    print(f"Damaged -> Destroy: {overall['pred_destroyed_pct']:.2f}%")
    print(f"Underestimate     : {overall['underestimate_pct']:.2f}%")
    print(f"Overestimate      : {overall['overestimate_pct']:.2f}%")
    dominant = (
        "INTACT"
        if overall["pred_intact_pct"] >= overall["pred_destroyed_pct"]
        else "DESTROYED"
    )
    print(f"Dominant D error  : Damaged -> {dominant}")

    valid_rows = [row for row in rows if int(row["gt_damaged_pixels"]) > 0]
    top_k = max(1, min(args.top_k, len(valid_rows)))
    by_rate = sorted(valid_rows, key=lambda row: float(row["error_rate_pct"]), reverse=True)
    by_pixels = sorted(valid_rows, key=lambda row: int(row["error_pixels"]), reverse=True)
    _print_rows("Worst events by damaged error rate", by_rate[:top_k])
    _print_rows("Largest contributors by damaged error pixels", by_pixels[:top_k])

    grouped: Dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["disaster_type"])].append(row)
    type_rows = []
    for name, values in grouped.items():
        summary = aggregate_rows(values)
        summary["event"] = name
        summary["disaster_type"] = name
        summary["error_pixels"] = round(
            summary["gt_damaged_pixels"]
            * (100.0 - summary["pred_damaged_pct"])
            / 100.0
        )
        summary["error_rate_pct"] = 100.0 - summary["pred_damaged_pct"]
        type_rows.append(summary)
    _print_rows(
        "Disaster types by damaged error rate",
        sorted(type_rows, key=lambda row: float(row["error_rate_pct"]), reverse=True),
    )

    output = Path(args.output_tsv) if args.output_tsv else source.with_suffix(".damaged_errors.tsv")
    write_tsv(output, sorted(rows, key=lambda row: str(row["event"])))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
