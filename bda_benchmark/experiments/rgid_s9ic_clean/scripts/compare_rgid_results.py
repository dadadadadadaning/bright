#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence


CSV_FIELDS = (
    "Method",
    "Precision",
    "Recall",
    "F1",
    "Mean IoU",
    "Merge",
    "Split",
    "Pred instances",
    "Center candidates",
    "Accepted splits",
)


def _row(method: str, values: Mapping[str, object]) -> Dict[str, object]:
    return {
        "Method": method.upper(),
        "Precision": float(values.get("precision", 0.0)),
        "Recall": float(values.get("recall", 0.0)),
        "F1": float(values.get("f1", 0.0)),
        "Mean IoU": float(values.get("mean_matched_iou", 0.0)),
        "Merge": int(values.get("merge", 0)),
        "Split": int(values.get("split", 0)),
        "Pred instances": int(values.get("pred_instances", 0)),
        "Center candidates": int(values.get("center_candidates_total", 0)),
        "Accepted splits": int(values.get("split_accepted_regions", 0)),
    }


def _delta(
    left_name: str,
    right_name: str,
    methods: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    left = methods[left_name]
    right = methods[right_name]
    return {
        "comparison": f"{left_name.upper()} - {right_name.upper()}",
        "ΔPrecision": float(left.get("precision", 0.0))
        - float(right.get("precision", 0.0)),
        "ΔRecall": float(left.get("recall", 0.0))
        - float(right.get("recall", 0.0)),
        "ΔF1": float(left.get("f1", 0.0)) - float(right.get("f1", 0.0)),
        "ΔMeanIoU": float(left.get("mean_matched_iou", 0.0))
        - float(right.get("mean_matched_iou", 0.0)),
        "ΔMerge": int(left.get("merge", 0)) - int(right.get("merge", 0)),
        "ΔSplit": int(left.get("split", 0)) - int(right.get("split", 0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compare cached RGID G0/G1/G2/G3 results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_path = Path(args.results).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    with open(results_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    methods = {}
    for method in ("g0", "g1", "g2", "g3"):
        values = payload.get(method)
        if not isinstance(values, Mapping):
            raise KeyError(f"results JSON has no mapping for {method}")
        methods[method] = values
    rows = [_row(method, methods[method]) for method in ("g0", "g1", "g2", "g3")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(" | ".join(CSV_FIELDS), flush=True)
    for row in rows:
        print(
            " | ".join(
                f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key])
                for key in CSV_FIELDS
            ),
            flush=True,
        )
    deltas = [
        _delta("g1", "g0", methods),
        _delta("g2", "g1", methods),
        _delta("g3", "g2", methods),
        _delta("g3", "g0", methods),
    ]
    print("\nChanges", flush=True)
    for row in deltas:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"Saved CSV: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
