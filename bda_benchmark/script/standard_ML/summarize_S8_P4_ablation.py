#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a unified P4 validation comparison from training summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


METRICS = (
    "miou4",
    "valid_miou",
    "loc_precision",
    "loc_recall",
    "loc_f1",
    "loc_iou",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_f1",
    "destroyed_f1",
    "damage_score",
    "balanced_score",
    "damaged_to_intact",
    "destroyed_to_intact",
    "event_mean_official_style_4class_miou",
)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize S8-P4 controlled variants")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mode_tag", default="pilot3000")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--variants", default="a0 a1 a2 a3 a4")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    rows: List[Dict[str, object]] = []
    variants = tuple(value.lower() for value in args.variants.split() if value)
    if not variants or variants[0] != "a0":
        raise ValueError("--variants must be non-empty and start with a0 baseline")
    for variant in variants:
        name = f"S8_P4_{variant.upper()}_{args.mode_tag}_seed{args.seed}"
        summary_path = root / name / "training_summary.json"
        row: Dict[str, object] = {
            "variant": variant,
            "experiment": name,
            "summary_path": str(summary_path),
            "status": "missing",
        }
        if summary_path.is_file():
            with open(summary_path, "r", encoding="utf-8") as file:
                summary = json.load(file)
            best = summary.get("best_metrics", {})
            row.update(
                {
                    "status": (
                        "early_stopped"
                        if summary.get("early_stopped")
                        else "completed"
                    ),
                    "completed_step": int(summary.get("completed_step", -1)),
                    "crops_seen": int(summary.get("crops_seen", -1)),
                }
            )
            for metric in METRICS:
                if metric in best:
                    row[metric] = float(best[metric])
        rows.append(row)

    baseline = rows[0]
    for row in rows:
        for metric in METRICS:
            if metric in row and metric in baseline:
                row[f"delta_{metric}_pp_vs_a0"] = (
                    100.0 * (float(row[metric]) - float(baseline[metric]))
                )

    output_json = Path(args.output_json)
    output_tsv = Path(args.output_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(
            {
                "protocol": "S8-P4_controlled_validation_ablation_v2",
                "mode_tag": args.mode_tag,
                "seed": args.seed,
                "rows": rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    columns = sorted({key for row in rows for key in row})
    with open(output_tsv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'Variant':<8}{'Status':<16}{'mIoU4':>10}{'Valid':>10}"
        f"{'Dmg':>10}{'Des':>10}{'LocF1':>10}{'Event':>10}"
    )
    for row in rows:
        values = [
            100.0 * float(row[key]) if key in row else float("nan")
            for key in (
                "miou4",
                "valid_miou",
                "damaged_iou",
                "destroyed_iou",
                "loc_f1",
                "event_mean_official_style_4class_miou",
            )
        ]
        print(
            f"{str(row['variant']).upper():<8}{str(row['status']):<16}"
            + "".join(f"{value:>10.4f}" for value in values)
        )
    print(f"Saved: {output_json}")
    print(f"Saved: {output_tsv}")


if __name__ == "__main__":
    main()
