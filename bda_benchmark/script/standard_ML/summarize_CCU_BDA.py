# -*- coding: utf-8 -*-
"""Summarize CCU variants and event-wise gains without rerunning inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping


METRICS = (
    "valid_miou",
    "miou4",
    "damaged_iou",
    "destroyed_iou",
    "loc_f1",
    "event_mean_official_style_4class_miou",
    "event_worst_official_style_4class_miou",
)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize CCU-BDA independent evaluations")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--variants", nargs="+", default=["c0", "c1", "c2", "c3", "c4"])
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    records: Dict[str, Dict[str, object]] = {}
    raw: Dict[str, Mapping[str, object]] = {}
    for variant in args.variants:
        directory = root / f"CCU_{variant.upper()}_pilot{args.steps}_seed{args.seed}"
        path = directory / f"independent_{args.split}.json"
        if not path.is_file():
            records[variant] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw[variant] = payload
        metrics = payload["observed_metrics"]
        records[variant] = {
            "status": "ok",
            "path": str(path),
            "checkpoint_step": payload["checkpoint"]["step"],
            **{key: metrics[key] for key in METRICS},
        }

    baseline = raw.get("c0")
    event_rows = []
    if baseline is not None:
        baseline_events = baseline.get("event_metrics", {})
        for variant, payload in raw.items():
            if variant == "c0":
                continue
            positive = 0
            comparable = 0
            for event, current in payload.get("event_metrics", {}).items():
                if event not in baseline_events:
                    continue
                delta = float(current["miou4"]) - float(
                    baseline_events[event]["miou4"]
                )
                comparable += 1
                positive += int(delta >= 0.0)
                event_rows.append(
                    {
                        "variant": variant,
                        "event": event,
                        "disaster_type": current["disaster_type"],
                        "c0_miou4": float(baseline_events[event]["miou4"]),
                        "variant_miou4": float(current["miou4"]),
                        "delta_pp": 100.0 * delta,
                    }
                )
            records[variant]["nonnegative_events"] = positive
            records[variant]["comparable_events"] = comparable

    print(
        f"{'Variant':<8} {'Step':>6} {'Valid':>9} {'mIoU4':>9} "
        f"{'Dmg':>9} {'Des':>9} {'Event':>9} {'Worst':>9} {'Events+':>9}"
    )
    print("-" * 90)
    for variant in args.variants:
        record = records[variant]
        if record["status"] != "ok":
            print(f"{variant:<8} MISSING")
            continue
        print(
            f"{variant:<8} {int(record['checkpoint_step']):6d} "
            f"{100*float(record['valid_miou']):9.4f} "
            f"{100*float(record['miou4']):9.4f} "
            f"{100*float(record['damaged_iou']):9.4f} "
            f"{100*float(record['destroyed_iou']):9.4f} "
            f"{100*float(record['event_mean_official_style_4class_miou']):9.4f} "
            f"{100*float(record['event_worst_official_style_4class_miou']):9.4f} "
            f"{record.get('nonnegative_events', '-')!s:>9}"
        )

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": "CCU_BDA_comparison_v1",
                "split": args.split,
                "records": records,
                "event_deltas": event_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tsv = output.with_suffix(".events.tsv")
    with tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "event",
                "disaster_type",
                "c0_miou4",
                "variant_miou4",
                "delta_pp",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(event_rows)
    print("Saved:", output)
    print("Saved:", tsv)


if __name__ == "__main__":
    main()
