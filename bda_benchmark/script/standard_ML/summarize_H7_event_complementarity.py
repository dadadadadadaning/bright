# -*- coding: utf-8 -*-
"""Create event- and disaster-type reports from an H7 oracle audit JSON."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    CLASS_NAMES,
    metrics_from_histogram,
    rates,
    transition_records,
)


KNOWN_DISASTER_TYPES = (
    "earthquake",
    "wildfire",
    "hurricane",
    "volcano",
    "explosion",
    "flood",
    "conflict",
)


def disaster_type(event: str) -> str:
    lowered = event.lower().replace("_", "-")
    for name in KNOWN_DISASTER_TYPES:
        if lowered.endswith(f"-{name}") or lowered == name:
            return name
    return "other"


def _metric(row: Mapping[str, object], expert: str, key: str) -> float:
    value = row[expert]
    if not isinstance(value, Mapping):
        raise TypeError(f"event row has invalid {expert} metrics")
    return float(value[key])


def _counts(row: Mapping[str, object]) -> Mapping[str, int]:
    value = row["counts"]
    if not isinstance(value, Mapping):
        raise TypeError("event row has invalid complementarity counts")
    return value  # type: ignore[return-value]


def event_report_rows(events: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    rows = []
    for event in events:
        counts = _counts(event)
        p3_valid = _metric(event, "p3", "valid_miou")
        a2_valid = _metric(event, "a2", "valid_miou")
        oracle_valid = _metric(event, "oracle", "valid_miou")
        a2_only = int(counts.get("a2_only_correct", 0))
        p3_only = int(counts.get("p3_only_correct", 0))
        rows.append(
            {
                "event": str(event["event"]),
                "disaster_type": disaster_type(str(event["event"])),
                "p3_valid_miou": p3_valid,
                "a2_valid_miou": a2_valid,
                "oracle_valid_miou": oracle_valid,
                "a2_minus_p3_pp": 100.0 * (a2_valid - p3_valid),
                "oracle_minus_p3_pp": 100.0 * (oracle_valid - p3_valid),
                "p3_damaged_iou": _metric(event, "p3", "damaged_iou"),
                "a2_damaged_iou": _metric(event, "a2", "damaged_iou"),
                "oracle_damaged_iou": _metric(event, "oracle", "damaged_iou"),
                "a2_only_correct": a2_only,
                "p3_only_correct": p3_only,
                "net_a2_correct": a2_only - p3_only,
                "disagreement": int(counts.get("disagreement", 0)),
                "a2_standalone_wins": a2_valid > p3_valid,
            }
        )
    return sorted(rows, key=lambda value: float(value["oracle_minus_p3_pp"]), reverse=True)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def _sum_confusions(
    events_by_type: Sequence[Mapping[str, object]], expert: str
) -> torch.Tensor | None:
    total = torch.zeros(4, 4, dtype=torch.int64)
    for row in events_by_type:
        confusion = row.get("confusion")
        if not isinstance(confusion, Mapping) or expert not in confusion:
            return None
        total += torch.as_tensor(confusion[expert], dtype=torch.int64)
    return total


def _sum_transition_cube(events_by_type: Sequence[Mapping[str, object]]) -> torch.Tensor | None:
    total = torch.zeros(4, 4, 4, dtype=torch.int64)
    for row in events_by_type:
        value = row.get("expert_transition_cube")
        if value is None:
            return None
        total += torch.as_tensor(value, dtype=torch.int64)
    return total


def disaster_report_rows(events: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    grouped: MutableMapping[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in events:
        grouped[disaster_type(str(row["event"]))].append(row)

    report = []
    for type_name, source_rows in sorted(grouped.items()):
        simple_rows = event_report_rows(source_rows)
        counts: MutableMapping[str, int] = defaultdict(int)
        for row in source_rows:
            for key, value in _counts(row).items():
                counts[key] += int(value)
        entry: Dict[str, object] = {
            "disaster_type": type_name,
            "event_count": len(source_rows),
            "events": sorted(str(row["event"]) for row in source_rows),
            "macro": {
                key: _mean(simple_rows, key)
                for key in (
                    "p3_valid_miou",
                    "a2_valid_miou",
                    "oracle_valid_miou",
                    "a2_minus_p3_pp",
                    "oracle_minus_p3_pp",
                    "p3_damaged_iou",
                    "a2_damaged_iou",
                    "oracle_damaged_iou",
                )
            },
            "counts": dict(counts),
            "rates": rates(counts),
            "a2_standalone_win_events": sum(
                int(bool(row["a2_standalone_wins"])) for row in simple_rows
            ),
            "net_a2_correct": int(counts.get("a2_only_correct", 0))
            - int(counts.get("p3_only_correct", 0)),
        }
        exact_metrics = {}
        for expert in ("p3", "a2", "oracle"):
            confusion = _sum_confusions(source_rows, expert)
            if confusion is not None:
                exact_metrics[expert] = metrics_from_histogram(confusion)
        if exact_metrics:
            entry["exact_aggregated_metrics"] = exact_metrics
        transition_cube = _sum_transition_cube(source_rows)
        if transition_cube is not None:
            entry["expert_transition_cube"] = transition_cube.tolist()
            entry["expert_transition_records"] = transition_records(transition_cube)
        report.append(entry)
    return sorted(
        report,
        key=lambda value: float(value["macro"]["oracle_minus_p3_pp"]),  # type: ignore[index]
        reverse=True,
    )


def write_event_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "event",
        "disaster_type",
        "p3_valid_miou",
        "a2_valid_miou",
        "oracle_valid_miou",
        "a2_minus_p3_pp",
        "oracle_minus_p3_pp",
        "p3_damaged_iou",
        "a2_damaged_iou",
        "oracle_damaged_iou",
        "a2_only_correct",
        "p3_only_correct",
        "net_a2_correct",
        "disagreement",
        "a2_standalone_wins",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_disaster_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "disaster_type",
        "event_count",
        "events",
        "p3_valid_macro",
        "a2_valid_macro",
        "oracle_valid_macro",
        "a2_minus_p3_pp_macro",
        "oracle_minus_p3_pp_macro",
        "a2_only_correct",
        "p3_only_correct",
        "net_a2_correct",
        "a2_standalone_win_events",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            macro = row["macro"]
            counts = row["counts"]
            assert isinstance(macro, Mapping) and isinstance(counts, Mapping)
            writer.writerow(
                {
                    "disaster_type": row["disaster_type"],
                    "event_count": row["event_count"],
                    "events": ",".join(row["events"]),
                    "p3_valid_macro": macro["p3_valid_miou"],
                    "a2_valid_macro": macro["a2_valid_miou"],
                    "oracle_valid_macro": macro["oracle_valid_miou"],
                    "a2_minus_p3_pp_macro": macro["a2_minus_p3_pp"],
                    "oracle_minus_p3_pp_macro": macro["oracle_minus_p3_pp"],
                    "a2_only_correct": counts.get("a2_only_correct", 0),
                    "p3_only_correct": counts.get("p3_only_correct", 0),
                    "net_a2_correct": row["net_a2_correct"],
                    "a2_standalone_win_events": row["a2_standalone_win_events"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H7 complementarity by event and disaster")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_event_tsv", required=True)
    parser.add_argument("--output_disaster_tsv", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if source.get("protocol") != "H7_P3_A2_validation_oracle_v1":
        raise RuntimeError(f"unexpected input protocol: {source.get('protocol')!r}")
    if source.get("test_accessed") is not False:
        raise RuntimeError("H7 development report must be validation-only")
    events = source.get("events")
    if not isinstance(events, list) or not events:
        raise RuntimeError("oracle JSON contains no event records")

    event_rows = event_report_rows(events)
    disaster_rows = disaster_report_rows(events)
    payload = {
        "protocol": "H7_event_disaster_complementarity_v1",
        "source_json": str(Path(args.input_json).resolve()),
        "split": "val",
        "test_accessed": False,
        "event_rows": event_rows,
        "disaster_type_rows": disaster_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_event_tsv(Path(args.output_event_tsv), event_rows)
    write_disaster_tsv(Path(args.output_disaster_tsv), disaster_rows)

    print(
        f"{'Event':<28} {'Type':<12} {'P3':>8} {'A2':>8} "
        f"{'A2-P3':>8} {'Oracle':>8} {'O-P3':>8} {'NetPix':>11}"
    )
    print("-" * 105)
    for row in event_rows:
        print(
            f"{row['event']:<28} {row['disaster_type']:<12} "
            f"{100*float(row['p3_valid_miou']):8.3f} "
            f"{100*float(row['a2_valid_miou']):8.3f} "
            f"{float(row['a2_minus_p3_pp']):+8.3f} "
            f"{100*float(row['oracle_valid_miou']):8.3f} "
            f"{float(row['oracle_minus_p3_pp']):+8.3f} "
            f"{int(row['net_a2_correct']):+11d}"
        )
    print("\nDisaster-type macro comparison")
    print(
        f"{'Type':<12} {'N':>3} {'P3':>8} {'A2':>8} {'A2-P3':>8} "
        f"{'Oracle':>8} {'O-P3':>8} {'A2 wins':>8}"
    )
    print("-" * 78)
    for row in disaster_rows:
        macro = row["macro"]
        assert isinstance(macro, Mapping)
        print(
            f"{row['disaster_type']:<12} {int(row['event_count']):3d} "
            f"{100*float(macro['p3_valid_miou']):8.3f} "
            f"{100*float(macro['a2_valid_miou']):8.3f} "
            f"{float(macro['a2_minus_p3_pp']):+8.3f} "
            f"{100*float(macro['oracle_valid_miou']):8.3f} "
            f"{float(macro['oracle_minus_p3_pp']):+8.3f} "
            f"{int(row['a2_standalone_win_events']):8d}"
        )
    print(f"Saved JSON: {output}")
    print(f"Saved event TSV: {args.output_event_tsv}")
    print(f"Saved disaster TSV: {args.output_disaster_tsv}")


if __name__ == "__main__":
    main()
