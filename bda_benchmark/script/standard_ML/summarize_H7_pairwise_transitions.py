# -*- coding: utf-8 -*-
"""Summarise deployable P3-to-A2 disagreement directions from H7 JSON."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple


KNOWN_TYPES = (
    "earthquake",
    "wildfire",
    "hurricane",
    "volcano",
    "explosion",
    "flood",
    "conflict",
)


def disaster_type(event: str) -> str:
    value = event.lower().replace("_", "-")
    for name in KNOWN_TYPES:
        if value.endswith("-" + name):
            return name
    return "other"


def _pair(row: Mapping[str, object]) -> Tuple[str, str]:
    return str(row["p3_class"]), str(row["a2_class"])


def aggregate(records: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    values: MutableMapping[Tuple[str, str], MutableMapping[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in records:
        bucket = values[_pair(row)]
        for key in ("count", "p3_correct", "a2_correct", "neither_correct"):
            bucket[key] += int(row.get(key, 0))
    output = []
    for (p3_class, a2_class), counts in values.items():
        p3_correct = int(counts["p3_correct"])
        a2_correct = int(counts["a2_correct"])
        decisive = p3_correct + a2_correct
        output.append(
            {
                "p3_class": p3_class,
                "a2_class": a2_class,
                "count": int(counts["count"]),
                "p3_correct": p3_correct,
                "a2_correct": a2_correct,
                "neither_correct": int(counts["neither_correct"]),
                "net_a2_correct": a2_correct - p3_correct,
                "a2_win_rate_when_one_expert_correct": a2_correct
                / max(decisive, 1),
                "safe_candidate": a2_correct > p3_correct and decisive >= 100,
            }
        )
    return sorted(output, key=lambda row: int(row["net_a2_correct"]), reverse=True)


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "scope",
        "p3_class",
        "a2_class",
        "count",
        "p3_correct",
        "a2_correct",
        "neither_correct",
        "net_a2_correct",
        "a2_win_rate_when_one_expert_correct",
        "safe_candidate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H7 pairwise expert transitions")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    analysis = source.get("expert_transition_analysis")
    if not isinstance(analysis, Mapping) or not isinstance(analysis.get("records"), list):
        raise RuntimeError(
            "Oracle JSON lacks pairwise transitions; rerun the updated oracle val mode"
        )
    overall = aggregate(analysis["records"])
    events = source.get("events", [])
    if not isinstance(events, list):
        raise RuntimeError("Oracle JSON contains invalid event records")

    event_rows = []
    type_records: MutableMapping[str, list[Mapping[str, object]]] = defaultdict(list)
    for event in events:
        records = event.get("expert_transition_records")
        if not isinstance(records, list):
            raise RuntimeError(
                "Oracle event records lack pairwise transitions; rerun oracle val"
            )
        name = str(event["event"])
        for row in aggregate(records):
            event_rows.append({"scope": name, **row})
        type_records[disaster_type(name)].extend(records)

    disaster_rows = []
    for name, records in sorted(type_records.items()):
        for row in aggregate(records):
            disaster_rows.append({"scope": name, **row})
    flat_rows = (
        [{"scope": "overall", **row} for row in overall]
        + disaster_rows
        + event_rows
    )
    payload = {
        "protocol": "H7_pairwise_transition_report_v1",
        "source_json": str(Path(args.input_json).resolve()),
        "split": "val",
        "test_accessed": False,
        "overall": overall,
        "by_disaster_type": disaster_rows,
        "by_event": event_rows,
        "decision": (
            "DESIGN_PAIR_CONDITIONAL_SELECTOR"
            if any(bool(row["safe_candidate"]) for row in overall)
            else "NO_RAW_PAIR_IS_SAFE_UPGRADE_SELECTOR_FEATURES"
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tsv = Path(args.output_tsv)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(tsv, flat_rows)

    print(
        f"{'P3':<12} {'A2':<12} {'Count':>12} {'A2 ok':>12} "
        f"{'P3 ok':>12} {'Net A2':>12} {'Win%':>9} {'Candidate':>10}"
    )
    print("-" * 102)
    for row in overall:
        print(
            f"{row['p3_class']:<12} {row['a2_class']:<12} "
            f"{int(row['count']):12,d} {int(row['a2_correct']):12,d} "
            f"{int(row['p3_correct']):12,d} {int(row['net_a2_correct']):+12,d} "
            f"{100*float(row['a2_win_rate_when_one_expert_correct']):8.2f}% "
            f"{str(bool(row['safe_candidate'])):>10}"
        )
    print(f"Decision: {payload['decision']}")
    print(f"Saved JSON: {output}")
    print(f"Saved TSV: {tsv}")


if __name__ == "__main__":
    main()
