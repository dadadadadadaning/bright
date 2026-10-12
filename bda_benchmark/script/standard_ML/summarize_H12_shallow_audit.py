# -*- coding: utf-8 -*-
"""Print the H12 shallow-evidence audit decision without re-running inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping


def rows_from_payload(payload: Mapping[str, object]) -> list[dict[str, object]]:
    if payload.get("test_accessed"):
        raise ValueError("refusing a test-accessed H12 audit")
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("audit JSON has no tasks")
    rows = []
    for task_name, probes in tasks.items():
        if not isinstance(probes, Mapping):
            continue
        for probe_name, record in probes.items():
            if not isinstance(record, Mapping):
                continue
            rows.append(
                {
                    "task": task_name,
                    "probe": probe_name,
                    "count": record.get("count", 0),
                    "positive_count": record.get("positive_count", 0),
                    "auroc": record.get("auroc"),
                    "auprc": record.get("auprc"),
                    "event_macro_auroc": record.get("event_macro_auroc"),
                    "event_worst_auroc": record.get("event_worst_auroc"),
                    "event_direction_consistent": record.get(
                        "event_direction_consistent"
                    ),
                    "comparable_events": record.get("comparable_events"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H12 shallow evidence audit")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_tsv", default="")
    args = parser.parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    rows = rows_from_payload(payload)
    decision = payload.get("decision", {})
    print(json.dumps(decision, indent=2), flush=True)
    if args.output_tsv:
        output = Path(args.output_tsv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else [
            "task",
            "probe",
            "auroc",
            "event_macro_auroc",
        ]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {output}", flush=True)
    recommendation = decision.get("recommendation", "UNKNOWN")
    print(f"Recommendation: {recommendation}", flush=True)


if __name__ == "__main__":
    main()
