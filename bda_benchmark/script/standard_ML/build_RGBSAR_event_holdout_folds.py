# -*- coding: utf-8 -*-
"""Build event-disjoint train/validation folds without touching the test set."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def disaster_type(event: str) -> str:
    return str(event).rsplit("-", 1)[-1]


def assign_event_folds(
    event_counts: Mapping[str, int], fold_count: int
) -> List[List[str]]:
    """Greedily balance image count while spreading repeated disaster types."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(event_counts) < fold_count:
        raise ValueError("fewer events than folds")
    folds: List[List[str]] = [[] for _ in range(fold_count)]
    fold_sizes = [0 for _ in range(fold_count)]
    fold_types: List[Counter[str]] = [Counter() for _ in range(fold_count)]
    ordered = sorted(event_counts, key=lambda key: (-event_counts[key], key))
    for event in ordered:
        kind = disaster_type(event)
        target = min(
            range(fold_count),
            key=lambda index: (
                fold_types[index][kind],
                fold_sizes[index],
                len(folds[index]),
                index,
            ),
        )
        folds[target].append(event)
        fold_sizes[target] += int(event_counts[event])
        fold_types[target][kind] += 1
    return [sorted(values) for values in folds]


def read_names(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Build event-disjoint RGB/SAR development folds"
    )
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("balanced", "leave_one_event_out", "leave_one_type_out"),
        default="balanced",
        help=(
            "balanced builds a small number of sample-balanced event-disjoint "
            "folds; leave_one_event_out holds out one complete event; "
            "leave_one_type_out holds out every event of one hazard type"
        ),
    )
    args = parser.parse_args()

    source = Path(args.train_list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = read_names(source)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        grouped[event_name(name)].append(name)
    if args.mode == "leave_one_event_out":
        assignments = [[event] for event in sorted(grouped)]
    elif args.mode == "leave_one_type_out":
        events_by_type: Dict[str, List[str]] = defaultdict(list)
        for event in sorted(grouped):
            events_by_type[disaster_type(event)].append(event)
        assignments = [events_by_type[kind] for kind in sorted(events_by_type)]
    else:
        assignments = assign_event_folds(
            {event: len(values) for event, values in grouped.items()}, args.folds
        )

    records: List[Dict[str, object]] = []
    all_names = set(names)
    for index, validation_events in enumerate(assignments):
        validation = sorted(
            name
            for event in validation_events
            for name in grouped[event]
        )
        training = sorted(all_names.difference(validation))
        train_path = output_dir / f"fold{index}_train.txt"
        val_path = output_dir / f"fold{index}_val.txt"
        write_lines(train_path, training)
        write_lines(val_path, validation)
        records.append(
            {
                "fold": index,
                "train_count": len(training),
                "validation_count": len(validation),
                "validation_events": validation_events,
                "validation_disaster_types": sorted(
                    {disaster_type(event) for event in validation_events}
                ),
                "train_list": str(train_path.resolve()),
                "validation_list": str(val_path.resolve()),
            }
        )

    manifest = {
        "protocol": "RGBSAR_event_disjoint_development_folds_v1",
        "mode": args.mode,
        "source_train_list": str(source.resolve()),
        "test_accessed": False,
        "requires_fold_specific_pretrained_sources": True,
        "leakage_warning": (
            "A checkpoint pretrained on the complete standard train split has "
            "already seen every held-out event and must not be used to claim "
            "event-disjoint generalization. Rebuild every supervised source "
            "checkpoint inside each fold."
        ),
        "event_count": len(grouped),
        "fold_count": args.folds,
        "folds": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(
            f"fold={record['fold']} train={record['train_count']} "
            f"val={record['validation_count']} events="
            f"{','.join(record['validation_events'])}"
        )
    print(f"Saved: {manifest_path}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
