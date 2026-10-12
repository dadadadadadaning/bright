#!/usr/bin/env python3
"""Compute a reproducible MaskDINO query budget from training-set counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.maskdino import compute_query_budget


def _read_counts(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if isinstance(payload, Mapping):
        payload = payload.get("instance_counts")
    if not isinstance(payload, list):
        raise TypeError(
            "counts JSON must be a list or an object containing instance_counts"
        )
    return [int(value) for value in payload]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute MaskDINO num_queries from training instance counts"
    )
    parser.add_argument("--counts_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--headroom", type=float, default=1.25)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--maximum", type=int, default=500)
    parser.add_argument("--round_to", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.counts_json).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"counts JSON not found: {input_path}")
    report = compute_query_budget(
        _read_counts(input_path),
        percentile=args.percentile,
        headroom=args.headroom,
        minimum=args.minimum,
        maximum=args.maximum,
        round_to=args.round_to,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
