#!/usr/bin/env python3
"""Cross-seed confirmation for preregistered SPI T0/T1 paired audits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence


METRICS = (
    "valid_miou_pp",
    "damaged_f1_pp",
    "building_macro_f1_pp",
    "intact_iou_pp",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Summarize SPI T0/T1 raw paired audits")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum_valid_miou_gain_pp", type=float, default=0.5)
    parser.add_argument("--minimum_damaged_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_building_macro_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--maximum_intact_iou_drop_pp", type=float, default=0.5)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    args = parser.parse_args(argv)
    if len(args.inputs) < 3:
        raise ValueError("cross-seed confirmation requires at least three paired audits")

    payloads = []
    seen_seeds = set()
    base_hash = None
    steps = None
    lambda_cb = None
    for path in args.inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("protocol") != "SPI_T0_T1_RAW_PAIRED_DUAL_GT_AUDIT_v2":
            raise RuntimeError(f"unexpected audit protocol: {path}")
        if bool(payload.get("test_accessed", True)):
            raise RuntimeError(f"audit does not certify test isolation: {path}")
        manifest = payload["pair_manifest"]
        seed = int(manifest["seed"])
        if seed in seen_seeds:
            raise RuntimeError(f"duplicate seed {seed}")
        seen_seeds.add(seed)
        identity = (
            manifest["base_checkpoint_sha256"],
            int(manifest["optimizer_steps"]),
            float(manifest["t1_lambda_cb"]),
        )
        if base_hash is None:
            base_hash, steps, lambda_cb = identity
        elif identity != (base_hash, steps, lambda_cb):
            raise RuntimeError("cross-seed audits do not share base/steps/lambda_cb")
        payloads.append(payload)

    gains = [payload["decision"]["gain_pp"] for payload in payloads]
    summaries = {
        metric: {
            "mean": mean(float(row[metric]) for row in gains),
            "median": median(float(row[metric]) for row in gains),
            "values": [float(row[metric]) for row in gains],
        }
        for metric in METRICS
    }
    majority_required = math.floor(len(payloads) / 2) + 1
    positive_seed_count = sum(
        float(row["valid_miou_pp"]) > 0.0
        and float(row["damaged_f1_pp"]) > 0.0
        and float(row["building_macro_f1_pp"]) > 0.0
        for row in gains
    )
    event_names = sorted(
        set.intersection(
            *[
                {str(row["event"]) for row in payload["events"]}
                for payload in payloads
            ]
        )
    )
    nonnegative_events = 0
    event_mean_gains = {}
    for event in event_names:
        values = []
        for payload in payloads:
            row = next(item for item in payload["events"] if item["event"] == event)
            values.append(float(row["gain_pp"]["building_macro_f1_pp"]))
        value = mean(values)
        event_mean_gains[event] = value
        nonnegative_events += value >= 0.0

    checks = {
        "mean_valid_miou_gain": summaries["valid_miou_pp"]["mean"] >= args.minimum_valid_miou_gain_pp,
        "mean_damaged_f1_gain": summaries["damaged_f1_pp"]["mean"] >= args.minimum_damaged_f1_gain_pp,
        "mean_building_macro_f1_gain": summaries["building_macro_f1_pp"]["mean"] >= args.minimum_building_macro_f1_gain_pp,
        "majority_seeds_positive_on_all_primary_metrics": positive_seed_count >= majority_required,
        "every_seed_preserves_intact": all(
            float(row["intact_iou_pp"]) >= -args.maximum_intact_iou_drop_pp
            for row in gains
        ),
        "every_seed_t1_not_worse_than_locked_source": all(
            bool(payload["decision"]["checks"]["t1_not_worse_than_locked_source"])
            for payload in payloads
        ),
        "mean_event_nonnegative_count": nonnegative_events >= args.minimum_nonnegative_events,
    }
    passed = all(checks.values())
    output = {
        "protocol": "SPI_T0_T1_CROSS_SEED_DUAL_GT_CONFIRMATION_v2",
        "test_accessed": False,
        "seeds": sorted(seen_seeds),
        "base_checkpoint_sha256": base_hash,
        "optimizer_steps": steps,
        "t1_lambda_cb": lambda_cb,
        "summaries_pp": summaries,
        "positive_seed_count": positive_seed_count,
        "majority_required": majority_required,
        "event_mean_building_macro_f1_gain_pp": event_mean_gains,
        "nonnegative_events": nonnegative_events,
        "checks": checks,
        "passed": passed,
        "decision": (
            "PROCEED_TO_T2_PSEUDO_GEOMETRY_LEARNING"
            if passed
            else "STOP_AT_T1_COMPONENT_BALANCING_INSUFFICIENT"
        ),
        "sources": [str(Path(path).resolve()) for path in args.inputs],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 96)
    print("SPI T0/T1 cross-seed confirmation")
    for metric in METRICS:
        row = summaries[metric]
        print(f"{metric:<30} mean={row['mean']:+.4f} median={row['median']:+.4f} values={row['values']}")
    print(f"Positive seeds: {positive_seed_count}/{len(payloads)} (required {majority_required})")
    print(f"Checks        : {checks}")
    print(f"Decision      : {output['decision']}")
    print(f"Saved         : {output_path}")
    print("Test accessed : False")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
