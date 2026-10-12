#!/usr/bin/env python3
"""Summarize the frozen-P3 main line and add the missing S2-S1 comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    stratified_paired_bootstrap,
)


def _pp(value: float) -> float:
    return 100.0 * float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selective-json", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    data = json.loads(args.selective_json.read_text(encoding="utf-8"))
    if data.get("official_instances_accessed") is not False:
        raise RuntimeError("selective result is not marked instance-free")
    if data.get("original_standard_test_accessed") is not False:
        raise RuntimeError("selective result is not marked test-free")
    reproduction = data.get("p3_reproduction", {})
    if reproduction.get("passed") is not True:
        raise RuntimeError("P3 reproduction did not pass")

    reports = data["reports"]
    baseline = "s0_raw_p3"
    variants = tuple(reports)
    rows = data.get("image_histograms", [])
    if not rows:
        raise RuntimeError("image_histograms are required for paired bootstrap")
    bootstrap = stratified_paired_bootstrap(
        rows,
        variants=variants,
        baseline=baseline,
        comparison_pairs=(("s2_loccc_soft_context", "s1_loccc_hard_a6"),),
        iterations=args.iterations,
        seed=args.seed,
    )
    soft_minus_hard = bootstrap["comparisons"][
        "s2_loccc_soft_context_minus_s1_loccc_hard_a6"
    ]

    summary: dict[str, object] = {
        "protocol": "SPI_PSEUDO_INSTANCE_MAINLINE_SUMMARY_V1",
        "selective_source": str(args.selective_json),
        "p3_reproduction": reproduction,
        "variants": {},
        "soft_minus_hard_bootstrap": soft_minus_hard,
        "change_audit": data.get("change_audit"),
        "event_wins": data.get("event_wins"),
        "official_instances_accessed": False,
        "original_standard_test_accessed": False,
    }
    base_miou = reports[baseline]["valid_miou"]
    for name, report in reports.items():
        summary["variants"][name] = {
            "valid_miou_percent": _pp(report["valid_miou"]),
            "gain_vs_p3_pp": _pp(report["valid_miou"] - base_miou),
            "damaged_f1_percent": _pp(report["damaged_f1"]),
            "intact_iou_percent": _pp(report["intact_iou"]),
            "damaged_iou_percent": _pp(report["damaged_iou"]),
            "destroyed_iou_percent": _pp(report["destroyed_iou"]),
        }

    if args.geometry_json:
        geometry = json.loads(args.geometry_json.read_text(encoding="utf-8"))
        summary["geometry_source"] = str(args.geometry_json)
        summary["geometry_reports"] = geometry.get("reports")
        summary["rgid_geometry_decision"] = geometry.get("decision", {}).get(
            "rgid_geometry"
        )
        summary["rgid_decode_totals"] = geometry.get("rgid_decode_totals")

    print("Pseudo-instance semantic main line")
    print(f"{'Variant':36s} {'mIoU':>9s} {'gain':>10s} {'D-F1':>9s}")
    for name, report in summary["variants"].items():
        print(
            f"{name:36s} {report['valid_miou_percent']:9.4f} "
            f"{report['gain_vs_p3_pp']:+10.4f} "
            f"{report['damaged_f1_percent']:9.4f}"
        )
    print(f"S2-S1 paired bootstrap: {soft_minus_hard}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

