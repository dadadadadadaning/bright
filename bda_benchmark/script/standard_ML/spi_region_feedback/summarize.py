#!/usr/bin/env python3
"""Compact, human-readable summary of a region-feedback JSON audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def bin_label(value: float, edges: Sequence[float]) -> str:
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return f">={edges[-1]:g}"


def change_sum(rows, change_key: str):
    output = {"regions": 0, "changed": 0, "corrected": 0, "harmed": 0, "net": 0}
    for row in rows:
        change = row[change_key]
        output["regions"] += 1
        output["changed"] += int(change["changed_pixels"])
        output["corrected"] += int(change["corrected_pixels"])
        output["harmed"] += int(change["harmed_pixels"])
        output["net"] += int(change["net_gain_pixels"])
    return output


def aggregate_bins(rows, field: str, edges: Sequence[float], change_key: str):
    output = {}
    for row in rows:
        label = bin_label(float(row[field]), edges)
        bucket = output.setdefault(
            label,
            {"regions": 0, "changed": 0, "corrected": 0, "harmed": 0, "net": 0},
        )
        change = row[change_key]
        bucket["regions"] += 1
        bucket["changed"] += int(change["changed_pixels"])
        bucket["corrected"] += int(change["corrected_pixels"])
        bucket["harmed"] += int(change["harmed_pixels"])
        bucket["net"] += int(change["net_gain_pixels"])
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Summarize SPI region feedback")
    parser.add_argument("json_path")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args(argv)
    source = Path(args.json_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    reports = data["reports"]
    baseline = reports["a0_raw_p3"]

    print("=" * 112)
    print("Region feedback summary | all gains are versus locked P3")
    print(f"{'Variant':<28} {'mIoU':>9} {'gain pp':>9} {'D-F1':>9} {'I-IoU':>9} {'D-IoU':>9} {'X-IoU':>9}")
    for name, report in reports.items():
        print(
            f"{name:<28} {100*report['valid_miou']:9.4f} "
            f"{100*(report['valid_miou']-baseline['valid_miou']):+9.4f} "
            f"{100*report['damaged_f1']:9.4f} {100*report['intact_iou']:9.4f} "
            f"{100*report['damaged_iou']:9.4f} {100*report['destroyed_iou']:9.4f}"
        )

    print("\nChange audit")
    for name, row in data.get("change_audit_vs_a0", {}).items():
        print(
            f"{name:<28} changed={row['changed_pixels']:>10d} "
            f"corrected={row['corrected_pixels']:>10d} "
            f"harmed={row['harmed_pixels']:>10d} net={row['net_gain_pixels']:+10d}"
        )

    print("\nEvent consistency")
    for name, row in data.get("event_win_counts_vs_a0", {}).items():
        print(
            f"{name:<28} nonnegative={row['nonnegative']}/{row['total_events']} "
            f"strictly_improved={row['improved']}/{row['total_events']}"
        )
    event_rows = []
    for event, table in data.get("event_reports", {}).items():
        raw = table["a0_raw_p3"]
        event_rows.append(
            (
                event,
                100.0 * (table["a1_majority"]["valid_miou"] - raw["valid_miou"]),
                100.0 * (table["a4_gated_majority"]["valid_miou"] - raw["valid_miou"]),
                100.0
                * (table["a5_gated_core_majority"]["valid_miou"] - raw["valid_miou"]),
            )
        )
    print("\nPer-event mIoU gain (pp)")
    print(f"{'Event':<32} {'A1-A0':>10} {'A4-A0':>10} {'A5-A0':>10}")
    for event, a1, a4, a5 in sorted(event_rows, key=lambda item: item[2]):
        print(f"{event:<32} {a1:+10.4f} {a4:+10.4f} {a5:+10.4f}")

    bootstrap = data.get("paired_bootstrap") or {}
    print("\nPaired bootstrap")
    for name, row in bootstrap.get("deltas_vs_a0", {}).items():
        print(
            f"{name:<28} mean={row['mean']:+.4f}pp "
            f"CI95=[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}]"
        )
    for name, row in bootstrap.get("comparisons", {}).items():
        print(
            f"{name:<58} mean={row['mean']:+.4f}pp "
            f"CI95=[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}]"
        )

    region_rows = data.get("region_audit", [])
    accepted = sum(bool(row["gate_accepted"]) for row in region_rows)
    print(
        f"\nGate accepted regions: {accepted}/{len(region_rows)} "
        f"({100*accepted/max(len(region_rows), 1):.2f}%)"
    )
    rejection_groups = {
        "purity_fail": [row for row in region_rows if not row["purity_pass"]],
        "margin_fail": [row for row in region_rows if not row["margin_pass"]],
        "area_fail": [row for row in region_rows if not row["area_pass"]],
    }
    print("\nA1 contribution removed by each rejection condition (groups may overlap)")
    rejection_summary = {}
    for reason, selected_rows in rejection_groups.items():
        row = change_sum(selected_rows, "a1_change")
        rejection_summary[reason] = row
        print(
            f"{reason:<16} regions={row['regions']:>6d} changed={row['changed']:>9d} "
            f"corrected={row['corrected']:>9d} harmed={row['harmed']:>9d} net={row['net']:+9d}"
        )
    bins = {
        "area": aggregate_bins(
            region_rows,
            "area",
            (0, 64, 256, 1024, 4096, 16384, 65536, float("inf")),
            "a1_change",
        ),
        "purity": aggregate_bins(
            region_rows,
            "purity",
            (0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01),
            "a1_change",
        ),
        "margin": aggregate_bins(
            region_rows,
            "margin",
            (float("-inf"), 0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01),
            "a1_change",
        ),
    }
    for field, table in bins.items():
        print(f"\nA1 net gain grouped by {field}")
        for label, row in table.items():
            print(
                f"  {label:<16} regions={row['regions']:>6d} changed={row['changed']:>9d} "
                f"corrected={row['corrected']:>9d} harmed={row['harmed']:>9d} net={row['net']:+9d}"
            )

    summary = {
        "source": str(source.resolve()),
        "p3_reproduction": data.get("p3_reproduction"),
        "event_win_counts_vs_a0": data.get("event_win_counts_vs_a0"),
        "change_audit_vs_a0": data.get("change_audit_vs_a0"),
        "gate_accepted_regions": accepted,
        "total_regions": len(region_rows),
        "a1_diagnostic_bins": bins,
        "a1_contribution_removed_by_gate_reason": rejection_summary,
        "per_event_miou_gain_pp": {
            event: {"a1": a1, "a4": a4, "a5": a5}
            for event, a1, a4, a5 in event_rows
        },
    }
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved compact summary: {target}")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
