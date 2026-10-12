#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_row(summary: dict) -> dict | None:
    admitted = summary.get("best_admitted")
    if admitted:
        return admitted
    rows = [row for row in summary.get("history", []) if row.get("step", 0) > 0]
    return max(
        rows,
        key=lambda row: row.get("metrics", {}).get("valid_miou", -1.0),
        default=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser("summarize sequential P3-HR screens")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--memory-result", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = {}
    for variant in ("hr0", "hr1", "hr1o", "hr1s", "hr2", "hr3o"):
        directory = (
            root / f"P3_HR0_exact_guard_seed{args.seed}"
            if variant == "hr0"
            else root / f"P3_{variant.upper()}_screen500_seed{args.seed}"
        )
        path = directory / "training_summary.json"
        if not path.is_file():
            rows[variant] = {"status": "missing"}
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        selected = best_row(summary)
        baseline = summary.get("baseline", {})
        metrics = selected.get("metrics", {}) if selected else baseline
        rows[variant] = {
            "status": "complete",
            "decision": summary.get("decision"),
            "selected_step": selected.get("step", 0) if selected else 0,
            "valid_miou": metrics.get("valid_miou"),
            "gain_vs_p3_pp": (
                100.0 * (metrics["valid_miou"] - baseline["valid_miou"])
                if "valid_miou" in metrics and "valid_miou" in baseline
                else None
            ),
            "damaged_iou": metrics.get("damaged_iou"),
            "destroyed_iou": metrics.get("destroyed_iou"),
            "affected_recall": metrics.get("affected_recall"),
            "nonnegative_events": (
                selected.get("diagnostics", {}).get("nonnegative_events")
                if selected else None
            ),
            "test_accessed": summary.get("test_accessed"),
        }

    memory_path = Path(args.memory_result)
    memory = (
        json.loads(memory_path.read_text(encoding="utf-8"))
        if memory_path.is_file()
        else {"status": "missing"}
    )
    output = {
        "protocol": "P3_HR_MEMORY_NIGHT_QUEUE_v1",
        "seed": args.seed,
        "hierarchical_variants": rows,
        "memory": {
            "decision": memory.get("decision"),
            "checks": memory.get("checks"),
            "source_variant": memory.get("source_variant"),
        },
        "test_accessed": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("P3-HR overnight summary")
    for name, row in rows.items():
        if row["status"] != "complete":
            print(f"{name:5s} missing")
            continue
        gain = row["gain_vs_p3_pp"]
        gain_text = "n/a" if gain is None else f"{gain:+.4f}pp"
        print(f"{name:5s} {gain_text:>10s}  {row['decision']}")
    print("Memory:", output["memory"]["decision"])
    print("Test accessed: False")
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
