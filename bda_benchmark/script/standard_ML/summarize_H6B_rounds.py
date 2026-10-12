# -*- coding: utf-8 -*-
"""Summarize paired validation/test monitoring rounds for H6-B.

Test metrics are intentionally descriptive only.  The selected checkpoint is
always the one recorded by the validation selector in training_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


P3_TEST = {
    "miou4": 0.672725,
    "valid_miou": 0.574086,
    "damaged_iou": 0.400857,
    "destroyed_iou": 0.569040,
    "event_mean_official_style_4class_miou": 0.526139,
}


def pct(metrics: Mapping[str, object], key: str) -> float:
    return 100.0 * float(metrics.get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H6-B validation/test rounds")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    directory = Path(args.experiment_dir)
    history_path = directory / "test_monitor_history.json"
    summary_path = directory / "training_summary.json"
    if not history_path.is_file():
        raise SystemExit(f"missing test monitor history: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    selected_step = (
        summary.get("selection_records", {})
        .get("valid", {})
        .get("step")
    )

    header = (
        f"{'Step':>6} {'Selected':>8} | {'Val3C':>8} {'ValDmg':>8} "
        f"{'ValDes':>8} {'ValEvt':>8} | {'Test4C':>8} {'Test3C':>8} "
        f"{'TestDmg':>8} {'TestDes':>8} {'TestEvt':>8} {'dP3-4C':>8}"
    )
    print(header)
    print("-" * len(header))
    records = []
    for record in history:
        test = record.get("metrics", {})
        validation = record.get("validation_metrics", {})
        step = int(record.get("step", 0))
        selected = step == selected_step
        print(
            f"{step:6d} {('*' if selected else ''):>8} | "
            f"{pct(validation, 'valid_miou'):8.4f} "
            f"{pct(validation, 'damaged_iou'):8.4f} "
            f"{pct(validation, 'destroyed_iou'):8.4f} "
            f"{pct(validation, 'event_mean_official_style_4class_miou'):8.4f} | "
            f"{pct(test, 'miou4'):8.4f} "
            f"{pct(test, 'valid_miou'):8.4f} "
            f"{pct(test, 'damaged_iou'):8.4f} "
            f"{pct(test, 'destroyed_iou'):8.4f} "
            f"{pct(test, 'event_mean_official_style_4class_miou'):8.4f} "
            f"{100.0 * (float(test.get('miou4', 0.0)) - P3_TEST['miou4']):8.4f}"
        )
        records.append(
            {
                "step": step,
                "selected_by_validation": selected,
                "validation": validation,
                "test_monitor": test,
                "test_minus_p3": {
                    key: float(test.get(key, 0.0)) - value
                    for key, value in P3_TEST.items()
                },
            }
        )

    payload = {
        "protocol": "H6B_paired_round_summary_v1",
        "experiment_dir": str(directory.resolve()),
        "checkpoint_selector": "validation_only",
        "selected_step": selected_step,
        "p3_test_reference": P3_TEST,
        "rounds": records,
    }
    output = Path(args.output_json) if args.output_json else directory / "round_comparison.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
