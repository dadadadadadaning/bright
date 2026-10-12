# -*- coding: utf-8 -*-
"""Validation-only decision report for H6-A2 DG0--DG3 ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


METRIC_KEYS = (
    "valid_miou",
    "damaged_iou",
    "destroyed_iou",
    "event_mean_official_style_4class_miou",
    "event_worst_official_style_4class_miou",
)

DEFAULT_GATES_PP = {
    "valid_miou": -0.10,
    "damaged_iou": 0.30,
    "destroyed_iou": -0.20,
    "event_mean_official_style_4class_miou": 0.00,
    "event_worst_official_style_4class_miou": -0.10,
}

SELECTOR_FILES = {
    "valid": "best_val_miou.pth",
    "event": "best_event.pth",
    "balance": "best_balance.pth",
}


def metric_deltas_pp(
    metrics: Mapping[str, float], baseline: Mapping[str, float]
) -> Dict[str, float]:
    return {
        key: 100.0 * (float(metrics[key]) - float(baseline[key]))
        for key in METRIC_KEYS
    }


def passes_promotion_gate(
    deltas_pp: Mapping[str, float],
    gates_pp: Mapping[str, float] = DEFAULT_GATES_PP,
) -> bool:
    return all(float(deltas_pp[key]) >= float(threshold) for key, threshold in gates_pp.items())


def candidate_rank(candidate: Mapping[str, object]) -> Tuple[float, ...]:
    """Predeclared order: Damaged first, then robust event and safety metrics."""

    delta = candidate["delta_pp"]
    assert isinstance(delta, Mapping)
    return (
        float(delta["damaged_iou"]),
        float(delta["event_worst_official_style_4class_miou"]),
        float(delta["event_mean_official_style_4class_miou"]),
        float(delta["valid_miou"]),
        float(delta["destroyed_iou"]),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_metrics(path: Path) -> Dict[str, float]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
    missing = [key for key in METRIC_KEYS if key not in raw]
    if missing:
        raise RuntimeError(f"baseline checkpoint lacks metrics: {missing}")
    return {key: float(raw[key]) for key in METRIC_KEYS}


def experiment_name(index: int, steps: int, seed: int) -> str:
    labels = ("control", "focus", "event", "combined")
    return f"H6_A2_DG{index}_{labels[index]}{steps}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser("Summarize H6-A2 DG validation ablations")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    root = Path(args.output_root)
    baseline_path = Path(args.baseline_checkpoint)
    locked_baseline = load_checkpoint_metrics(baseline_path)
    records = []
    header = (
        f"{'Run':<5} {'Sel':<8} {'Step':>5} {'Valid':>8} {'Dmg':>8} "
        f"{'Des':>8} {'Event':>8} {'Worst':>8} {'dDmg':>8} {'Pass':>5}"
    )
    print(header)
    print("-" * len(header))

    for index in range(4):
        name = experiment_name(index, args.steps, args.seed)
        directory = root / name
        summary_path = directory / "training_summary.json"
        if not summary_path.is_file():
            print(f"DG{index:<3} MISSING {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if bool(summary.get("test_accessed", False)):
            raise SystemExit(f"test leakage detected in {summary_path}")
        selectors = summary.get("selection_records", {})
        for selector in ("valid", "event", "balance"):
            record = selectors.get(selector)
            if not isinstance(record, Mapping) or "metrics" not in record:
                continue
            metrics = {key: float(record["metrics"][key]) for key in METRIC_KEYS}
            checkpoint = directory / SELECTOR_FILES[selector]
            checkpoint_exists = checkpoint.is_file()
            candidate = {
                "run": f"DG{index}",
                "experiment": name,
                "experiment_dir": str(directory.resolve()),
                "selector": selector,
                "selected_step": int(record["step"]),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_exists": checkpoint_exists,
                "metrics": metrics,
                "delta_to_locked_h6_a2_pp": metric_deltas_pp(
                    metrics, locked_baseline
                ),
                "delta_pp": {},
                "passed_promotion_gate": False,
            }
            records.append(candidate)

    # Screening must compare equal training budgets and identical checkpoint
    # selectors.  The historical 3000-step H6-A2 checkpoint is recorded for
    # context only; it becomes the promotion baseline after the winning
    # protocol is rerun for 3000 steps.
    control_by_selector = {
        str(record["selector"]): record
        for record in records
        if record["run"] == "DG0"
    }
    for candidate in records:
        selector = str(candidate["selector"])
        control = control_by_selector.get(selector)
        if control is None:
            raise SystemExit(f"DG0 has no equal-selector control for {selector}")
        delta = metric_deltas_pp(candidate["metrics"], control["metrics"])
        passed = candidate["run"] != "DG0" and passes_promotion_gate(delta)
        if passed and not candidate["checkpoint_exists"]:
            raise SystemExit(
                f"passing candidate checkpoint is missing: {candidate['checkpoint']}"
            )
        candidate["screening_reference"] = {
            "run": "DG0",
            "selector": selector,
            "selected_step": control["selected_step"],
        }
        candidate["delta_pp"] = delta
        candidate["passed_promotion_gate"] = passed
        metrics = candidate["metrics"]
        print(
            f"{candidate['run']:<5} {selector:<8} "
            f"{int(candidate['selected_step']):5d} "
            f"{100*metrics['valid_miou']:8.4f} "
            f"{100*metrics['damaged_iou']:8.4f} "
            f"{100*metrics['destroyed_iou']:8.4f} "
            f"{100*metrics['event_mean_official_style_4class_miou']:8.4f} "
            f"{100*metrics['event_worst_official_style_4class_miou']:8.4f} "
            f"{delta['damaged_iou']:+8.4f} "
            f"{('YES' if passed else 'NO'):>5}"
        )

    passing = sorted(
        (record for record in records if record["passed_promotion_gate"]),
        key=candidate_rank,
        reverse=True,
    )
    locked = dict(passing[0]) if passing else None
    if locked is not None:
        checkpoint = Path(str(locked["checkpoint"]))
        locked["sha256"] = file_sha256(checkpoint)

    output = (
        Path(args.output_json)
        if args.output_json
        else root / f"H6_A2_DG_ablation{args.steps}_seed{args.seed}.json"
    )
    payload = {
        "protocol": "H6_A2_DG_validation_only_ablation_v1",
        "test_accessed": False,
        "screening_baseline": "DG0 with equal steps and equal selector",
        "locked_h6_a2_checkpoint": str(baseline_path.resolve()),
        "locked_h6_a2_metrics": locked_baseline,
        "locked_h6_a2_role": (
            "context_only_during_1500_step_screen; mandatory_baseline_after_"
            "the_winning_protocol_is_rerun_for_3000_steps"
        ),
        "promotion_gates_pp": DEFAULT_GATES_PP,
        "ranking_policy": [
            "damaged_iou_delta",
            "event_worst_delta",
            "event_mean_delta",
            "valid_miou_delta",
            "destroyed_iou_delta",
        ],
        "eligible_for_longer_validation_only_run": locked is not None,
        "locked_candidate": locked,
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    if locked is None:
        print("Decision: STOP_DG. No DG1-DG3 candidate passed every validation gate.")
    else:
        print(
            "Decision: CONTINUE_VALIDATION_ONLY_WITH_LOCKED_CANDIDATE | "
            f"{locked['run']} / {locked['selector']} / {locked['checkpoint']}"
        )
        print("Test remains locked; do not run eval_test at the screening stage.")


if __name__ == "__main__":
    main()
