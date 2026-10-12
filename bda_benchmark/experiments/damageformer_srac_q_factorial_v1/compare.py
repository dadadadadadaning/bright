"""Compare completed, protocol-matched Q Val450 evaluations without training.

The original E0 reproduction gate belongs to each evaluation. This module
refuses smoke runs and reports descriptive paired *aggregate* differences;
it does not infer significance from 450 correlated pixels or events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EVAL_FORMAT = "SRAC_Q_FACTORIAL_EVAL_V1"
ARMS = ("a1_pixel", "q_pixel_grid", "q_region_mean", "q_grid_probq", "q_region")
FOREGROUND = ("Intact", "Damaged", "Destroyed")


def _metric(record: dict, arm: str) -> dict:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or arm not in metrics:
        raise ValueError(f"Missing metric arm: {arm}")
    row = metrics[arm]
    if not isinstance(row, dict) or len(row.get("iou", [])) != 4:
        raise ValueError(f"Incomplete four-class IoU: {arm}")
    if row.get("valid_miou") is None:
        raise ValueError(f"Missing foreground mIoU: {arm}")
    for value in (*row["iou"], row["valid_miou"]):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"Non-finite metric: {arm}")
    return row


def checked_eval(path: str | Path) -> tuple[Path, dict]:
    """Read only a finished full-Val result with the historical E0 gate intact."""
    candidate = Path(path)
    source = candidate / "summary.json" if candidate.is_dir() else candidate
    record = json.loads(source.read_text(encoding="utf-8"))
    immutable = record.get("immutable")
    if (record.get("format") != EVAL_FORMAT
            or record.get("complete") is not True
            or record.get("full_validation_evidence") is not True
            or record.get("image_count") != 450
            or record.get("reproduction", {}).get("passed") is not True
            or record.get("native_a1_replay_ok") is not True
            or record.get("test_accessed") is not False
            or not isinstance(immutable, dict) or not immutable
            or not all(value is True for value in immutable.values())):
        raise ValueError(f"Incomplete, nonformal, changed, or Test evaluation: {source}")
    for field in ("protocol_sha256", "source_sha256", "checkpoint_sha256", "arm"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"Missing evaluation provenance {field}: {source}")
    if not isinstance(record.get("per_event"), dict) or not record["per_event"]:
        raise ValueError(f"Missing per-event comparison evidence: {source}")
    for arm in ARMS:
        _metric(record, arm)
    return source, record


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else 100.0 * (b - a)


def metric_difference(base: dict, candidate: dict) -> dict:
    """Percentage-point differences in foreground mean and four class IoUs."""
    return {
        "foreground_miou_pp": _diff(base["valid_miou"], candidate["valid_miou"]),
        "iou_pp": [_diff(x, y) for x, y in zip(base["iou"], candidate["iou"])],
    }


def _event_differences(base: dict, candidate: dict, arm: str) -> dict:
    a = base.get("per_event", {})
    b = candidate.get("per_event", {})
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
        raise ValueError("Per-event sets differ or are unavailable")
    result = {}
    for event in sorted(a):
        if arm not in a[event] or arm not in b[event]:
            raise ValueError(f"Missing {arm} in event {event}")
        result[event] = metric_difference(a[event][arm], b[event][arm])
    return result


def compare(baseline: dict, candidate: dict) -> dict:
    for field in ("protocol_sha256", "numerical_profile", "pool_mode"):
        a, b = baseline.get(field), candidate.get(field)
        if a != b:
            raise ValueError(f"Incompatible evaluations: {field}")
    if baseline.get("source_sha256") != candidate.get("source_sha256"):
        raise ValueError("Executable source changed between evaluations")
    if (baseline.get("arm") != "baseline" or candidate.get("arm") in (None, "baseline", "original_q")
            or baseline.get("checkpoint_steps") != 2000
            or candidate.get("checkpoint_steps") != 2000):
        raise ValueError("Comparison requires baseline and candidate trained for the same 2000 steps")
    settings = baseline.get("training_hyperparameters")
    if (not isinstance(settings, dict) or not settings
            or candidate.get("training_hyperparameters") != settings):
        raise ValueError("Training budget, data order, precision or cue setting differs")
    return {
        "baseline_arm": baseline.get("arm"),
        "candidate_arm": candidate.get("arm"),
        "protocol_sha256": baseline["protocol_sha256"],
        "matched_training_steps": 2000,
        "training_hyperparameters": settings,
        "checkpoint_sha256": {
            "baseline": baseline.get("checkpoint_sha256"),
            "candidate": candidate.get("checkpoint_sha256"),
        },
        "differences_pp": {
            arm: metric_difference(_metric(baseline, arm), _metric(candidate, arm))
            for arm in ARMS
        },
        "candidate_within_run_q_gain_pp": metric_difference(
            _metric(candidate, "q_pixel_grid"), _metric(candidate, "q_grid_probq")
        ),
        "per_event_grid_probq_pp": _event_differences(
            baseline, candidate, "q_grid_probq"
        ),
        "per_event_region_pp": _event_differences(
            baseline, candidate, "q_region"
        ),
        "interpretation": (
            "Descriptive aggregate Val450 comparison. A1 and the historical "
            "1e-5 gate are verified independently for each run. This is not "
            "a pure Memory effect or a statistical significance test."
        ),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="Completed baseline Q Val450 evaluation")
    p.add_argument("--candidate", required=True, action="append",
                   help="Completed candidate Q Val450 evaluation; repeat as needed")
    p.add_argument("--output", required=True, help="Fresh comparison directory")
    return p


def main() -> None:
    args = parser().parse_args()
    base_path, base = checked_eval(args.baseline)
    seen = set()
    rows = []
    for item in args.candidate:
        path, candidate = checked_eval(item)
        if path == base_path or path in seen:
            raise ValueError("An evaluation was supplied more than once")
        seen.add(path)
        row = compare(base, candidate)
        row["candidate_summary"] = str(path.resolve())
        rows.append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "SRAC_Q_FACTORIAL_COMPARISON_V1",
        "complete": True,
        "test_accessed": False,
        "baseline_summary": str(base_path.resolve()),
        "comparisons": rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for row in rows:
        arm = row["candidate_arm"]
        gain = row["differences_pp"]["q_grid_probq"]["foreground_miou_pp"]
        region = row["differences_pp"]["q_region"]["foreground_miou_pp"]
        print(f"{arm}: grid_probq {gain:+.4f} pp, region {region:+.4f} pp")
    print(output / "comparison.json")


if __name__ == "__main__":
    main()
