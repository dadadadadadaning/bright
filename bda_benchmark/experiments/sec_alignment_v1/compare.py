# -*- coding: utf-8 -*-
"""Compare one SEC candidate with the already-completed D1 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load(path: str, protocols):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") not in protocols:
        raise RuntimeError(f"unexpected audit protocol in {path}")
    return payload


def mean_shift_axis_error(payload: Dict[str, object]) -> float | None:
    shift = payload.get("synthetic_shift_pixels", {})
    axis = "x" if int(shift.get("dx", 0)) != 0 else "y"
    values = [
        float(item[f"absolute_error_{axis}"])
        for item in payload.get("offset_delta_response", {}).values()
        if f"absolute_error_{axis}" in item
    ]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser("SEC admission against frozen D1")
    parser.add_argument("--reference_clean", required=True)
    parser.add_argument("--candidate_clean", required=True)
    parser.add_argument("--reference_x8", required=True)
    parser.add_argument("--reference_y8", required=True)
    parser.add_argument("--candidate_x8", required=True)
    parser.add_argument("--candidate_y8", required=True)
    parser.add_argument("--minimum_events", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    sec_protocol = {"SEC_AUDIT_v1"}
    rc = load(args.reference_clean, sec_protocol)
    cc = load(args.candidate_clean, sec_protocol)
    rx = load(args.reference_x8, sec_protocol)
    ry = load(args.reference_y8, sec_protocol)
    cx = load(args.candidate_x8, sec_protocol)
    cy = load(args.candidate_y8, sec_protocol)
    rm, cm = rc["metrics"], cc["metrics"]
    gain = 100.0 * (float(cm["valid_miou"]) - float(rm["valid_miou"]))
    damaged_gain = 100.0 * (float(cm["damaged_f1"]) - float(rm["damaged_f1"]))
    common = sorted(set(rm["event_miou4"]) & set(cm["event_miou4"]))
    event_gains = {
        event: 100.0 * (
            float(cm["event_miou4"][event]) - float(rm["event_miou4"][event])
        )
        for event in common
    }
    values = np.asarray(list(event_gains.values()), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    boot = values[
        rng.integers(0, len(values), size=(args.repeats, len(values)))
    ].mean(1)
    ci_low, ci_high = (float(v) for v in np.quantile(boot, [0.025, 0.975]))

    ref_drops = [
        100.0 * (float(rm["valid_miou"]) - float(p["metrics"]["valid_miou"]))
        for p in (rx, ry)
    ]
    cand_drops = [
        100.0 * (float(cm["valid_miou"]) - float(p["metrics"]["valid_miou"]))
        for p in (cx, cy)
    ]
    ref_errors = [value for value in (mean_shift_axis_error(rx), mean_shift_axis_error(ry)) if value is not None]
    cand_errors = [value for value in (mean_shift_axis_error(cx), mean_shift_axis_error(cy)) if value is not None]
    ref_error = float(np.mean(ref_errors)) if ref_errors else None
    cand_error = float(np.mean(cand_errors)) if cand_errors else None
    nonnegative = int(np.count_nonzero(values >= 0.0))

    screen_checks = {
        "trained_checkpoint_selected": bool(int(cc.get("checkpoint_step", 0)) > 0),
        "clean_noninferiority_ge_minus_0_30pp": bool(gain >= -0.30),
        "damaged_f1_nonnegative": bool(damaged_gain >= 0.0),
        "at_least_10_of_14_events_nonnegative": bool(nonnegative >= args.minimum_events),
        "x8_shift_drop_not_worse_than_d1": bool(np.mean(cand_drops) <= np.mean(ref_drops)),
    }
    if cc["variant"] in (
        "s2", "s3", "s2r", "s2r0", "s2rp", "s2rr", "s2rcf"
    ):
        screen_checks["offset_delta_mae_reduced_at_least_30pct"] = bool(
            ref_error is not None and cand_error is not None
            and cand_error <= 0.70 * ref_error
        )
    result = {
        "protocol": "SEC_ADMISSION_v1",
        "reference": rc.get("variant", "unknown"),
        "candidate": cc["variant"],
        "candidate_checkpoint_step": int(cc.get("checkpoint_step", 0)),
        "clean_gain_pp": gain,
        "damaged_f1_gain_pp": damaged_gain,
        "events": {
            "count": len(common),
            "nonnegative": nonnegative,
            "improved": int(np.count_nonzero(values > 0.0)),
            "gain_pp": event_gains,
        },
        "bootstrap_event_macro_gain_pp": {
            "observed": float(values.mean()),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "repeats": args.repeats,
        },
        "x8_shift_robustness": {
            "d1_mean_drop_pp": float(np.mean(ref_drops)),
            "candidate_mean_drop_pp": float(np.mean(cand_drops)),
            "d1_offset_absolute_error": ref_error,
            "candidate_relative_delta_shift_axis_mae": cand_error,
            "note": "D1 and SEC use the same shifted-minus-clean audit and GT-building support.",
        },
        "screen_checks": screen_checks,
        "screen_pass": bool(all(screen_checks.values())),
        "confirm_checks": {
            "clean_gain_ge_0_30pp": bool(gain >= 0.30),
            "event_bootstrap_ci_low_positive": bool(ci_low > 0.0),
        },
        "confirm_ready": bool(
            all(screen_checks.values()) and gain >= 0.30 and ci_low > 0.0
        ),
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
