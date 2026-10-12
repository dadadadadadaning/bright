# -*- coding: utf-8 -*-
"""Paired event-level admission report for one SALT-CDE ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != "SALT_CDE_AUDIT_v1":
        raise RuntimeError(f"not a SALT audit: {path}")
    return payload


def json_default(value):
    """Convert NumPy scalar diagnostics to standard JSON scalar types."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser("Compare SALT-CDE audits")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference_perturbed", default="")
    parser.add_argument("--candidate_perturbed", default="")
    parser.add_argument("--minimum_events", type=int, default=10)
    parser.add_argument("--minimum_gate_reduction", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    reference, candidate = load(args.reference), load(args.candidate)
    rm, cm = reference["metrics"], candidate["metrics"]
    gain = 100.0 * (float(cm["valid_miou"]) - float(rm["valid_miou"]))
    damaged_gain = 100.0 * (float(cm["damaged_f1"]) - float(rm["damaged_f1"]))
    ref_events, cand_events = rm["event_miou4"], cm["event_miou4"]
    common = sorted(set(ref_events) & set(cand_events))
    event_gains = {event: 100.0 * (float(cand_events[event]) - float(ref_events[event])) for event in common}
    values = np.asarray([event_gains[event] for event in common], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    samples = values[rng.integers(0, len(values), size=(args.repeats, len(values)))].mean(1)
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975]).tolist()
    perturbation = None
    if args.reference_perturbed and args.candidate_perturbed:
        rp, cp = load(args.reference_perturbed), load(args.candidate_perturbed)
        ref_drop = 100.0 * (float(rm["valid_miou"]) - float(rp["metrics"]["valid_miou"]))
        cand_drop = 100.0 * (float(cm["valid_miou"]) - float(cp["metrics"]["valid_miou"]))
        perturbation = {
            "name": cp["perturbation"],
            "reference_drop_pp": ref_drop,
            "candidate_drop_pp": cand_drop,
            "candidate_more_stable": cand_drop <= ref_drop,
        }
        candidate_clean_gates = [
            float(value) for key, value in candidate.get("diagnostics", {}).items()
            if key.endswith("_sar_gate")
        ]
        candidate_wrong_gates = [
            float(value) for key, value in cp.get("diagnostics", {}).items()
            if key.endswith("_sar_gate")
        ]
        if candidate_clean_gates and candidate_wrong_gates:
            perturbation["candidate_clean_gate"] = float(np.mean(candidate_clean_gates))
            perturbation["candidate_perturbed_gate"] = float(np.mean(candidate_wrong_gates))
            perturbation["candidate_gate_reduction"] = (
                perturbation["candidate_clean_gate"]
                - perturbation["candidate_perturbed_gate"]
            )
    nonnegative_events = int(np.count_nonzero(values >= 0.0))
    improved_events = int(np.count_nonzero(values > 0.0))
    checks = {
        "gain_at_least_0_30pp": bool(gain >= 0.30),
        "damaged_f1_nonnegative": bool(damaged_gain >= 0.0),
        "enough_nonnegative_events": bool(nonnegative_events >= args.minimum_events),
        "event_bootstrap_ci_low_positive": bool(ci_low > 0.0),
    }
    if perturbation is not None:
        checks["perturbation_more_stable"] = bool(perturbation["candidate_more_stable"])
        if perturbation["name"] in ("pair_shuffle", "zero") and "candidate_gate_reduction" in perturbation:
            checks["wrong_sar_reduces_gate"] = bool(
                perturbation["candidate_gate_reduction"]
                >= args.minimum_gate_reduction
            )
    result = {
        "protocol": "SALT_CDE_ADMISSION_v1",
        "reference_variant": reference["variant"],
        "candidate_variant": candidate["variant"],
        "gain_pp": gain,
        "damaged_f1_gain_pp": damaged_gain,
        "events": {
            "count": len(common),
            "nonnegative": nonnegative_events,
            "improved": improved_events,
            "gain_pp": event_gains,
        },
        "bootstrap_event_macro_gain_pp": {
            "observed": float(values.mean()),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "repeats": args.repeats,
        },
        "perturbation": perturbation,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, default=json_default))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
