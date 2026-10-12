#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HM-OB0: Train-locked boundary-conditional Occurrence audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from script.standard_ML.hm_ob0_boundary_runtime import (
    MODEL_NAMES,
    apply_final_models,
    boundary_candidate_table,
    event_codes,
    fit_final_models,
    locked_point_at_threshold,
    model_report,
    operating_point_passes,
    select_train_operating_point,
    strict_event_crossfit_scores,
)
from script.standard_ML.im1_matching_reliability_runtime import serialise_calibrator


def load_p0(path: str, role: str) -> Mapping[str, object]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError(f"OB0 input is not a frozen CMR-P0 cache: {path}")
    expected = "train" if role == "train" else "query"
    if metadata.get("role") != expected:
        raise RuntimeError(f"OB0 P0 role mismatch: {metadata.get('role')} != {expected}")
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("OB0 refuses Test-derived caches")
    if not bool(metadata.get("diagnostic_gt_eligibility", False)):
        raise RuntimeError("OB0 requires the declared diagnostic P0 eligibility contract")
    return cache


def parse_coverages(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 or item > 1.0 for item in result):
        raise ValueError("OB0 coverages must be a non-empty comma list in (0,1]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser("HM-OB0 boundary-conditioned Occurrence audit")
    parser.add_argument("--train_p0_cache", required=True)
    parser.add_argument("--validation_p0_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cap_per_event_class", type=int, default=500)
    parser.add_argument("--coverages", default="0.005,0.01,0.025,0.05,0.10")
    parser.add_argument("--logistic_l2", type=float, default=1e-2)
    parser.add_argument("--logistic_max_iter", type=int, default=200)
    parser.add_argument("--minimum_validation_candidates", type=int, default=100)
    parser.add_argument("--minimum_global_auc", type=float, default=0.70)
    parser.add_argument("--minimum_event_auc", type=float, default=0.60)
    parser.add_argument("--minimum_auc_gain_vs_p3", type=float, default=0.05)
    parser.add_argument("--minimum_precision", type=float, default=0.55)
    parser.add_argument("--minimum_rescued", type=int, default=10)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    parser.add_argument("--minimum_positive_events", type=int, default=3)
    args = parser.parse_args()
    coverages = parse_coverages(args.coverages)

    train_cache = load_p0(args.train_p0_cache, "train")
    val_cache = load_p0(args.validation_p0_cache, "validation")
    train_sha = train_cache["metadata"].get("p3_checkpoint_sha256")
    val_sha = val_cache["metadata"].get("p3_checkpoint_sha256")
    if not train_sha or train_sha != val_sha:
        raise RuntimeError("OB0 Train/Validation P3 checkpoint SHA differs")

    train_table = boundary_candidate_table(train_cache)
    val_table = boundary_candidate_table(val_cache)
    train_event_tensor, val_event_tensor, event_map = event_codes(
        train_table["all_events"], val_table["all_events"]
    )

    crossfit = strict_event_crossfit_scores(
        train_table,
        train_event_tensor,
        cap_per_event_class=args.cap_per_event_class,
        seed=args.seed,
        logistic_l2=args.logistic_l2,
        logistic_max_iter=args.logistic_max_iter,
    )
    final_models = fit_final_models(
        train_table,
        train_event_tensor,
        crossfit["scores"]["ob0_b_old_mean"],
        cap_per_event_class=args.cap_per_event_class,
        seed=args.seed,
        logistic_l2=args.logistic_l2,
        logistic_max_iter=args.logistic_max_iter,
    )
    validation_scores = apply_final_models(val_table, final_models)

    train_results = {}
    validation_results = {}
    thresholds = {}
    for name in MODEL_NAMES:
        train_result = model_report(
            crossfit["scores"][name], train_table, coverages=coverages
        )
        operating_point = select_train_operating_point(
            train_result["risk_coverage"],
            minimum_precision=args.minimum_precision,
            minimum_rescued=args.minimum_rescued,
            minimum_nonnegative_events=args.minimum_nonnegative_events,
            minimum_positive_events=args.minimum_positive_events,
        )
        threshold = (
            float(operating_point["score_threshold"])
            if operating_point is not None else None
        )
        train_result["locked_operating_point"] = operating_point
        train_results[name] = train_result
        thresholds[name] = threshold

        validation_result = model_report(
            validation_scores[name], val_table, coverages=coverages
        )
        validation_result["train_locked_operating_point"] = (
            locked_point_at_threshold(
                validation_scores[name],
                torch.as_tensor(val_table["truth"]),
                val_table["events"],
                threshold=threshold,
            )
            if threshold is not None else None
        )
        validation_results[name] = validation_result

    primary = "ob0_c_combined"
    p3_baseline = "ob0_a_p3"
    primary_auc = validation_results[primary]["auc"]
    p3_auc = validation_results[p3_baseline]["auc"]
    auc_gain = primary_auc["global_auroc"] - p3_auc["global_auroc"]
    event_auc_gain = primary_auc["event_macro_auroc"] - p3_auc["event_macro_auroc"]
    train_point = train_results[primary]["locked_operating_point"]
    validation_point = validation_results[primary]["train_locked_operating_point"]
    point_kwargs = {
        "minimum_precision": args.minimum_precision,
        "minimum_rescued": args.minimum_rescued,
        "minimum_nonnegative_events": args.minimum_nonnegative_events,
        "minimum_positive_events": args.minimum_positive_events,
    }
    checks = {
        "at_least_declared_validation_candidates": (
            int(val_table["candidate_count"]) >= args.minimum_validation_candidates
        ),
        "combined_global_auroc_at_least_declared": (
            float(primary_auc["global_auroc"]) >= args.minimum_global_auc
        ),
        "combined_event_auroc_at_least_declared": (
            float(primary_auc["event_macro_auroc"]) >= args.minimum_event_auc
        ),
        "combined_global_auroc_gain_vs_p3_at_least_declared": (
            float(auc_gain) >= args.minimum_auc_gain_vs_p3
        ),
        "train_crossfit_has_admissible_locked_point": operating_point_passes(
            train_point, **point_kwargs
        ),
        "validation_train_locked_point_is_admissible": operating_point_passes(
            validation_point, **point_kwargs
        ),
    }
    passed = all(checks.values())
    payload = {
        "protocol": "HM_OB0_boundary_conditioned_old_mean_occurrence_v1",
        "purpose": (
            "diagnose whether frozen old-P3 Mean adds safe Affected evidence "
            "inside the frozen-P3 Intact boundary pool"
        ),
        "primary_model": primary,
        "controls": {
            "ob0_a_p3": "frozen P3 p(Damaged)+p(Destroyed)",
            "ob0_b_old_mean": "cross-event old-P3 Mean binary-centre margin",
            "ob0_c_combined": "tiny Train-only logistic over the two fixed signals",
        },
        "selection_contract": (
            "No model is selected on Validation.  Combined is predeclared as primary; "
            "each operating threshold is selected only from strict Train event-cross-fit scores."
        ),
        "candidate_contract": "frozen P3 predicts Intact; GT is used only for diagnostic audit",
        "event_map": event_map,
        "train_candidates": {
            key: train_table[key]
            for key in (
                "candidate_count", "missed_damaged", "missed_destroyed", "true_intact"
            )
        },
        "validation_candidates": {
            key: val_table[key]
            for key in (
                "candidate_count", "missed_damaged", "missed_destroyed", "true_intact"
            )
        },
        "train_results": train_results,
        "validation_results": validation_results,
        "diagnostics": {
            "combined_global_auc_gain_vs_p3": float(auc_gain),
            "combined_event_auc_gain_vs_p3": float(event_auc_gain),
        },
        "crossfit_folds": crossfit["folds"],
        "final_model": {
            "fit_centre_rows": int(final_models["fit_centre_rows"]),
            "centres": torch.as_tensor(final_models["centres"]).tolist(),
            "combined_calibrator": serialise_calibrator(final_models["calibrator"]),
        },
        "thresholds": {
            "coverages": list(coverages),
            "train_locked_score_thresholds": thresholds,
            "minimum_validation_candidates": args.minimum_validation_candidates,
            "minimum_global_auc": args.minimum_global_auc,
            "minimum_event_auc": args.minimum_event_auc,
            "minimum_auc_gain_vs_p3": args.minimum_auc_gain_vs_p3,
            "minimum_precision": args.minimum_precision,
            "minimum_rescued": args.minimum_rescued,
            "minimum_nonnegative_events": args.minimum_nonnegative_events,
            "minimum_positive_events": args.minimum_positive_events,
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_TO_HM_OB1_LABEL_FREE_QUERY_AUDIT_NO_FUSION"
            if passed else "STOP_HIERARCHICAL_MEMORY_BOUNDARY_EVIDENCE_INSUFFICIENT"
        ),
        "p3_checkpoint_sha256": train_sha,
        "training": False,
        "fuses_p3": False,
        "validation_selects_model": False,
        "validation_selects_threshold": False,
        "diagnostic_gt_eligibility": True,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 120)
    print("HM-OB0 Train-locked boundary-conditional Occurrence audit")
    print(
        "Train candidates : "
        f"{train_table['candidate_count']} | "
        f"D/X/I={train_table['missed_damaged']}/"
        f"{train_table['missed_destroyed']}/{train_table['true_intact']}"
    )
    print(
        "Valid candidates : "
        f"{val_table['candidate_count']} | "
        f"D/X/I={val_table['missed_damaged']}/"
        f"{val_table['missed_destroyed']}/{val_table['true_intact']}"
    )
    for name in MODEL_NAMES:
        result = validation_results[name]
        auc = result["auc"]
        point = result["train_locked_operating_point"]
        print(
            f"{name:<22} | global={auc['global_auroc']:.4f} | "
            f"event={auc['event_macro_auroc']:.4f} | "
            f"consistent={auc['direction_consistent_events']}/"
            f"{auc['comparable_events']} | locked={point}"
        )
    print(f"Combined gain    : global={auc_gain:+.4f} | event={event_auc_gain:+.4f}")
    print(f"Checks           : {checks}")
    print(f"Decision         : {payload['decision']}")
    print(f"Saved            : {output}")
    print("Training         : False")
    print("P3 fusion        : False")
    print("Test accessed    : False")


if __name__ == "__main__":
    main()
