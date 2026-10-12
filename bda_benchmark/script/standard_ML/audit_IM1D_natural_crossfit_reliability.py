# -*- coding: utf-8 -*-
"""IM1-D natural-query, event-cross-fitted rescue/harm reliability audit.

The class/event-balanced train cache is used only as the retrieval support
memory.  A separate unbalanced train cache supplies natural-distribution
calibration queries.  Logistic scores and the acceptance threshold are chosen
from leave-one-event-out predictions, then frozen for one validation audit.
P3 predictions are never modified and test caches are rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from script.standard_ML.audit_RM_region_retrieval import load_cache, retrieve_setting
from script.standard_ML.im1_matching_reliability_runtime import (
    apply_logistic_calibrator,
    cross_fitted_logistic_scores,
    event_macro_auroc,
    fit_logistic_calibrator,
    operating_point_at_threshold,
    proposal_signal_table,
    risk_coverage_table,
    select_train_operating_point,
    serialise_calibrator,
)
from script.standard_ML.rm_region_retrieval_runtime import (
    binary_auroc,
    validate_disjoint_ids,
)


GROUP_NAMES = (
    "im1_a_p3_memory",
    "im1_b_matching",
    "im1_c_combined",
)


def _validate_cache(
    cache: Mapping[str, object], *, role: str, split: str
) -> None:
    metadata = cache["metadata"]
    if metadata.get("role") != role or metadata.get("source_split") != split:
        raise RuntimeError(
            f"IM1-D expected role/split={role}/{split}, observed="
            f"{metadata.get('role')}/{metadata.get('source_split')}"
        )
    if not bool(metadata.get("matching_stats_enabled", False)):
        raise RuntimeError("IM1-D requires caches built with --matching_stats")
    for key in (
        "base_probabilities", "colocated_stats", "matching_stats",
        "relative_raw_stats",
    ):
        if key not in cache:
            raise KeyError(f"IM1-D cache is missing {key!r}")


def _evaluate_group(
    calibration_group: Mapping[str, object],
    validation_group: Mapping[str, object],
    calibration_target: torch.Tensor,
    validation_target: torch.Tensor,
    calibration_events: list[str],
    validation_events: list[str],
    coverages: list[float],
    l2: float,
) -> Dict[str, object]:
    crossfit = cross_fitted_logistic_scores(
        calibration_group["features"], calibration_target, calibration_events,
        l2=l2,
        balance_classes=False,
    )
    out_of_fold_score = crossfit["scores"]
    out_of_fold_rows = risk_coverage_table(
        out_of_fold_score, calibration_target, calibration_events,
        coverages=coverages,
    )
    selected = select_train_operating_point(out_of_fold_rows)
    locked_threshold = (
        float(selected["score_threshold"]) if selected is not None else None
    )

    final_model = fit_logistic_calibrator(
        calibration_group["features"], calibration_target,
        l2=l2,
        balance_classes=False,
    )
    validation_score = apply_logistic_calibrator(
        validation_group["features"], final_model
    )
    validation_rows = risk_coverage_table(
        validation_score, validation_target, validation_events,
        coverages=coverages,
    )
    locked_validation = (
        operating_point_at_threshold(
            validation_score, validation_target, validation_events,
            threshold=locked_threshold,
        )
        if locked_threshold is not None else None
    )
    return {
        "signal_names": list(calibration_group["names"]),
        "feature_dimension": int(calibration_group["features"].shape[1]),
        "calibration_contract": (
            "natural train queries; leave-one-event-out scores; "
            "unbalanced BCE; frozen validation threshold"
        ),
        "crossfit_folds": crossfit["folds"],
        "out_of_fold_train": {
            "auroc": binary_auroc(out_of_fold_score, calibration_target),
            **event_macro_auroc(
                out_of_fold_score, calibration_target, calibration_events
            ),
            "operating_points": out_of_fold_rows,
            "selected_operating_point": selected,
        },
        "final_calibrator": serialise_calibrator(final_model),
        "validation": {
            "auroc": binary_auroc(validation_score, validation_target),
            **event_macro_auroc(
                validation_score, validation_target, validation_events
            ),
            "diagnostic_operating_points": validation_rows,
            "out_of_fold_locked_operating_point": locked_validation,
        },
    }


def _proposal_summary(table: Mapping[str, object]) -> Dict[str, object]:
    auditable = int(table["auditable_proposals"])
    rescue = int(table["rescue_proposals"])
    return {
        key: table[key] for key in (
            "proposed_total", "auditable_proposals", "rescue_proposals",
            "harm_proposals", "destroyed_proposals_excluded",
        )
    } | {"rescue_prior": rescue / max(auditable, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(
        "IM1-D natural-query event-cross-fitted matching reliability"
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--calibration_queries", required=True)
    parser.add_argument("--validation_queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw_weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    parser.add_argument("--coverages", default="0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--logistic_l2", type=float, default=0.01)
    args = parser.parse_args()

    memory = load_cache(args.memory, "memory")
    calibration = load_cache(args.calibration_queries, "calibration")
    validation = load_cache(args.validation_queries, "query")
    _validate_cache(memory, role="memory", split="train")
    _validate_cache(calibration, role="calibration", split="train")
    _validate_cache(validation, role="query", split="val")
    if memory["metadata"].get("balance_mode") != "class_event":
        raise RuntimeError("IM1-D support memory must be class-event balanced")
    if calibration["metadata"].get("balance_mode") != "none":
        raise RuntimeError("IM1-D calibration queries must keep natural distribution")
    hashes = {
        cache["metadata"]["p3_checkpoint_sha256"]
        for cache in (memory, calibration, validation)
    }
    if len(hashes) != 1:
        raise RuntimeError("IM1-D caches use different P3 checkpoints")
    validate_disjoint_ids(memory["image_ids"], validation["image_ids"])
    validate_disjoint_ids(calibration["image_ids"], validation["image_ids"])

    retrieval_kwargs = dict(
        raw_weight=float(args.raw_weight),
        topk=int(args.topk),
        beta=float(args.beta),
        exclude_same_event=True,
        max_neighbors_per_event=int(args.max_neighbors_per_event),
        sar_stat_mode="scene_relative",
        device=torch.device(args.device),
    )
    calibration_retrieval = retrieve_setting(
        memory, calibration, **retrieval_kwargs
    )
    validation_retrieval = retrieve_setting(
        memory, validation, **retrieval_kwargs
    )
    calibration_table = proposal_signal_table(
        calibration,
        calibration_retrieval["prediction"],
        calibration_retrieval["scores"],
    )
    validation_table = proposal_signal_table(
        validation,
        validation_retrieval["prediction"],
        validation_retrieval["scores"],
    )
    coverages = [
        float(value) for value in args.coverages.split(",") if value.strip()
    ]
    if not coverages:
        raise ValueError("IM1-D coverages cannot be empty")

    results: Dict[str, object] = {}
    for name in GROUP_NAMES:
        results[name] = _evaluate_group(
            calibration_table["groups"][name],
            validation_table["groups"][name],
            calibration_table["target"],
            validation_table["target"],
            calibration_table["events"],
            validation_table["events"],
            coverages,
            float(args.logistic_l2),
        )

    baseline_auc = float(results[GROUP_NAMES[0]]["validation"]["auroc"])
    matching_auc = float(results[GROUP_NAMES[1]]["validation"]["auroc"])
    combined_auc = float(results[GROUP_NAMES[2]]["validation"]["auroc"])
    combined_point = results[GROUP_NAMES[2]]["validation"][
        "out_of_fold_locked_operating_point"
    ]
    point_passed = bool(
        combined_point is not None
        and float(combined_point["precision"]) >= 0.55
        and int(combined_point["net_correction"]) > 0
        and int(combined_point["rescued"]) >= 10
        and int(combined_point["nonnegative_events"]) >= 10
    )
    checks = {
        "at_least_100_validation_proposals": (
            int(validation_table["auditable_proposals"]) >= 100
        ),
        "combined_auroc_at_least_0_70": combined_auc >= 0.70,
        "combined_auroc_plus_0_05_vs_p3_memory": (
            combined_auc >= baseline_auc + 0.05
        ),
        "crossfit_locked_positive_validation_point": point_passed,
    }
    passed = all(checks.values())
    if passed:
        decision = "ADMIT_IM1D_MATCHING_RELIABILITY_DO_NOT_TEST"
    elif combined_auc > baseline_auc and combined_auc >= 0.60:
        decision = "UPGRADE_DESCRIPTOR_UNDER_IM1D_PROTOCOL_DO_NOT_FUSE"
    else:
        decision = "STOP_P3_LOCAL_MATCHING_ROUTE_DO_NOT_FUSE"

    payload: Dict[str, object] = {
        "protocol": "IM1D_natural_query_event_crossfit_reliability_v1",
        "scope": "train-only natural calibration; frozen one-shot validation",
        "proposal_contract": (
            "P3=Intact and cross-event Memory=Damaged; truth Intact/Damaged only"
        ),
        "memory": str(Path(args.memory).resolve()),
        "calibration_queries": str(Path(args.calibration_queries).resolve()),
        "validation_queries": str(Path(args.validation_queries).resolve()),
        "retrieval_contract": {
            "raw_weight": float(args.raw_weight),
            "sar_stat_mode": "scene_relative",
            "exclude_same_event": True,
            "max_neighbors_per_event": int(args.max_neighbors_per_event),
            "topk_per_class": int(args.topk),
            "beta": float(args.beta),
        },
        "calibration_proposals": _proposal_summary(calibration_table),
        "validation_proposals": _proposal_summary(validation_table),
        "results": results,
        "validation_auc_summary": {
            "im1_a_p3_memory": baseline_auc,
            "im1_b_matching": matching_auc,
            "im1_c_combined": combined_auc,
            "combined_minus_p3_memory": combined_auc - baseline_auc,
        },
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 108)
    print("IM1-D natural-query event-cross-fitted reliability audit")
    print(
        f"Calibration proposals: {calibration_table['auditable_proposals']} | "
        f"rescue/harm={calibration_table['rescue_proposals']}/"
        f"{calibration_table['harm_proposals']} | "
        f"prior={payload['calibration_proposals']['rescue_prior']:.4f}"
    )
    print(
        f"Validation proposals : {validation_table['auditable_proposals']} | "
        f"rescue/harm={validation_table['rescue_proposals']}/"
        f"{validation_table['harm_proposals']} | "
        f"prior={payload['validation_proposals']['rescue_prior']:.4f}"
    )
    for name in GROUP_NAMES:
        result = results[name]
        print(
            f"{name:<20} | OOF={result['out_of_fold_train']['auroc']:.4f} | "
            f"val={result['validation']['auroc']:.4f} | "
            f"event={result['validation']['event_macro_auroc']:.4f} | "
            f"locked={result['validation']['out_of_fold_locked_operating_point']}"
        )
    print(f"Checks          : {checks}")
    print(f"Decision        : {decision}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()
