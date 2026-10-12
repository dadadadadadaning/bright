# -*- coding: utf-8 -*-
"""Train-LOEO calibrated IM1 rescue/harm reliability audit.

The audit fits three tiny logistic calibrators on train-memory leave-one-event-
out proposals and evaluates their frozen scores once on validation.  It never
changes P3 predictions and never accepts a test cache.
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
    class_event_stratum_weights,
    validate_disjoint_ids,
)


def _validate_matching_cache(cache: Mapping[str, object]) -> None:
    metadata = cache["metadata"]
    if not bool(metadata.get("matching_stats_enabled", False)):
        raise RuntimeError("IM1 requires caches built with --matching_stats")
    for key in ("colocated_stats", "matching_stats", "base_probabilities"):
        if key not in cache:
            raise KeyError(f"IM1 cache is missing {key!r}")


def _evaluate_group(
    train_group: Mapping[str, object],
    validation_group: Mapping[str, object],
    train_target: torch.Tensor,
    validation_target: torch.Tensor,
    train_events: list[str],
    validation_events: list[str],
    train_weights: torch.Tensor,
    coverages: list[float],
    l2: float,
) -> Dict[str, object]:
    model = fit_logistic_calibrator(
        train_group["features"], train_target,
        sample_weight=train_weights, l2=l2,
    )
    train_score = apply_logistic_calibrator(train_group["features"], model)
    validation_score = apply_logistic_calibrator(
        validation_group["features"], model
    )
    train_rows = risk_coverage_table(
        train_score, train_target, train_events, coverages=coverages
    )
    selected_train = select_train_operating_point(train_rows)
    selected_threshold = (
        float(selected_train["score_threshold"])
        if selected_train is not None else None
    )
    validation_rows = risk_coverage_table(
        validation_score, validation_target, validation_events,
        coverages=coverages,
    )
    validation_selected = (
        operating_point_at_threshold(
            validation_score, validation_target, validation_events,
            threshold=selected_threshold,
        )
        if selected_threshold is not None else None
    )
    train_event_auc = event_macro_auroc(train_score, train_target, train_events)
    validation_event_auc = event_macro_auroc(
        validation_score, validation_target, validation_events
    )
    return {
        "signal_names": list(train_group["names"]),
        "feature_dimension": int(train_group["features"].shape[1]),
        "calibrator": serialise_calibrator(model),
        "selection_split": "train-memory leave-one-event-out proposals only",
        "train": {
            "auroc": binary_auroc(train_score, train_target),
            **train_event_auc,
            "operating_points": train_rows,
            "selected_operating_point": selected_train,
        },
        "validation": {
            "auroc": binary_auroc(validation_score, validation_target),
            **validation_event_auc,
            "operating_points": validation_rows,
            "train_selected_operating_point": validation_selected,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser("IM1 proposal-conditional matching reliability")
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True)
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
    query = load_cache(args.queries, "query")
    _validate_matching_cache(memory)
    _validate_matching_cache(query)
    if memory["metadata"].get("source_split") != "train":
        raise RuntimeError("IM1 calibration memory must come from train split")
    if query["metadata"].get("source_split") != "val":
        raise RuntimeError("IM1 audit queries must come from val split; test is forbidden")
    validate_disjoint_ids(memory["image_ids"], query["image_ids"])
    if memory["metadata"]["p3_checkpoint_sha256"] != query["metadata"]["p3_checkpoint_sha256"]:
        raise RuntimeError("IM1 memory/query P3 checkpoint mismatch")
    if memory["metadata"].get("balance_mode") != "class_event":
        raise RuntimeError("IM1 requires class-event-balanced train memory")

    device = torch.device(args.device)
    retrieval_kwargs = dict(
        raw_weight=float(args.raw_weight),
        topk=int(args.topk),
        beta=float(args.beta),
        exclude_same_event=True,
        max_neighbors_per_event=int(args.max_neighbors_per_event),
        sar_stat_mode="scene_relative",
        device=device,
    )
    # Memory acts as a query only against other events, so no item can retrieve
    # itself or another sample from its own disaster event.
    train_retrieval = retrieve_setting(memory, memory, **retrieval_kwargs)
    validation_retrieval = retrieve_setting(memory, query, **retrieval_kwargs)
    train_table = proposal_signal_table(
        memory, train_retrieval["prediction"], train_retrieval["scores"]
    )
    validation_table = proposal_signal_table(
        query, validation_retrieval["prediction"], validation_retrieval["scores"]
    )

    metadata = memory["metadata"]
    region_weights = class_event_stratum_weights(
        memory["labels"], memory["events"],
        metadata["class_event_counts_before_balance"],
        metadata["class_event_counts"],
    )
    candidate_weights = region_weights[train_table["indices"]]
    coverages = [float(value) for value in args.coverages.split(",") if value.strip()]
    if not coverages:
        raise ValueError("IM1 coverages cannot be empty")

    results: Dict[str, object] = {}
    for name in ("im1_a_p3_memory", "im1_b_matching", "im1_c_combined"):
        results[name] = _evaluate_group(
            train_table["groups"][name], validation_table["groups"][name],
            train_table["target"], validation_table["target"],
            train_table["events"], validation_table["events"],
            candidate_weights, coverages, float(args.logistic_l2),
        )

    baseline_auc = float(results["im1_a_p3_memory"]["validation"]["auroc"])
    matching_auc = float(results["im1_b_matching"]["validation"]["auroc"])
    combined_auc = float(results["im1_c_combined"]["validation"]["auroc"])
    combined_point = results["im1_c_combined"]["validation"][
        "train_selected_operating_point"
    ]
    point_passed = bool(
        combined_point is not None
        and float(combined_point["precision"]) >= 0.55
        and int(combined_point["net_correction"]) > 0
        and int(combined_point["rescued"]) >= 10
        and int(combined_point["nonnegative_events"]) >= 10
    )
    checks = {
        "at_least_100_validation_proposals": validation_table["auditable_proposals"] >= 100,
        "combined_auroc_at_least_0_70": combined_auc >= 0.70,
        "combined_auroc_plus_0_05_vs_p3_memory": combined_auc >= baseline_auc + 0.05,
        "train_selected_positive_validation_point": point_passed,
    }
    passed = all(checks.values())
    if passed:
        decision = "ADMIT_MATCHING_AS_RM6P_RELIABILITY_DO_NOT_TEST"
    elif combined_auc > baseline_auc and combined_auc >= 0.60:
        decision = "UPGRADE_DESCRIPTOR_WITH_SAME_IM1_PROTOCOL_DO_NOT_FUSE"
    else:
        decision = "STOP_MATCHING_RELIABILITY_ROUTE_DO_NOT_FUSE"

    payload: Dict[str, object] = {
        "protocol": "IM1_proposal_conditional_matching_reliability_v1",
        "scope": "train-LOEO calibrated; frozen validation audit; P3 unchanged",
        "proposal_contract": "P3=Intact and cross-event Memory=Damaged; truth Intact/Damaged only",
        "memory": str(Path(args.memory).resolve()),
        "queries": str(Path(args.queries).resolve()),
        "retrieval_contract": {
            "raw_weight": float(args.raw_weight),
            "sar_stat_mode": "scene_relative",
            "exclude_same_event": True,
            "max_neighbors_per_event": int(args.max_neighbors_per_event),
            "topk_per_class": int(args.topk),
            "beta": float(args.beta),
        },
        "train_proposals": {
            key: train_table[key] for key in (
                "proposed_total", "auditable_proposals", "rescue_proposals",
                "harm_proposals", "destroyed_proposals_excluded",
            )
        },
        "validation_proposals": {
            key: validation_table[key] for key in (
                "proposed_total", "auditable_proposals", "rescue_proposals",
                "harm_proposals", "destroyed_proposals_excluded",
            )
        },
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
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print("IM1 proposal-conditional rescue/harm reliability audit")
    print(
        f"Train proposals : {train_table['auditable_proposals']} | "
        f"rescue/harm={train_table['rescue_proposals']}/{train_table['harm_proposals']}"
    )
    print(
        f"Valid proposals : {validation_table['auditable_proposals']} | "
        f"rescue/harm={validation_table['rescue_proposals']}/{validation_table['harm_proposals']}"
    )
    for name in ("im1_a_p3_memory", "im1_b_matching", "im1_c_combined"):
        result = results[name]
        selected = result["validation"]["train_selected_operating_point"]
        print(
            f"{name:<20} | AUC={result['validation']['auroc']:.4f} | "
            f"event={result['validation']['event_macro_auroc']:.4f} | "
            f"locked-point={selected}"
        )
    print(f"Checks          : {checks}")
    print(f"Decision        : {decision}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()
