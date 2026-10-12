# -*- coding: utf-8 -*-
"""CMR-V2 EC0--EC3: event-factorized counterfactual memory audit.

This script is deliberately diagnostic-only.  It reads frozen P0/M3 caches,
uses Train labels to define the labelled reference memory, evaluates only
Train/Validation, and never changes a P3 prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from model.cmr_v2.event_factorization import (
    event_expert_deltas,
    fit_event_factorization,
    frozen_topk_delta,
    transform_features,
)
from script.standard_ML.im1_matching_reliability_runtime import (
    event_macro_auroc,
    operating_point_at_threshold,
    risk_coverage_table,
    select_train_operating_point,
)
from script.standard_ML.rm_region_retrieval_runtime import binary_auroc, validate_disjoint_ids


def load_p0(path: str, role: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("CMR-V2 requires frozen CMR-P0")
    if metadata.get("role") != role or bool(metadata.get("test_accessed", True)):
        raise RuntimeError(f"P0 role/Test contract failed for {path}")
    return payload


def load_candidates(path: str, role: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("CMR-V2 requires frozen M3 Raw-Mean candidates")
    if payload.get("role") != role or bool(payload.get("test_accessed", True)):
        raise RuntimeError(f"candidate role/Test contract failed for {path}")
    return payload


def validate_alignment(p0: Mapping[str, object], candidate: Mapping[str, object]) -> None:
    truth = p0["labels"].long().flatten()
    if not torch.equal(truth, candidate["query_labels"].long().flatten()):
        raise RuntimeError("P0/candidate GT rows do not align")
    memory_rows = candidate["memory_indices_in_train"].long().flatten()
    if bool((memory_rows < 0).any()):
        raise RuntimeError("candidate cache has negative memory source rows")


def proposal_rows(p0: Mapping[str, object], candidate: Mapping[str, object]) -> torch.Tensor:
    """Label-free admission followed by I/D-only diagnostic evaluation."""
    has_damaged = (candidate["candidate_labels"].long() == 1).any(dim=1)
    admitted = p0["base_predictions"].long().flatten() == 0
    auditable = (p0["labels"].long() == 0) | (p0["labels"].long() == 1)
    return torch.nonzero(admitted & has_damaged & auditable, as_tuple=False).flatten()


def recall_at_k(candidate_labels: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> Dict[str, object]:
    output: Dict[str, object] = {"queries": int(mask.sum())}
    for k in (1, 5, 20):
        if k > candidate_labels.shape[1]:
            continue
        hit = (candidate_labels[:, :k] == truth[:, None]).any(dim=1) & mask
        output[f"hit@{k}"] = int(hit.sum())
        output[f"recall@{k}"] = float(hit.sum()) / max(int(mask.sum()), 1)
    return output


def score_split(
    p0: Mapping[str, object],
    candidate: Mapping[str, object],
    memory_features_by_variant: Mapping[str, torch.Tensor],
    query_features_by_variant: Mapping[str, torch.Tensor],
    memory_labels: torch.Tensor,
    memory_events: Sequence[str],
    memory_rows: torch.Tensor,
    *,
    case_k: int,
    temperature: float,
    device: str,
    query_chunk: int,
    train_split: bool,
) -> Dict[str, object]:
    rows = proposal_rows(p0, candidate)
    query_events = [str(p0["events"][row]) for row in rows.tolist()]
    source_rows = rows.long()
    ec0_all = frozen_topk_delta(
        candidate["candidate_scores"], candidate["candidate_labels"],
        case_k=case_k, temperature=temperature,
    )["delta"]
    scores: Dict[str, torch.Tensor] = {"ec0_frozen_cross_event_top20": ec0_all[rows]}
    diagnostics: Dict[str, object] = {}

    experiment_specs = (
        ("ec1_same_event_raw", "raw", "same_event"),
        ("ec2_exclude_same_event_raw", "raw", "exclude_same_event"),
        ("ec3_exclude_same_event_centered", "centered", "exclude_same_event"),
        ("ec3_exclude_same_event_whitened", "whitened", "exclude_same_event"),
    )
    for name, variant, protocol in experiment_specs:
        result = event_expert_deltas(
            query_features_by_variant[variant][rows], query_events, source_rows,
            memory_features_by_variant[variant], memory_labels, memory_events, memory_rows,
            protocol=protocol, case_k=case_k, temperature=temperature,
            device=device, query_chunk=query_chunk,
            exclude_identical_source=train_split,
        )
        scores[name] = result["mean_delta"]
        diagnostics[name] = {
            "expert_order": result["expert_order"],
            "mean_expert_count": float(result["expert_count"].float().mean()),
            "mean_expert_std": float(result["expert_std"][torch.isfinite(result["expert_std"])].mean()),
            "mean_positive_fraction": float(
                result["positive_fraction"][torch.isfinite(result["positive_fraction"])].mean()
            ),
            "at_least_three_positive_experts": int((result["positive_experts"] >= 3).sum()),
            "positive_fraction_at_least_0_70": int((result["positive_fraction"] >= 0.70).sum()),
            "expert_count": result["expert_count"],
            "expert_std": result["expert_std"],
            "positive_experts": result["positive_experts"],
            "positive_fraction": result["positive_fraction"],
        }
    common = torch.ones(rows.numel(), dtype=torch.bool)
    for value in scores.values():
        common &= torch.isfinite(value)
    if not bool(common.any()):
        raise RuntimeError("EC0--EC3 have no common auditable proposals")
    diagnostic_summary: Dict[str, object] = {}
    for name, row in diagnostics.items():
        count = row["expert_count"][common]
        std = row["expert_std"][common]
        fraction = row["positive_fraction"][common]
        positive = row["positive_experts"][common]
        diagnostic_summary[name] = {
            "expert_order": row["expert_order"],
            "mean_expert_count": float(count.float().mean()),
            "mean_expert_std": float(std[torch.isfinite(std)].mean()),
            "mean_positive_fraction": float(fraction[torch.isfinite(fraction)].mean()),
            "at_least_three_positive_experts": int((positive >= 3).sum()),
            "positive_fraction_at_least_0_70": int((fraction >= 0.70).sum()),
            "joint_three_experts_and_0_70_fraction": int(
                ((positive >= 3) & (fraction >= 0.70)).sum()
            ),
        }
    return {
        "source_rows": rows[common],
        "scores": {name: value[common] for name, value in scores.items()},
        "target": p0["labels"][rows][common].long() == 1,
        "events": [event for event, keep in zip(query_events, common.tolist()) if keep],
        "diagnostics": diagnostic_summary,
        "label_free_admitted": int(rows.numel()),
        "common_evaluable": int(common.sum()),
        "dropped_nonfinite": int((~common).sum()),
    }


def evaluate_experiment(
    train_score: torch.Tensor,
    validation_score: torch.Tensor,
    train_target: torch.Tensor,
    validation_target: torch.Tensor,
    train_events: Sequence[str],
    validation_events: Sequence[str],
    coverages: Sequence[float],
) -> Dict[str, object]:
    train_rows = risk_coverage_table(train_score, train_target, train_events, coverages=coverages)
    selected = select_train_operating_point(train_rows)
    threshold = None if selected is None else float(selected["score_threshold"])
    locked = None if threshold is None else operating_point_at_threshold(
        validation_score, validation_target, validation_events, threshold=threshold
    )
    return {
        "train": {
            "rescue_vs_harm_auroc": binary_auroc(train_score, train_target),
            **event_macro_auroc(train_score, train_target, train_events),
            "operating_points": train_rows,
            "selected_operating_point": selected,
        },
        "validation": {
            "rescue_vs_harm_auroc": binary_auroc(validation_score, validation_target),
            **event_macro_auroc(validation_score, validation_target, validation_events),
            "train_locked_operating_point": locked,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-V2 EC0--EC3 event-factorized audit")
    parser.add_argument("--train_p0", required=True)
    parser.add_argument("--val_p0", required=True)
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--scale_floor", type=float, default=1e-3)
    parser.add_argument("--query_chunk", type=int, default=2048)
    parser.add_argument("--coverages", default="0.01,0.025,0.05,0.10,0.20")
    args = parser.parse_args()

    train = load_p0(args.train_p0, "train")
    validation = load_p0(args.val_p0, "query")
    train_candidate = load_candidates(args.train_candidates, "train")
    validation_candidate = load_candidates(args.val_candidates, "validation")
    validate_alignment(train, train_candidate)
    validate_alignment(validation, validation_candidate)
    validate_disjoint_ids(train["image_ids"], validation["image_ids"])
    if train["metadata"].get("p3_checkpoint_sha256") != validation["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("Train/Validation use different P3 checkpoints")

    memory_rows = train_candidate["memory_indices_in_train"].long()
    if not torch.equal(memory_rows, train["memory_indices"].long()):
        raise RuntimeError("M3 memory source rows differ from frozen P0 memory")
    memory_labels = train["labels"][memory_rows].long()
    memory_events = [str(train["events"][row]) for row in memory_rows.tolist()]
    factorization = fit_event_factorization(
        train["key_variants"]["mean"], train["labels"], train["events"],
        scale_floor=args.scale_floor,
    )
    variants = ("raw", "centered", "whitened")
    memory_features = {
        variant: transform_features(
            train["key_variants"]["mean"][memory_rows], memory_events, factorization, variant
        ) for variant in variants
    }
    train_features = {
        variant: transform_features(
            train["key_variants"]["mean"], train["events"], factorization, variant
        ) for variant in variants
    }
    validation_features = {
        variant: transform_features(
            validation["key_variants"]["mean"], validation["events"], factorization, variant
        ) for variant in variants
    }
    common_kwargs = dict(
        memory_features_by_variant=memory_features,
        memory_labels=memory_labels,
        memory_events=memory_events,
        memory_rows=memory_rows,
        case_k=args.case_k,
        temperature=args.temperature,
        device=args.device,
        query_chunk=args.query_chunk,
    )
    train_table = score_split(
        train, train_candidate, query_features_by_variant=train_features,
        train_split=True, **common_kwargs,
    )
    validation_table = score_split(
        validation, validation_candidate, query_features_by_variant=validation_features,
        train_split=False, **common_kwargs,
    )
    coverages = [float(value) for value in args.coverages.split(",") if value.strip()]
    results = {
        name: evaluate_experiment(
            train_table["scores"][name], validation_table["scores"][name],
            train_table["target"], validation_table["target"],
            train_table["events"], validation_table["events"], coverages,
        ) for name in train_table["scores"]
    }

    ec0 = results["ec0_frozen_cross_event_top20"]["validation"]
    ec2 = results["ec2_exclude_same_event_raw"]["validation"]
    centered = results["ec3_exclude_same_event_centered"]["validation"]
    centered_event = centered["event_auroc"]
    ec0_event = ec0["event_auroc"]
    common_events = sorted(set(centered_event) & set(ec0_event))
    nondegraded = sum(centered_event[event] >= ec0_event[event] for event in common_events)
    centered_locked = centered["train_locked_operating_point"]
    damaged_missed = (validation["labels"] == 1) & (validation["base_predictions"] == 0)
    damaged_recall = recall_at_k(
        validation_candidate["candidate_labels"], validation["labels"], damaged_missed
    )
    checks = {
        "p3_missed_damaged_recall_at_20_at_least_0_90": damaged_recall.get("recall@20", 0.0) >= 0.90,
        "centered_rescue_vs_harm_auroc_at_least_0_82": centered["rescue_vs_harm_auroc"] >= 0.82,
        "centered_event_macro_auroc_at_least_0_75": centered["event_macro_auroc"] >= 0.75,
        "at_least_10_events_nondegraded_vs_ec0": nondegraded >= 10,
        "centering_improves_ec2_raw_by_0_02": (
            centered["rescue_vs_harm_auroc"] - ec2["rescue_vs_harm_auroc"] >= 0.02
        ),
        "train_locked_validation_net_at_least_30": (
            centered_locked is not None and centered_locked["net_correction"] >= 30
        ),
    }
    factor_summary = {
        "protocol": factorization["protocol"],
        "center_definition": factorization["center_definition"],
        "event_order": factorization["event_order"],
        "class_counts": factorization["class_counts"].tolist(),
        "reliable_three_class_center": factorization["reliable"].tolist(),
        "fallback_events": [
            event for event, reliable in zip(factorization["event_order"], factorization["reliable"].tolist())
            if not reliable
        ],
        "whitening": "class-balanced diagonal residual scale",
    }
    payload = {
        "protocol": "CMR_V2_ISOLATED_EC0_EC3_event_factorized_audit_v1",
        "scope": "frozen P3/P0/M3; Train/Validation diagnostic; no prediction fusion",
        "experiments": {
            "EC0": "frozen cross-event Raw-Mean Top-20 counterfactual baseline",
            "EC1": "same-event raw-feature counterfactual upper bound",
            "EC2": "same-event-excluded raw-feature event experts",
            "EC3": "EC2 with class-balanced event centering or diagonal whitening",
        },
        "factorization": factor_summary,
        "train_proposals": train_table["common_evaluable"],
        "validation_proposals": validation_table["common_evaluable"],
        "proposal_rule": "P3 predicts Intact and frozen cross-event Top-20 contains Damaged",
        "proposal_rule_uses_query_gt": False,
        "evaluation_filters_gt_to_intact_or_damaged": True,
        "p3_missed_damaged_recall": damaged_recall,
        "results": results,
        "multi_event_consistency_diagnostics": {
            "train": train_table["diagnostics"],
            "validation": validation_table["diagnostics"],
            "note": "diagnostic only; no acceptance/fusion is performed in EC0--EC3",
        },
        "event_nondegradation_vs_ec0": {
            "common_events": len(common_events),
            "nondegraded_events": nondegraded,
            "per_event_delta": {
                event: centered_event[event] - ec0_event[event] for event in common_events
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "ADMIT_EC4_MULTIPROTOTYPE_DO_NOT_FUSE"
            if all(checks.values()) else "STOP_BEFORE_EC4_EVENT_FACTORIZATION_INSUFFICIENT"
        ),
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print("CMR-V2 isolated EC0--EC3 event-factorized audit")
    for name, row in results.items():
        validation_row = row["validation"]
        print(
            f"{name:<39} AUC={validation_row['rescue_vs_harm_auroc']:.4f} "
            f"event={validation_row['event_macro_auroc']:.4f} "
            f"locked={validation_row['train_locked_operating_point']}"
        )
    print(f"Checks   : {checks}")
    print(f"Decision : {payload['decision']}")
    print(f"Saved    : {output}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
