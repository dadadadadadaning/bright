# -*- coding: utf-8 -*-
"""RM1--RM3E validation-only retrieval audit over frozen region caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from script.standard_ML.rm_region_retrieval_runtime import (
    build_combined_keys,
    class_event_stratum_weights,
    class_balanced_knn,
    complement_audit,
    gain_retention_audit,
    metrics_from_predictions,
    p3_anchored_memory_fusion,
    prior_corrected_retrieval,
    selective_damaged_rescue_audit,
    shuffled_labels,
    validate_disjoint_ids,
    weighted_complement_audit,
)


def load_cache(path: str, expected_role: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"cache must contain a mapping: {path}")
    for key in ("keys", "labels", "base_predictions", "raw_stats", "events", "image_ids", "metadata"):
        if key not in payload:
            raise KeyError(f"cache {path} is missing {key!r}")
    metadata = payload["metadata"]
    if metadata.get("role") != expected_role:
        raise RuntimeError(
            f"cache role mismatch: observed={metadata.get('role')}, expected={expected_role}"
        )
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("cache is not marked validation-safe")
    return payload


def retrieve_setting(
    memory: Mapping[str, object],
    query: Mapping[str, object],
    *,
    raw_weight: float,
    topk: int,
    beta: float,
    exclude_same_event: bool = False,
    max_neighbors_per_event: int = 0,
    sar_stat_mode: str = "absolute",
    label_override: torch.Tensor | None = None,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if sar_stat_mode not in ("absolute", "scene_relative"):
        raise ValueError(f"unsupported SAR statistic mode: {sar_stat_mode}")
    stats_key = "raw_stats" if sar_stat_mode == "absolute" else "relative_raw_stats"
    normalizer_key = (
        "raw_stat_normalizer"
        if sar_stat_mode == "absolute"
        else "relative_raw_stat_normalizer"
    )
    if stats_key not in memory or stats_key not in query:
        raise KeyError(
            f"{sar_stat_mode} statistics require RM_region_cache_v2; rebuild caches"
        )
    normalizer = memory["metadata"][normalizer_key]
    memory_key = build_combined_keys(
        memory["keys"], memory[stats_key], normalizer, raw_weight
    ).to(device)
    query_key = build_combined_keys(
        query["keys"], query[stats_key], normalizer, raw_weight
    ).to(device)
    labels = memory["labels"] if label_override is None else label_override
    return class_balanced_knn(
        query_key,
        memory_key,
        labels,
        topk_per_class=topk,
        beta=beta,
        query_events=query["events"],
        memory_events=memory["events"],
        exclude_same_event=exclude_same_event,
        max_neighbors_per_event=max_neighbors_per_event,
    )


def evaluate_setting(
    memory: Mapping[str, object],
    query: Mapping[str, object],
    *,
    raw_weight: float,
    topk: int,
    beta: float,
    exclude_same_event: bool = False,
    max_neighbors_per_event: int = 0,
    sar_stat_mode: str = "absolute",
    label_override: torch.Tensor | None = None,
    device: torch.device,
) -> Dict[str, object]:
    result = retrieve_setting(
        memory,
        query,
        raw_weight=raw_weight,
        topk=topk,
        beta=beta,
        exclude_same_event=exclude_same_event,
        max_neighbors_per_event=max_neighbors_per_event,
        sar_stat_mode=sar_stat_mode,
        label_override=label_override,
        device=device,
    )
    prediction = result["prediction"]
    metrics = metrics_from_predictions(query["labels"], prediction, query["events"])
    base_metrics = metrics_from_predictions(
        query["labels"], query["base_predictions"], query["events"]
    )
    complement = complement_audit(
        query["labels"], query["base_predictions"], prediction
    )
    event_delta = {
        event: 100.0
        * (
            metrics["events"][event]["macro_f1"]
            - base_metrics["events"][event]["macro_f1"]
        )
        for event in metrics["events"]
    }
    return {
        "raw_weight": float(raw_weight),
        "exclude_same_event": bool(exclude_same_event),
        "max_neighbors_per_event": int(max_neighbors_per_event),
        "sar_stat_mode": sar_stat_mode,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "complement": complement,
        "event_macro_f1_delta_pp_vs_p3": event_delta,
        "nonnegative_events_vs_p3": sum(value >= 0.0 for value in event_delta.values()),
        "mean_best_similarity": float(result["best_similarity_by_class"].mean()),
        "mean_unique_neighbor_events_by_class": [
            float(value)
            for value in result["unique_neighbor_events_by_class"].mean(dim=0)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser("Audit RM1--RM3E frozen region retrieval")
    parser.add_argument(
        "--stage",
        choices=("rm1", "rm2", "rm3", "rm3e_a", "rm3e_b", "rm3e_c", "rm4", "rm6a", "rm6p"),
        required=True,
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--raw_weights", default="0,0.25,0.5,1.0")
    parser.add_argument("--selected_raw_weight", type=float, default=0.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    parser.add_argument("--shuffle_repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--prior_taus", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--prior_smoothing", type=float, default=1.0)
    parser.add_argument("--memory_alphas", default="0,0.01,0.025,0.05,0.1,0.2,0.3,0.5")
    args = parser.parse_args()

    memory = load_cache(args.memory, "memory")
    query = load_cache(args.queries, "query")
    validate_disjoint_ids(memory["image_ids"], query["image_ids"])
    if memory["metadata"]["p3_checkpoint_sha256"] != query["metadata"]["p3_checkpoint_sha256"]:
        raise RuntimeError("memory and query caches were not extracted by the same frozen P3")
    device = torch.device(args.device)
    payload: Dict[str, object] = {
        "protocol": "RM_region_retrieval_audit_v1",
        "stage": args.stage,
        "memory": str(Path(args.memory).resolve()),
        "queries": str(Path(args.queries).resolve()),
        "test_accessed": False,
    }

    if args.stage == "rm1":
        payload["result"] = evaluate_setting(
            memory,
            query,
            raw_weight=0.0,
            topk=args.topk,
            beta=args.beta,
            device=device,
        )
        payload["decision"] = "PROCEED_TO_RM2" if payload["result"]["complement"]["per_class"]["damaged"]["net_complement"] > 0 else "STOP_RETRIEVAL_NO_DAMAGED_COMPLEMENT"
    elif args.stage == "rm2":
        weights = [float(value) for value in args.raw_weights.split(",") if value.strip()]
        results = [
            evaluate_setting(
                memory,
                query,
                raw_weight=weight,
                topk=args.topk,
                beta=args.beta,
                device=device,
            )
            for weight in weights
        ]
        winner = max(
            results,
            key=lambda item: (
                item["metrics"]["event_macro_f1"],
                item["metrics"]["macro_f1"],
                item["metrics"]["class_recall"][1],
            ),
        )
        ordinary = evaluate_setting(
            memory, query, raw_weight=winner["raw_weight"], topk=args.topk,
            beta=args.beta, exclude_same_event=False,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="scene_relative", device=device,
        )
        payload.update(
            {
                "results": results,
                "selected_raw_weight": winner["raw_weight"],
                "selection_order": "event_macro_f1 -> macro_f1 -> damaged_recall",
                "decision": "PROCEED_TO_RM3_CONTROLS",
            }
        )
    elif args.stage == "rm3":
        ordinary = evaluate_setting(
            memory,
            query,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            device=device,
        )
        cross_event = evaluate_setting(
            memory,
            query,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            device=device,
        )
        shuffled = []
        for repeat in range(args.shuffle_repeats):
            result = evaluate_setting(
                memory,
                query,
                raw_weight=args.selected_raw_weight,
                topk=args.topk,
                beta=args.beta,
                exclude_same_event=True,
                label_override=shuffled_labels(memory["labels"], args.seed + repeat),
                device=device,
            )
            shuffled.append(float(result["metrics"]["macro_f1"]))
        shuffled_tensor = torch.tensor(shuffled)
        shuffle_mean = float(shuffled_tensor.mean())
        shuffle_std = float(shuffled_tensor.std(unbiased=False))
        base_macro = float(ordinary["base_metrics"]["macro_f1"])
        gain_audit = gain_retention_audit(
            base_macro,
            ordinary["metrics"]["macro_f1"],
            cross_event["metrics"]["macro_f1"],
        )
        checks = {
            "cross_event_beats_shuffle_3sigma": float(cross_event["metrics"]["macro_f1"]) > shuffle_mean + 3.0 * shuffle_std,
            "damaged_net_complement_positive": cross_event["complement"]["per_class"]["damaged"]["net_complement"] > 0,
            "ten_events_nondecreasing_vs_p3": cross_event["nonnegative_events_vs_p3"] >= 10,
            str(gain_audit["check_name"]): bool(gain_audit["passed"]),
        }
        payload.update(
            {
                "ordinary": ordinary,
                "exclude_same_event": cross_event,
                "label_shuffle": {
                    "repeats": args.shuffle_repeats,
                    "macro_f1_values": shuffled,
                    "mean": shuffle_mean,
                    "std": shuffle_std,
                },
                "gain_retention": gain_audit["gain_retention"],
                "gain_audit": gain_audit,
                "checks": checks,
                "passed": all(checks.values()),
                "decision": "PROCEED_TO_TRAINED_CROSS_ATTENTION" if all(checks.values()) else "STOP_OR_REDESIGN_MEMORY_DO_NOT_TRAIN_FUSION",
            }
        )
    elif args.stage == "rm3e_a":
        if memory["metadata"].get("balance_mode") != "class_event":
            raise RuntimeError("RM3E-A requires a class_event-balanced memory cache")
        ordinary = evaluate_setting(
            memory, query, raw_weight=args.selected_raw_weight, topk=args.topk,
            beta=args.beta, device=device,
        )
        cross_event = evaluate_setting(
            memory, query, raw_weight=args.selected_raw_weight, topk=args.topk,
            beta=args.beta, exclude_same_event=True, device=device,
        )
        base_macro = float(ordinary["base_metrics"]["macro_f1"])
        gain_audit = gain_retention_audit(
            base_macro,
            ordinary["metrics"]["macro_f1"],
            cross_event["metrics"]["macro_f1"],
        )
        checks = {
            "damaged_net_complement_positive": cross_event["complement"]["per_class"]["damaged"]["net_complement"] > 0,
            "ten_events_nondecreasing_vs_p3": cross_event["nonnegative_events_vs_p3"] >= 10,
            str(gain_audit["check_name"]): bool(gain_audit["passed"]),
        }
        payload.update({
            "ordinary": ordinary,
            "exclude_same_event": cross_event,
            "gain_retention": gain_audit["gain_retention"],
            "gain_audit": gain_audit,
            "checks": checks,
            "passed": all(checks.values()),
            "decision": "PROCEED_TO_RM3E_B_WITH_A_PASS" if all(checks.values()) else "PROCEED_TO_RM3E_B_DIAGNOSTIC_DO_NOT_TRAIN_FUSION",
        })
    elif args.stage == "rm3e_b":
        standard = evaluate_setting(
            memory, query, raw_weight=args.selected_raw_weight, topk=args.topk,
            beta=args.beta, exclude_same_event=True, device=device,
        )
        diverse = evaluate_setting(
            memory, query, raw_weight=args.selected_raw_weight, topk=args.topk,
            beta=args.beta, exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event, device=device,
        )
        checks = {
            "event_macro_f1_nondecreasing": diverse["metrics"]["event_macro_f1"] >= standard["metrics"]["event_macro_f1"],
            "damaged_net_complement_positive": diverse["complement"]["per_class"]["damaged"]["net_complement"] > 0,
            "ten_events_nondecreasing_vs_p3": diverse["nonnegative_events_vs_p3"] >= 10,
        }
        payload.update({
            "standard_cross_event": standard,
            "event_diverse_cross_event": diverse,
            "checks": checks,
            "passed": all(checks.values()),
            "decision": "PROCEED_TO_RM3E_C_WITH_B_PASS" if all(checks.values()) else "PROCEED_TO_RM3E_C_DIAGNOSTIC_DO_NOT_TRAIN_FUSION",
        })
    elif args.stage == "rm3e_c":
        weights = [float(value) for value in args.raw_weights.split(",") if value.strip()]
        ordinary = evaluate_setting(
            memory,
            query,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="absolute",
            device=device,
        )
        results = [
            evaluate_setting(
                memory, query, raw_weight=weight, topk=args.topk, beta=args.beta,
                exclude_same_event=True,
                max_neighbors_per_event=args.max_neighbors_per_event,
                sar_stat_mode="scene_relative", device=device,
            )
            for weight in weights
        ]
        winner = max(
            results,
            key=lambda item: (
                item["metrics"]["event_macro_f1"], item["metrics"]["macro_f1"],
                item["metrics"]["class_recall"][1],
            ),
        )
        shuffled = []
        for repeat in range(args.shuffle_repeats):
            result = evaluate_setting(
                memory, query, raw_weight=winner["raw_weight"], topk=args.topk,
                beta=args.beta, exclude_same_event=True,
                max_neighbors_per_event=args.max_neighbors_per_event,
                sar_stat_mode="scene_relative",
                label_override=shuffled_labels(memory["labels"], args.seed + repeat),
                device=device,
            )
            shuffled.append(float(result["metrics"]["macro_f1"]))
        shuffled_tensor = torch.tensor(shuffled)
        shuffle_mean = float(shuffled_tensor.mean())
        shuffle_std = float(shuffled_tensor.std(unbiased=False))
        base_macro = float(ordinary["base_metrics"]["macro_f1"])
        gain_audit = gain_retention_audit(
            base_macro,
            ordinary["metrics"]["macro_f1"],
            winner["metrics"]["macro_f1"],
        )
        checks = {
            "cross_event_beats_shuffle_3sigma": float(winner["metrics"]["macro_f1"]) > shuffle_mean + 3.0 * shuffle_std,
            "damaged_net_complement_positive": winner["complement"]["per_class"]["damaged"]["net_complement"] > 0,
            "ten_events_nondecreasing_vs_p3": winner["nonnegative_events_vs_p3"] >= 10,
            str(gain_audit["check_name"]): bool(gain_audit["passed"]),
        }
        payload.update({
            "results": results,
            "selected_raw_weight": winner["raw_weight"],
            "selected": winner,
            "ordinary": ordinary,
            "gain_retention": gain_audit["gain_retention"],
            "gain_audit": gain_audit,
            "selection_order": "cross-event event_macro_f1 -> macro_f1 -> damaged_recall",
            "label_shuffle": {"repeats": args.shuffle_repeats, "macro_f1_values": shuffled, "mean": shuffle_mean, "std": shuffle_std},
            "checks": checks,
            "passed": all(checks.values()),
            "decision": "ADMIT_TRAINED_RETRIEVAL_FUSION" if all(checks.values()) else "STOP_BEFORE_TRAINED_FUSION",
        })
    elif args.stage == "rm4":
        if "base_probabilities" not in query:
            raise KeyError("RM4 requires base_probabilities in the query cache")
        retrieval = retrieve_setting(
            memory,
            query,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="scene_relative",
            device=device,
        )
        retrieval_metrics = metrics_from_predictions(
            query["labels"], retrieval["prediction"], query["events"]
        )
        retrieval_complement = complement_audit(
            query["labels"], query["base_predictions"], retrieval["prediction"]
        )
        confidence = selective_damaged_rescue_audit(
            query["labels"],
            query["base_predictions"],
            query["base_probabilities"],
            retrieval["prediction"],
            retrieval["scores"],
            query["events"],
        )
        payload.update({
            "selected_raw_weight": float(args.selected_raw_weight),
            "retrieval_contract": {
                "exclude_same_event": True,
                "max_neighbors_per_event": int(args.max_neighbors_per_event),
                "sar_stat_mode": "scene_relative",
                "topk_per_class": int(args.topk),
                "beta": float(args.beta),
            },
            "retrieval_metrics": retrieval_metrics,
            "retrieval_complement": retrieval_complement,
            "confidence_audit": confidence,
            "passed": bool(confidence["passed"]),
            "decision": (
                "PROCEED_TO_TRAIN_ONLY_CALIBRATION_SPLIT_DO_NOT_TEST"
                if confidence["passed"]
                else "STOP_BEFORE_TRAINED_FUSION_CONFIDENCE_NOT_SEPARABLE"
            ),
        })
    elif args.stage == "rm6a":
        # RM6-A: choose the prior strength strictly inside the train memory by
        # leave-one-event-out retrieval, then evaluate that locked value once
        # on validation.  Validation labels never participate in tau choice.
        metadata = memory["metadata"]
        required = ("class_counts_before_balance", "class_event_counts_before_balance", "class_event_counts")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise KeyError(f"RM6-A memory metadata is missing: {missing}; rebuild RM3E cache")
        if metadata.get("balance_mode") != "class_event":
            raise RuntimeError("RM6-A requires a class_event-balanced train memory")
        taus = [float(value) for value in args.prior_taus.split(",") if value.strip()]
        if not taus or 0.0 not in taus or any(value < 0.0 for value in taus):
            raise ValueError("prior_taus must contain 0 and only non-negative values")
        train_retrieval = retrieve_setting(
            memory,
            memory,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="scene_relative",
            device=device,
        )
        train_weights = class_event_stratum_weights(
            memory["labels"],
            memory["events"],
            metadata["class_event_counts_before_balance"],
            metadata["class_event_counts"],
        )
        train_candidates = []
        for tau in taus:
            corrected = prior_corrected_retrieval(
                train_retrieval["scores"],
                metadata["class_counts_before_balance"],
                tau,
                smoothing=args.prior_smoothing,
            )
            audit = weighted_complement_audit(
                memory["labels"],
                memory["base_predictions"],
                corrected["prediction"],
                train_weights,
                memory["events"],
            )
            train_candidates.append({
                "tau": float(tau),
                "weighted_complement": audit,
            })
        # The first objective is natural-distribution net correction.  Event
        # breadth and Damaged rescue break ties; smaller tau is the final safe
        # tie-breaker.  This is fully determined before validation is read.
        selected_train = max(
            train_candidates,
            key=lambda item: (
                item["weighted_complement"]["overall"]["net"],
                item["weighted_complement"]["nonnegative_events"],
                item["weighted_complement"]["per_class"]["damaged"]["net"],
                -item["tau"],
            ),
        )
        selected_tau = float(selected_train["tau"])

        validation_retrieval = retrieve_setting(
            memory,
            query,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="scene_relative",
            device=device,
        )
        uncorrected_prediction = prior_corrected_retrieval(
            validation_retrieval["scores"],
            metadata["class_counts_before_balance"],
            0.0,
            smoothing=args.prior_smoothing,
        )["prediction"]
        corrected = prior_corrected_retrieval(
            validation_retrieval["scores"],
            metadata["class_counts_before_balance"],
            selected_tau,
            smoothing=args.prior_smoothing,
        )
        base_metrics = metrics_from_predictions(
            query["labels"], query["base_predictions"], query["events"]
        )
        uncorrected_metrics = metrics_from_predictions(
            query["labels"], uncorrected_prediction, query["events"]
        )
        corrected_metrics = metrics_from_predictions(
            query["labels"], corrected["prediction"], query["events"]
        )
        uncorrected_complement = complement_audit(
            query["labels"], query["base_predictions"], uncorrected_prediction
        )
        corrected_complement = complement_audit(
            query["labels"], query["base_predictions"], corrected["prediction"]
        )
        event_delta = {
            event: 100.0 * (
                corrected_metrics["events"][event]["macro_f1"]
                - base_metrics["events"][event]["macro_f1"]
            )
            for event in corrected_metrics["events"]
        }
        intact_harm_before = uncorrected_complement["per_class"]["intact"]["p3_right_retrieval_wrong"]
        intact_harm_after = corrected_complement["per_class"]["intact"]["p3_right_retrieval_wrong"]
        intact_harm_reduction = (
            1.0 - intact_harm_after / intact_harm_before
            if intact_harm_before > 0 else 0.0
        )
        checks = {
            "intact_harm_reduced_at_least_50pct": intact_harm_reduction >= 0.50,
            "damaged_net_complement_positive": corrected_complement["per_class"]["damaged"]["net_complement"] > 0,
            "overall_net_correction_positive": corrected_complement["net_complement"] > 0,
            "ten_events_nondecreasing_vs_p3": sum(value >= 0.0 for value in event_delta.values()) >= 10,
            "macro_f1_within_1pp_of_p3": corrected_metrics["macro_f1"] >= base_metrics["macro_f1"] - 0.01,
        }
        payload.update({
            "selection_split": "train-memory leave-one-event-out only",
            "tau_candidates": train_candidates,
            "selected_tau": selected_tau,
            "selection_order": "weighted overall net -> nonnegative train events -> damaged net -> smaller tau",
            "class_counts_before_balance": list(metadata["class_counts_before_balance"]),
            "class_priors": corrected["priors"],
            "validation": {
                "base_metrics": base_metrics,
                "uncorrected_metrics": uncorrected_metrics,
                "corrected_metrics": corrected_metrics,
                "uncorrected_complement": uncorrected_complement,
                "corrected_complement": corrected_complement,
                "intact_harm_reduction": intact_harm_reduction,
                "event_macro_f1_delta_pp_vs_p3": event_delta,
                "nonnegative_events_vs_p3": sum(value >= 0.0 for value in event_delta.values()),
            },
            "retrieval_contract": {
                "score_correction": "log(affinity_mean + eps) + tau * log(train_prior)",
                "exclude_same_event": True,
                "max_neighbors_per_event": int(args.max_neighbors_per_event),
                "sar_stat_mode": "scene_relative",
                "raw_weight": float(args.selected_raw_weight),
                "topk_per_class": int(args.topk),
                "beta": float(args.beta),
            },
            "checks": checks,
            "passed": all(checks.values()),
            "decision": (
                "PROCEED_TO_RM6B_INDEPENDENT_RETRIEVAL_KEY_DO_NOT_TEST"
                if all(checks.values())
                else "STOP_MEMORY_FUSION_PRIOR_CORRECTION_INSUFFICIENT"
            ),
        })
    else:
        # RM6-P: P3-anchored product-of-experts.  Memory is never allowed to
        # act as a replacement classifier.  Alpha is selected with natural-
        # distribution stratum weights on train-memory LOEO retrieval.
        metadata = memory["metadata"]
        required = ("class_event_counts_before_balance", "class_event_counts")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise KeyError(f"RM6-P memory metadata is missing: {missing}; rebuild RM3E cache")
        if metadata.get("balance_mode") != "class_event":
            raise RuntimeError("RM6-P requires a class_event-balanced train memory")
        if "base_probabilities" not in memory or "base_probabilities" not in query:
            raise KeyError("RM6-P requires P3 conditional probabilities in both caches")
        alphas = [float(value) for value in args.memory_alphas.split(",") if value.strip()]
        if not alphas or 0.0 not in alphas or any(value < 0.0 for value in alphas):
            raise ValueError("memory_alphas must contain 0 and only non-negative values")

        train_retrieval = retrieve_setting(
            memory, memory,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="scene_relative",
            device=device,
        )
        train_weights = class_event_stratum_weights(
            memory["labels"], memory["events"],
            metadata["class_event_counts_before_balance"],
            metadata["class_event_counts"],
        )
        guarded_train = p3_anchored_memory_fusion(
            memory["base_probabilities"], train_retrieval["scores"], 0.0
        )["prediction"]
        if not torch.equal(guarded_train, memory["base_predictions"].long().cpu()):
            mismatch = int((guarded_train != memory["base_predictions"].long().cpu()).sum())
            raise AssertionError(f"RM6-P alpha=0 does not reproduce P3 for {mismatch} train regions")
        train_candidates = []
        for alpha in alphas:
            fused = p3_anchored_memory_fusion(
                memory["base_probabilities"], train_retrieval["scores"], alpha
            )
            audit = weighted_complement_audit(
                memory["labels"], memory["base_predictions"], fused["prediction"],
                train_weights, memory["events"],
            )
            train_candidates.append({
                "alpha": float(alpha),
                "weighted_complement": audit,
            })
        selected_train = max(
            train_candidates,
            key=lambda item: (
                item["weighted_complement"]["overall"]["net"],
                item["weighted_complement"]["nonnegative_events"],
                item["weighted_complement"]["per_class"]["damaged"]["net"],
                -item["alpha"],
            ),
        )
        selected_alpha = float(selected_train["alpha"])

        validation_retrieval = retrieve_setting(
            memory, query,
            raw_weight=args.selected_raw_weight,
            topk=args.topk,
            beta=args.beta,
            exclude_same_event=True,
            max_neighbors_per_event=args.max_neighbors_per_event,
            sar_stat_mode="scene_relative",
            device=device,
        )
        guarded_validation = p3_anchored_memory_fusion(
            query["base_probabilities"], validation_retrieval["scores"], 0.0
        )["prediction"]
        if not torch.equal(guarded_validation, query["base_predictions"].long().cpu()):
            mismatch = int((guarded_validation != query["base_predictions"].long().cpu()).sum())
            raise AssertionError(f"RM6-P alpha=0 does not reproduce P3 for {mismatch} validation regions")
        fused = p3_anchored_memory_fusion(
            query["base_probabilities"], validation_retrieval["scores"], selected_alpha
        )
        base_metrics = metrics_from_predictions(
            query["labels"], query["base_predictions"], query["events"]
        )
        fused_metrics = metrics_from_predictions(
            query["labels"], fused["prediction"], query["events"]
        )
        complement = complement_audit(
            query["labels"], query["base_predictions"], fused["prediction"]
        )
        event_delta = {
            event: 100.0 * (
                fused_metrics["events"][event]["macro_f1"]
                - base_metrics["events"][event]["macro_f1"]
            )
            for event in fused_metrics["events"]
        }
        changed = int((fused["prediction"] != query["base_predictions"].long().cpu()).sum())
        checks = {
            "selected_alpha_positive": selected_alpha > 0.0,
            "overall_net_correction_positive": complement["net_complement"] > 0,
            "damaged_net_complement_positive": complement["per_class"]["damaged"]["net_complement"] > 0,
            "macro_f1_nondecreasing": fused_metrics["macro_f1"] >= base_metrics["macro_f1"],
            "ten_events_nondecreasing_vs_p3": sum(value >= 0.0 for value in event_delta.values()) >= 10,
        }
        payload.update({
            "selection_split": "train-memory leave-one-event-out only",
            "alpha_candidates": train_candidates,
            "selected_alpha": selected_alpha,
            "selection_order": "weighted overall net -> nonnegative train events -> damaged net -> smaller alpha",
            "fusion_contract": "log(P3 conditional probability) + alpha * log(memory affinity)",
            "validation": {
                "base_metrics": base_metrics,
                "fused_metrics": fused_metrics,
                "complement": complement,
                "changed_regions": changed,
                "event_macro_f1_delta_pp_vs_p3": event_delta,
                "nonnegative_events_vs_p3": sum(value >= 0.0 for value in event_delta.values()),
            },
            "retrieval_contract": {
                "exclude_same_event": True,
                "max_neighbors_per_event": int(args.max_neighbors_per_event),
                "sar_stat_mode": "scene_relative",
                "raw_weight": float(args.selected_raw_weight),
                "topk_per_class": int(args.topk),
                "beta": float(args.beta),
            },
            "checks": checks,
            "passed": all(checks.values()),
            "decision": (
                "ADMIT_RM6P_FOR_MULTI_SEED_VALIDATION_DO_NOT_TEST"
                if all(checks.values())
                else "STOP_DIRECT_MEMORY_FUSION_P3_ANCHORED_EVIDENCE_INSUFFICIENT"
            ),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 96)
    print(f"RM stage       : {args.stage}")
    if args.stage == "rm1":
        result = payload["result"]
        print(f"Macro/Event F1 : {100*result['metrics']['macro_f1']:.4f} / {100*result['metrics']['event_macro_f1']:.4f}")
        print(f"Damaged recall : {100*result['metrics']['class_recall'][1]:.4f}")
        print(f"Damaged net    : {result['complement']['per_class']['damaged']['net_complement']}")
    elif args.stage == "rm2":
        print(f"Selected raw w : {payload['selected_raw_weight']}")
        for result in payload["results"]:
            print(f"  w={result['raw_weight']:<4} | macro={100*result['metrics']['macro_f1']:.4f} | event={100*result['metrics']['event_macro_f1']:.4f} | D-net={result['complement']['per_class']['damaged']['net_complement']}")
    elif args.stage == "rm3":
        print(f"Ordinary macro : {100*payload['ordinary']['metrics']['macro_f1']:.4f}")
        print(f"Cross-event    : {100*payload['exclude_same_event']['metrics']['macro_f1']:.4f}")
        print(f"Shuffle mean   : {100*payload['label_shuffle']['mean']:.4f} +/- {100*payload['label_shuffle']['std']:.4f}")
        print(f"Checks         : {payload['checks']}")
    elif args.stage == "rm3e_a":
        print(f"Ordinary macro : {100*payload['ordinary']['metrics']['macro_f1']:.4f}")
        print(f"Cross-event    : {100*payload['exclude_same_event']['metrics']['macro_f1']:.4f}")
        retention = payload["gain_retention"]
        print("Gain retention : N/A" if retention is None else f"Gain retention : {100*retention:.2f}%")
        print(f"Checks         : {payload['checks']}")
    elif args.stage == "rm3e_b":
        print(f"Standard event : {100*payload['standard_cross_event']['metrics']['event_macro_f1']:.4f}")
        print(f"Diverse event  : {100*payload['event_diverse_cross_event']['metrics']['event_macro_f1']:.4f}")
        print(f"Checks         : {payload['checks']}")
    elif args.stage == "rm3e_c":
        print(f"Selected rel w : {payload['selected_raw_weight']}")
        for result in payload["results"]:
            print(f"  w={result['raw_weight']:<4} | macro={100*result['metrics']['macro_f1']:.4f} | event={100*result['metrics']['event_macro_f1']:.4f} | D-net={result['complement']['per_class']['damaged']['net_complement']}")
        print(f"Checks         : {payload['checks']}")
    elif args.stage == "rm4":
        audit = payload["confidence_audit"]
        print(f"Auditable I->D : {audit['auditable_proposals']}")
        print(f"Rescue/Harm    : {audit['rescue_proposals']} / {audit['harm_proposals']}")
        print(f"Best signal    : {audit['best_signal']}")
        print(f"Best AUROC     : {audit['signal_auroc'][audit['best_signal']]:.4f}")
        print(f"Best point     : {audit['best_admissible_point']}")
        print(f"Checks         : {audit['checks']}")
    elif args.stage == "rm6a":
        validation = payload["validation"]
        before = validation["uncorrected_complement"]
        after = validation["corrected_complement"]
        print(f"Selected tau   : {payload['selected_tau']}")
        print(f"Train priors   : {[round(value, 6) for value in payload['class_priors']]}")
        print(f"Macro P3/raw/c : {100*validation['base_metrics']['macro_f1']:.4f} / {100*validation['uncorrected_metrics']['macro_f1']:.4f} / {100*validation['corrected_metrics']['macro_f1']:.4f}")
        print(f"Intact harm    : {before['per_class']['intact']['p3_right_retrieval_wrong']} -> {after['per_class']['intact']['p3_right_retrieval_wrong']}")
        print(f"Damaged net    : {before['per_class']['damaged']['net_complement']} -> {after['per_class']['damaged']['net_complement']}")
        print(f"Overall net    : {before['net_complement']} -> {after['net_complement']}")
        print(f"Nonneg events  : {validation['nonnegative_events_vs_p3']}/14")
        print(f"Checks         : {payload['checks']}")
    else:
        validation = payload["validation"]
        complement = validation["complement"]
        print(f"Selected alpha : {payload['selected_alpha']}")
        print(f"Changed regions: {validation['changed_regions']}")
        print(f"Macro P3/fused : {100*validation['base_metrics']['macro_f1']:.4f} / {100*validation['fused_metrics']['macro_f1']:.4f}")
        print(f"Damaged net    : {complement['per_class']['damaged']['net_complement']}")
        print(f"Overall net    : {complement['net_complement']}")
        print(f"Nonneg events  : {validation['nonnegative_events_vs_p3']}/14")
        print(f"Checks         : {payload['checks']}")
    print(f"Decision       : {payload['decision']}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
