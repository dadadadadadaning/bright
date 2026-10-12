# -*- coding: utf-8 -*-
"""CMR-C0: frozen critical-region counterfactual verification audit.

The only experimental variable is token selection over the same frozen
``fused_roi`` case card.  P3, regions, RawMean Top-20 and support identities are
held fixed.  Variant selection and thresholds use Train only; Test is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from model.cmr_c0.case_card import (
    CRITICAL_VARIANTS,
    class_logmeanexp,
    directional_variant_scores,
    validate_case_cache,
)
from script.standard_ML.im1_matching_reliability_runtime import (
    apply_logistic_calibrator,
    cross_fitted_logistic_scores,
    event_macro_auroc,
    fit_logistic_calibrator,
    operating_point_at_threshold,
    risk_coverage_table,
    select_train_operating_point,
    serialise_calibrator,
)
from script.standard_ML.rm_region_retrieval_runtime import binary_auroc, validate_disjoint_ids


def load_cache(path: str, role: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_case_cache(payload, role)
    return payload


def _support_lookup(support: Mapping[str, object], maximum_index: int) -> torch.Tensor:
    memory_local = support["memory_local_indices"].long()
    size = max(maximum_index + 1, int(memory_local.max()) + 1)
    lookup = torch.full((size,), -1, dtype=torch.long)
    lookup[memory_local] = torch.arange(memory_local.numel())
    return lookup


def _local_features(
    intact_scores: Sequence[torch.Tensor],
    damaged_scores: Sequence[torch.Tensor],
    class_temperature: float,
) -> torch.Tensor:
    intact = torch.stack(list(intact_scores)).float()
    damaged = torch.stack(list(damaged_scores)).float()
    s_i = class_logmeanexp(intact, class_temperature)
    s_d = class_logmeanexp(damaged, class_temperature)
    return torch.tensor([
        float(s_i), float(s_d), float(s_d - s_i),
        float(intact.max()), float(damaged.max()),
        float(damaged.max() - intact.max()),
    ])


def build_feature_table(
    query: Mapping[str, object],
    support: Mapping[str, object],
    *,
    device: str,
    neighbor_k: int,
    class_temperature: float,
) -> Dict[str, object]:
    auditable = (query["labels"].long() == 0) | (query["labels"].long() == 1)
    rows = torch.nonzero(auditable, as_tuple=False).flatten()
    if rows.numel() == 0:
        raise RuntimeError("CMR-C0 has no Intact/Damaged auditable proposals")
    lookup = _support_lookup(support, int(query["candidate_indices"].max()))
    run_device = torch.device(device)

    local_rows: Dict[str, list[torch.Tensor]] = {
        variant: [] for variant in CRITICAL_VARIANTS
    }
    coarse_rows: list[torch.Tensor] = []
    context_rows: list[torch.Tensor] = []
    kept_rows: list[int] = []
    skipped_missing_damaged = 0
    skipped_missing_intact = 0

    for progress, row in enumerate(rows.tolist(), 1):
        candidate_indices = query["candidate_indices"][row].long()
        candidate_labels = query["candidate_labels"][row].long()
        candidate_scores = query["candidate_scores"][row].float()
        local: Dict[str, Dict[int, list[torch.Tensor]]] = {
            variant: {0: [], 1: []} for variant in CRITICAL_VARIANTS
        }
        coarse: Dict[int, list[torch.Tensor]] = {0: [], 1: []}
        best_damaged_rank = -1
        best_damaged_coarse = -float("inf")

        query_features = query["features"][row].to(run_device)
        query_indices = {
            variant: query["critical_indices"][variant][row]
            for variant in CRITICAL_VARIANTS
        }
        for rank, (memory_index, label, coarse_score) in enumerate(
            zip(candidate_indices.tolist(), candidate_labels.tolist(), candidate_scores.tolist()), 1
        ):
            label = int(label)
            if label not in (0, 1):
                continue
            if memory_index >= lookup.numel() or int(lookup[memory_index]) < 0:
                continue
            support_row = int(lookup[memory_index])
            pair_scores = directional_variant_scores(
                query_features,
                support["features"][support_row].to(run_device),
                query_indices,
                {
                    variant: support["critical_indices"][variant][support_row]
                    for variant in CRITICAL_VARIANTS
                },
                neighbor_k=neighbor_k,
            )
            for variant, score in pair_scores.items():
                local[variant][label].append(score.detach().cpu())
            coarse[label].append(torch.tensor(float(coarse_score)))
            if label == 1 and float(coarse_score) > best_damaged_coarse:
                best_damaged_coarse = float(coarse_score)
                best_damaged_rank = rank

        if not coarse[1]:
            skipped_missing_damaged += 1
            continue
        if not coarse[0]:
            skipped_missing_intact += 1
            continue
        if any(not local[variant][0] or not local[variant][1] for variant in CRITICAL_VARIANTS):
            raise RuntimeError("critical variants lost support alignment")

        intact_coarse = torch.stack(coarse[0])
        damaged_coarse = torch.stack(coarse[1])
        c_i = class_logmeanexp(intact_coarse, class_temperature)
        c_d = class_logmeanexp(damaged_coarse, class_temperature)
        coarse_rows.append(torch.tensor([
            float(c_i), float(c_d), float(c_d - c_i),
            float(intact_coarse.max()), float(damaged_coarse.max()),
            float(damaged_coarse.max() - intact_coarse.max()),
            float(len(damaged_coarse)) / float(len(damaged_coarse) + len(intact_coarse)),
        ]))
        probability = query["base_probabilities"][row].float()
        sorted_probability = probability.sort(descending=True).values
        context_rows.append(torch.tensor([
            1.0 - float(sorted_probability[0] - sorted_probability[1]),
            float(probability[1:].sum()),
            float(best_damaged_rank) / float(candidate_indices.numel()),
        ]))
        for variant in CRITICAL_VARIANTS:
            local_rows[variant].append(_local_features(
                local[variant][0], local[variant][1], class_temperature
            ))
        kept_rows.append(row)
        if progress % 100 == 0 or progress == rows.numel():
            print(f"CMR-C0 retrieval: {progress}/{rows.numel()}", flush=True)

    if not kept_rows:
        raise RuntimeError("CMR-C0 retained no proposal with both I and D support")
    kept = torch.tensor(kept_rows, dtype=torch.long)
    coarse_tensor = torch.stack(coarse_rows)
    context_tensor = torch.stack(context_rows)
    base = torch.cat((coarse_tensor, context_tensor), dim=1)
    target = query["labels"][kept].long() == 1
    events = [str(query["events"][row]) for row in kept.tolist()]
    disaster_types = [str(query["disaster_types"][row]) for row in kept.tolist()]

    base_names = [
        "rawmean_intact", "rawmean_damaged", "rawmean_delta_d_i",
        "rawmean_intact_max", "rawmean_damaged_max", "rawmean_max_delta_d_i",
        "damaged_candidate_fraction", "p3_uncertainty",
        "p3_affected_probability", "best_damaged_rank_norm",
    ]
    local_names = [
        "local_intact", "local_damaged", "local_delta_d_i",
        "local_intact_max", "local_damaged_max", "local_max_delta_d_i",
    ]
    groups: Dict[str, Dict[str, object]] = {
        "rawmean_p3": {"features": base, "names": base_names}
    }
    direct: Dict[str, object] = {}
    top1_not_damaged = query["candidate_labels"][kept, 0].long() != 1
    unused_headroom = target & top1_not_damaged
    for variant in CRITICAL_VARIANTS:
        local_tensor = torch.stack(local_rows[variant])
        groups[f"{variant}_set_only"] = {
            "features": local_tensor,
            "names": local_names,
        }
        groups[f"{variant}_combined"] = {
            "features": torch.cat((base, local_tensor), dim=1),
            "names": base_names + local_names,
        }
        delta = local_tensor[:, 2]
        predicts_damaged = delta > 0
        direct[variant] = {
            "delta_auroc": binary_auroc(delta, target),
            **event_macro_auroc(delta, target, events),
            "decision_accuracy": float((predicts_damaged == target).float().mean()),
            "rescue_exposure": {
                "count": int(target.sum()),
                "positive_delta": int(predicts_damaged[target].sum()),
                "rate": float(predicts_damaged[target].float().mean()),
            },
            "intact_harm_exposure": {
                "count": int((~target).sum()),
                "positive_delta": int(predicts_damaged[~target].sum()),
                "rate": float(predicts_damaged[~target].float().mean()),
            },
            "unused_d_headroom": {
                "count": int(unused_headroom.sum()),
                "positive_delta": int(predicts_damaged[unused_headroom].sum()),
                "rate": (
                    float(predicts_damaged[unused_headroom].float().mean())
                    if bool(unused_headroom.any()) else None
                ),
            },
        }
    return {
        "target": target,
        "events": events,
        "disaster_types": disaster_types,
        "kept_query_rows": kept,
        "groups": groups,
        "direct": direct,
        "availability": {
            "auditable_rows": int(rows.numel()),
            "retained_rows": int(kept.numel()),
            "skipped_missing_damaged_support": int(skipped_missing_damaged),
            "skipped_no_intact_counterfactual": int(skipped_missing_intact),
        },
    }


def evaluate_group(
    train: Mapping[str, object],
    validation: Mapping[str, object],
    train_target: torch.Tensor,
    validation_target: torch.Tensor,
    train_events: list[str],
    validation_events: list[str],
    *,
    coverages: Sequence[float],
    l2: float,
) -> Dict[str, object]:
    crossfit = cross_fitted_logistic_scores(
        train["features"], train_target, train_events,
        l2=l2, balance_classes=False,
    )
    oof_score = crossfit["scores"]
    train_rows = risk_coverage_table(
        oof_score, train_target, train_events, coverages=coverages
    )
    selected = select_train_operating_point(train_rows)
    threshold = None if selected is None else float(selected["score_threshold"])
    final_model = fit_logistic_calibrator(
        train["features"], train_target, l2=l2, balance_classes=False
    )
    validation_score = apply_logistic_calibrator(validation["features"], final_model)
    locked = None if threshold is None else operating_point_at_threshold(
        validation_score, validation_target, validation_events, threshold=threshold
    )
    return {
        "signal_names": list(train["names"]),
        "crossfit_folds": crossfit["folds"],
        "out_of_fold_train": {
            "auroc": binary_auroc(oof_score, train_target),
            **event_macro_auroc(oof_score, train_target, train_events),
            "operating_points": train_rows,
            "selected_operating_point": selected,
        },
        "final_calibrator": serialise_calibrator(final_model),
        "validation": {
            "auroc": binary_auroc(validation_score, validation_target),
            **event_macro_auroc(validation_score, validation_target, validation_events),
            "out_of_fold_locked_operating_point": locked,
        },
    }


def _token_statistics(cache: Mapping[str, object]) -> Dict[str, object]:
    count = cache["token_count"].float()
    variants: Dict[str, object] = {}
    for variant in CRITICAL_VARIANTS:
        selected = torch.tensor([
            int(index.numel()) for index in cache["critical_indices"][variant]
        ], dtype=torch.float32)
        variants[variant] = {
            "mean": float(selected.mean()),
            "median": float(selected.median()),
            "minimum": int(selected.min()),
            "maximum": int(selected.max()),
            "fraction_of_all": float((selected / count).mean()),
            "nms_backfill_total": int(cache["nms_backfill"][variant].sum()),
        }
    return {
        "cases": int(count.numel()),
        "all_token_mean": float(count.mean()),
        "all_token_median": float(count.median()),
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-C0 frozen critical-region verifier")
    parser.add_argument("--support", required=True)
    parser.add_argument("--train_queries", required=True)
    parser.add_argument("--validation_queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--neighbor_k", type=int, default=3)
    parser.add_argument("--class_temperature", type=float, default=0.10)
    parser.add_argument("--coverages", default="0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--logistic_l2", type=float, default=0.01)
    args = parser.parse_args()

    support = load_cache(args.support, "support")
    train_query = load_cache(args.train_queries, "train_query")
    validation_query = load_cache(args.validation_queries, "validation_query")
    hashes = {item["p3_checkpoint_sha256"] for item in (support, train_query, validation_query)}
    if len(hashes) != 1:
        raise RuntimeError("CMR-C0 caches use different P3 checkpoints")
    validate_disjoint_ids(train_query["image_ids"], validation_query["image_ids"])
    if int(train_query["candidate_indices"].shape[1]) != 20 or int(validation_query["candidate_indices"].shape[1]) != 20:
        raise RuntimeError("CMR-C0 freezes the experiment to RawMean Top-20")

    common = {
        "support": support, "device": args.device,
        "neighbor_k": args.neighbor_k,
        "class_temperature": args.class_temperature,
    }
    train_table = build_feature_table(train_query, **common)
    validation_table = build_feature_table(validation_query, **common)
    coverages = [float(value) for value in args.coverages.split(",") if value.strip()]
    results = {
        name: evaluate_group(
            train_table["groups"][name], validation_table["groups"][name],
            train_table["target"], validation_table["target"],
            train_table["events"], validation_table["events"],
            coverages=coverages, l2=args.logistic_l2,
        )
        for name in train_table["groups"]
    }

    # This is a Train-only model-selection decision.  Validation never chooses
    # the critical ratio; it only estimates the locked choice.
    selected_variant = max(
        CRITICAL_VARIANTS,
        key=lambda variant: float(
            results[f"{variant}_combined"]["out_of_fold_train"]["auroc"]
        ),
    )
    baseline = results["all_combined"]["validation"]
    selected_combined = results[f"{selected_variant}_combined"]["validation"]
    baseline_set = results["all_set_only"]["validation"]
    selected_set = results[f"{selected_variant}_set_only"]["validation"]
    point = selected_combined["out_of_fold_locked_operating_point"]
    point_passed = bool(
        point is not None
        and point["precision"] >= 0.55
        and point["net_correction"] > 0
        and point["rescued"] >= 10
        and point["nonnegative_events"] >= 10
    )
    checks = {
        "at_least_100_validation_proposals": validation_table["target"].numel() >= 100,
        "train_selected_variant_is_critical_not_all": selected_variant != "all",
        "critical_combined_adds_at_least_0_05_validation_auc": (
            float(selected_combined["auroc"]) - float(baseline["auroc"]) >= 0.05
        ),
        "critical_set_only_adds_at_least_0_05_validation_auc": (
            float(selected_set["auroc"]) - float(baseline_set["auroc"]) >= 0.05
        ),
        "event_macro_does_not_decrease": (
            float(selected_combined["event_macro_auroc"])
            >= float(baseline["event_macro_auroc"])
        ),
        "crossfit_locked_positive_validation_point": point_passed,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "CMR_C0_ISOLATED_frozen_critical_verifier_v1",
        "scope": "frozen P3 + frozen component + frozen RawMean Top20; token selection only",
        "retrieval": {
            "direction": "query_critical_tokens_to_memory_critical_tokens",
            "local_neighbor_k": int(args.neighbor_k),
            "class_aggregation": "temperature_logmeanexp",
            "class_temperature": float(args.class_temperature),
            "counterfactual": "S_D_minus_S_I",
        },
        "selection_split": "train leave-one-incident-out only",
        "selected_variant": selected_variant,
        "train_availability": train_table["availability"],
        "validation_availability": validation_table["availability"],
        "train_direct_diagnostics": train_table["direct"],
        "validation_direct_diagnostics": validation_table["direct"],
        "case_card_statistics": {
            "support": _token_statistics(support),
            "train_query": _token_statistics(train_query),
            "validation_query": _token_statistics(validation_query),
        },
        "results": results,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_CRITICAL_REGION_REPRESENTATION_TO_NEXT_MEMORY_STAGE_DO_NOT_TEST"
            if passed else "STOP_AT_C0_CRITICAL_REGION_REPRESENTATION_INSUFFICIENT_DO_NOT_FUSE"
        ),
        "p3_changed": False,
        "rawmean_top20_changed": False,
        "uses_disaster_type_for_scoring": False,
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 112)
    print("CMR-C0 frozen critical-region verifier")
    print(f"Train-selected variant: {selected_variant}")
    for variant in CRITICAL_VARIANTS:
        combined = results[f"{variant}_combined"]["validation"]
        set_only = results[f"{variant}_set_only"]["validation"]
        harm = validation_table["direct"][variant]["intact_harm_exposure"]["rate"]
        headroom = validation_table["direct"][variant]["unused_d_headroom"]["rate"]
        print(
            f"{variant:<16} combined={combined['auroc']:.4f} "
            f"set={set_only['auroc']:.4f} event={combined['event_macro_auroc']:.4f} "
            f"harm_exp={harm:.4f} unused_D={headroom}"
        )
    print(f"Checks       : {checks}")
    print(f"Decision     : {payload['decision']}")
    print(f"Saved        : {output}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
