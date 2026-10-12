# -*- coding: utf-8 -*-
"""CMR-V1B: Top-20 counterfactual set-to-set case verification.

For every frozen-P3 Intact proposal, score Intact and Damaged historical cases
inside the SAME Raw-Mean Top-K pool.  The key signal is not an absolute
Damaged score but the counterfactual difference:

    local evidence(Damaged supports) - local evidence(Intact supports)

The audit remains Train/Validation only and never fuses predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from model.cmr_v1.set_matcher import class_logmeanexp, pair_score
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
    if payload.get("protocol") != "CMR_V1_ISOLATED_counterfactual_local_token_cache_v2":
        raise RuntimeError("V1B refuses an unknown token cache")
    layout = payload.get("storage_layout", "padded")
    if layout not in {"padded", "packed_list"}:
        raise RuntimeError(f"unknown V1 token storage layout: {layout}")
    if payload.get("role") != role or bool(payload.get("test_accessed", True)):
        raise RuntimeError(f"V1 cache role/Test contract failed for {path}")
    return payload


def _support_lookup(support: Mapping[str, object], maximum_index: int) -> torch.Tensor:
    memory_local = support["memory_local_indices"].long()
    size = max(maximum_index + 1, int(memory_local.max()) + 1)
    lookup = torch.full((size,), -1, dtype=torch.long)
    lookup[memory_local] = torch.arange(memory_local.numel())
    return lookup


def build_features(
    query: Mapping[str, object],
    support: Mapping[str, object],
    *,
    method: str,
    device: str,
    class_temperature: float,
    neighbor_k: int,
    epsilon: float,
    iterations: int,
    ridge: float,
) -> Dict[str, object]:
    # V1 is an Intact-vs-Damaged rescue audit.  Destroyed rows can exist because
    # proposal admission is label-free; exclude them only here for evaluation.
    auditable = (query["labels"].long() == 0) | (query["labels"].long() == 1)
    rows = torch.nonzero(auditable, as_tuple=False).flatten()
    if rows.numel() == 0:
        raise RuntimeError("V1B has no Intact/Damaged auditable proposals")

    max_candidate = int(query["candidate_indices"].max())
    lookup = _support_lookup(support, max_candidate)
    run_device = torch.device(device)

    local_rows = []
    coarse_rows = []
    context_rows = []
    best_case_rows = []
    skipped_missing_support = 0
    skipped_no_intact_counterfactual = 0
    for progress, row in enumerate(rows.tolist(), 1):
        q_tokens = query["tokens"][row].to(run_device)
        q_mask = query["token_mask"][row].to(run_device)
        candidate_indices = query["candidate_indices"][row].long()
        candidate_labels = query["candidate_labels"][row].long()
        candidate_scores = query["candidate_scores"][row].float()

        local_class: Dict[int, list[torch.Tensor]] = {0: [], 1: []}
        coarse_class: Dict[int, list[torch.Tensor]] = {0: [], 1: []}
        best_case = {0: (-float("inf"), -1, -1), 1: (-float("inf"), -1, -1)}
        for rank, (memory_index, label, coarse_score) in enumerate(
            zip(candidate_indices.tolist(), candidate_labels.tolist(), candidate_scores.tolist()), 1
        ):
            if int(label) not in (0, 1):
                continue
            if memory_index >= lookup.numel() or int(lookup[memory_index]) < 0:
                # A region that failed the strict one-time identity freeze is
                # quarantined.  Never substitute another component.
                continue
            support_row = int(lookup[memory_index])
            score = pair_score(
                method,
                q_tokens,
                q_mask,
                support["tokens"][support_row].to(run_device),
                support["token_mask"][support_row].to(run_device),
                neighbor_k=neighbor_k,
                epsilon=epsilon,
                iterations=iterations,
                ridge=ridge,
            ).detach().cpu()
            local_class[int(label)].append(score)
            coarse_class[int(label)].append(torch.tensor(float(coarse_score)))
            if float(score) > best_case[int(label)][0]:
                best_case[int(label)] = (float(score), int(rank), int(memory_index))

        if not local_class[1]:
            skipped_missing_support += 1
            continue
        # It is possible (though unusual) that no Intact case is in Top-K.  For a
        # counterfactual audit skip such rows rather than inventing an Intact case.
        if not local_class[0]:
            skipped_no_intact_counterfactual += 1
            continue

        i_local = torch.stack(local_class[0])
        d_local = torch.stack(local_class[1])
        i_coarse = torch.stack(coarse_class[0])
        d_coarse = torch.stack(coarse_class[1])
        s_i = class_logmeanexp(i_local, class_temperature)
        s_d = class_logmeanexp(d_local, class_temperature)
        c_i = class_logmeanexp(i_coarse, class_temperature)
        c_d = class_logmeanexp(d_coarse, class_temperature)
        local_rows.append(torch.tensor([
            float(s_i), float(s_d), float(s_d - s_i),
            float(i_local.max()), float(d_local.max()),
            float(d_local.max() - i_local.max()),
        ]))
        coarse_rows.append(torch.tensor([
            float(c_i), float(c_d), float(c_d - c_i),
            float(i_coarse.max()), float(d_coarse.max()),
            float(d_coarse.max() - i_coarse.max()),
            float(len(d_local)) / float(len(d_local) + len(i_local)),
        ]))
        probability = query["base_probabilities"][row].float()
        sorted_probability = probability.sort(descending=True).values
        context_rows.append(torch.tensor([
            1.0 - float(sorted_probability[0] - sorted_probability[1]),
            float(probability[1:].sum()),
            float(best_case[1][1]) / float(query["candidate_indices"].shape[1]),
        ]))
        best_case_rows.append({
            "query_row": int(row),
            "best_intact_rank": best_case[0][1],
            "best_intact_memory_index": best_case[0][2],
            "best_damaged_rank": best_case[1][1],
            "best_damaged_memory_index": best_case[1][2],
        })
        if progress % 100 == 0 or progress == rows.numel():
            print(f"CMR-V1B {method}: {progress}/{rows.numel()}", flush=True)

    # Because rows with no Intact support are skipped, recover their original
    # indices from best_case_rows for labels/events.
    kept_rows = torch.tensor([item["query_row"] for item in best_case_rows], dtype=torch.long)
    if not local_rows:
        raise RuntimeError("no query retained both Intact and Damaged frozen supports")
    local = torch.stack(local_rows)
    coarse = torch.stack(coarse_rows)
    context = torch.stack(context_rows)
    target = query["labels"][kept_rows].long() == 1
    events = [query["events"][row] for row in kept_rows.tolist()]
    direct_prediction = local[:, 2] > 0
    return {
        "target": target,
        "events": events,
        "kept_query_rows": kept_rows,
        "best_cases": best_case_rows,
        "availability": {
            "auditable_rows": int(rows.numel()),
            "retained_rows": int(kept_rows.numel()),
            "skipped_missing_damaged_support": int(skipped_missing_support),
            "skipped_no_intact_counterfactual": int(skipped_no_intact_counterfactual),
        },
        "direct_set_decision_accuracy": float((direct_prediction == target).float().mean()),
        "groups": {
            "v1_a_p3_rawmean": {
                "features": torch.cat((coarse, context), dim=1),
                "names": [
                    "coarse_intact", "coarse_damaged", "coarse_delta_d_i",
                    "coarse_intact_max", "coarse_damaged_max", "coarse_max_delta_d_i",
                    "damaged_candidate_fraction",
                    "p3_uncertainty", "p3_affected_probability", "best_damaged_rank_norm",
                ],
            },
            "v1_b_set_only": {
                "features": local,
                "names": [
                    "set_intact", "set_damaged", "set_delta_d_i",
                    "set_intact_max", "set_damaged_max", "set_max_delta_d_i",
                ],
            },
            "v1_c_counterfactual_combined": {
                "features": torch.cat((coarse, context, local), dim=1),
                "names": [
                    "coarse_intact", "coarse_damaged", "coarse_delta_d_i",
                    "coarse_intact_max", "coarse_damaged_max", "coarse_max_delta_d_i",
                    "damaged_candidate_fraction",
                    "p3_uncertainty", "p3_affected_probability", "best_damaged_rank_norm",
                    "set_intact", "set_damaged", "set_delta_d_i",
                    "set_intact_max", "set_damaged_max", "set_max_delta_d_i",
                ],
            },
        },
    }


def evaluate_group(
    train: Mapping[str, object], validation: Mapping[str, object],
    train_target: torch.Tensor, validation_target: torch.Tensor,
    train_events: list[str], validation_events: list[str],
    *, coverages: list[float], l2: float,
) -> Dict[str, object]:
    crossfit = cross_fitted_logistic_scores(
        train["features"], train_target, train_events,
        l2=l2, balance_classes=False,
    )
    oof_score = crossfit["scores"]
    train_rows = risk_coverage_table(oof_score, train_target, train_events, coverages=coverages)
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


def main() -> None:
    parser = argparse.ArgumentParser("CMR-V1 counterfactual set verifier")
    parser.add_argument("--support", required=True)
    parser.add_argument("--train_queries", required=True)
    parser.add_argument("--validation_queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("chamfer", "dn4", "emd", "frn"), default="emd")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--class_temperature", type=float, default=0.10)
    parser.add_argument("--neighbor_k", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.07)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--coverages", default="0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--logistic_l2", type=float, default=0.01)
    args = parser.parse_args()

    support = load_cache(args.support, "support")
    train_query = load_cache(args.train_queries, "train_query")
    validation_query = load_cache(args.validation_queries, "validation_query")
    hashes = {item["p3_checkpoint_sha256"] for item in (support, train_query, validation_query)}
    if len(hashes) != 1:
        raise RuntimeError("V1 caches use different P3 checkpoints")
    validate_disjoint_ids(train_query["image_ids"], validation_query["image_ids"])

    kwargs = dict(
        method=args.method, device=args.device,
        class_temperature=args.class_temperature,
        neighbor_k=args.neighbor_k, epsilon=args.epsilon,
        iterations=args.iterations, ridge=args.ridge,
    )
    train_table = build_features(train_query, support, **kwargs)
    validation_table = build_features(validation_query, support, **kwargs)
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
    baseline_auc = float(results["v1_a_p3_rawmean"]["validation"]["auroc"])
    combined = results["v1_c_counterfactual_combined"]["validation"]
    combined_auc = float(combined["auroc"])
    point = combined["out_of_fold_locked_operating_point"]
    point_passed = bool(
        point is not None
        and point["precision"] >= 0.55
        and point["net_correction"] > 0
        and point["rescued"] >= 10
        and point["nonnegative_events"] >= 10
    )
    checks = {
        "at_least_100_validation_proposals": validation_table["target"].numel() >= 100,
        "combined_auroc_at_least_0_75": combined_auc >= 0.75,
        "set_matching_adds_at_least_0_05_auc": combined_auc - baseline_auc >= 0.05,
        "crossfit_locked_positive_validation_point": point_passed,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "CMR_V1_ISOLATED_top20_counterfactual_set_verifier_v1",
        "method": args.method,
        "scope": "frozen P3 + frozen Raw-Mean Top-K + frozen local tokens; no fusion",
        "train_proposals": int(train_table["target"].numel()),
        "validation_proposals": int(validation_table["target"].numel()),
        "train_direct_set_decision_accuracy": train_table["direct_set_decision_accuracy"],
        "validation_direct_set_decision_accuracy": validation_table["direct_set_decision_accuracy"],
        "results": results,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_SET_MATCHER_TO_P3_ANCHORED_RESCUE_SCREEN_DO_NOT_TEST"
            if passed else "STOP_OR_UPGRADE_SET_MATCHING_DO_NOT_FUSE"
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print(f"CMR-V1 counterfactual set verifier | method={args.method}")
    for name, row in results.items():
        print(
            f"{name:<31} | AUC={row['validation']['auroc']:.4f} | "
            f"event={row['validation']['event_macro_auroc']:.4f} | "
            f"locked={row['validation']['out_of_fold_locked_operating_point']}"
        )
    print(f"Checks          : {checks}")
    print(f"Decision        : {payload['decision']}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()
