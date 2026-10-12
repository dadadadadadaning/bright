# -*- coding: utf-8 -*-
"""CMR-V0: event-cross-fitted local-token rescue/harm verification audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from model.cmr.local_verifier import LOCAL_CORRELATION_NAMES, token_correlation_statistics
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
    if payload.get("protocol") != "CMR_V0_proposal_local_token_cache_v1":
        raise RuntimeError("V0 refuses an unknown local-token cache")
    if payload.get("role") != role or bool(payload.get("test_accessed", True)):
        raise RuntimeError(f"V0 cache role/Test contract failed for {path}")
    return payload


def proposal_feature_groups(query: Mapping[str, object], support: Mapping[str, object]) -> Dict[str, object]:
    memory_local = support["memory_local_indices"].long()
    lookup_size = max(
        int(memory_local.max()) + 1,
        int(query["candidate_indices"].max()) + 1,
    )
    lookup = torch.full((lookup_size,), -1, dtype=torch.long)
    lookup[memory_local] = torch.arange(memory_local.numel())
    local_rows = []
    context_rows = []
    for row in range(query["tokens"].shape[0]):
        damaged = query["candidate_labels"][row].long() == 1
        candidate_local = query["candidate_indices"][row].long()[damaged]
        candidate_score = query["candidate_scores"][row].float()
        support_rows = lookup[candidate_local]
        valid = support_rows >= 0
        support_rows = support_rows[valid]
        if support_rows.numel() == 0:
            raise RuntimeError("V0 proposal has no materialised Damaged support tokens")
        q = query["tokens"][row].float()[None].expand(support_rows.numel(), -1, -1)
        qm = query["token_mask"][row][None].expand(support_rows.numel(), -1)
        pair = token_correlation_statistics(
            q, qm,
            support["tokens"][support_rows].float(), support["token_mask"][support_rows],
        )
        local_rows.append(torch.cat((pair.mean(dim=0), pair.max(dim=0).values, pair.std(dim=0, unbiased=False))))

        probability = query["base_probabilities"][row].float()
        sorted_probability = probability.sort(descending=True).values
        d_score = candidate_score[query["candidate_labels"][row] == 1]
        non_d_score = candidate_score[query["candidate_labels"][row] != 1]
        support_events = [support["events"][index] for index in support_rows.tolist()]
        context_rows.append(torch.tensor([
            1.0 - float(sorted_probability[0] - sorted_probability[1]),
            float(probability[1:].sum()),
            float(d_score.max()),
            float(d_score.max() - non_d_score.max()) if non_d_score.numel() else 0.0,
            float(damaged.float().mean()),
            float(len(set(support_events))) / max(int(support_rows.numel()), 1),
        ]))
    local = torch.stack(local_rows)
    context = torch.stack(context_rows)
    return {
        "target": query["labels"].long() == 1,
        "events": list(map(str, query["events"])),
        "groups": {
            "v0_a_p3_mean": {
                "features": context,
                "names": [
                    "p3_uncertainty", "p3_affected_probability", "mean_damaged_score",
                    "mean_damaged_margin", "damaged_neighbor_fraction", "damaged_event_diversity",
                ],
            },
            "v0_b_local_tokens": {
                "features": local,
                "names": [f"{aggregate}_{name}" for aggregate in ("mean", "max", "std") for name in LOCAL_CORRELATION_NAMES],
            },
            "v0_c_combined": {
                "features": torch.cat((context, local), dim=1),
                "names": [
                    "p3_uncertainty", "p3_affected_probability", "mean_damaged_score",
                    "mean_damaged_margin", "damaged_neighbor_fraction", "damaged_event_diversity",
                ] + [f"{aggregate}_{name}" for aggregate in ("mean", "max", "std") for name in LOCAL_CORRELATION_NAMES],
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
    parser = argparse.ArgumentParser("CMR-V0 local-token verifier audit")
    parser.add_argument("--support", required=True)
    parser.add_argument("--train_queries", required=True)
    parser.add_argument("--validation_queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverages", default="0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--logistic_l2", type=float, default=0.01)
    args = parser.parse_args()

    support = load_cache(args.support, "support")
    train_query = load_cache(args.train_queries, "train_query")
    validation_query = load_cache(args.validation_queries, "validation_query")
    hashes = {item["p3_checkpoint_sha256"] for item in (support, train_query, validation_query)}
    if len(hashes) != 1:
        raise RuntimeError("V0 caches use different P3 checkpoints")
    validate_disjoint_ids(train_query["image_ids"], validation_query["image_ids"])
    train_table = proposal_feature_groups(train_query, support)
    validation_table = proposal_feature_groups(validation_query, support)
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
    baseline_auc = float(results["v0_a_p3_mean"]["validation"]["auroc"])
    combined = results["v0_c_combined"]["validation"]
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
        "combined_auroc_at_least_0_70": combined_auc >= 0.70,
        "local_tokens_add_at_least_0_05_auc": combined_auc - baseline_auc >= 0.05,
        "crossfit_locked_positive_validation_point": point_passed,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "CMR_V0_mean_recall_local_token_verifier_crossfit_v1",
        "scope": "frozen Mean candidates + frozen P3/MICB local tokens; no prediction fusion",
        "train_proposals": int(train_table["target"].numel()),
        "validation_proposals": int(validation_table["target"].numel()),
        "results": results,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_LOCAL_VERIFIER_TO_P3_ANCHORED_RESCUE_SCREEN_DO_NOT_TEST"
            if passed else "STOP_OR_UPGRADE_LOCAL_DESCRIPTOR_DO_NOT_FUSE"
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print("CMR-V0 Mean-recall + local-token verification audit")
    for name, row in results.items():
        print(
            f"{name:<20} | AUC={row['validation']['auroc']:.4f} | "
            f"event={row['validation']['event_macro_auroc']:.4f} | "
            f"locked={row['validation']['out_of_fold_locked_operating_point']}"
        )
    print(f"Checks          : {checks}")
    print(f"Decision        : {payload['decision']}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()

