from __future__ import annotations

import torch

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    per_image_selective_histograms,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram


ARMS = ("b0_fixed_s2", "b1_benefit_only", "b2_harm_only", "b3_dual")


def arm_acceptance(
    arm: str,
    metric_scores: torch.Tensor,
    harm_scores: torch.Tensor,
    metric_threshold: float,
    harm_threshold: float,
) -> torch.Tensor:
    if arm == "b0_fixed_s2":
        return torch.ones_like(metric_scores, dtype=torch.bool)
    if arm == "b1_benefit_only":
        return metric_scores >= float(metric_threshold)
    if arm == "b2_harm_only":
        return harm_scores < float(harm_threshold)
    if arm == "b3_dual":
        return (metric_scores >= float(metric_threshold)) & (
            harm_scores < float(harm_threshold)
        )
    raise ValueError(f"unknown V2.1 arm: {arm}")


def threshold_pairs(arm: str) -> list[tuple[float, float]]:
    grid = [float(value) for value in torch.linspace(0.30, 0.70, 9)]
    if arm == "b1_benefit_only":
        return [(metric, 1.0) for metric in grid]
    if arm == "b2_harm_only":
        return [(0.0, harm) for harm in grid]
    if arm == "b3_dual":
        return [(metric, harm) for metric in grid for harm in grid]
    if arm == "b0_fixed_s2":
        return [(0.0, 1.0)]
    raise ValueError(f"unknown V2.1 arm: {arm}")


def select_arm_policy(
    arm: str,
    cache: dict,
    metric_scores: torch.Tensor,
    harm_scores: torch.Tensor,
    calibration_cluster_mask: torch.Tensor,
    calibration_image_mask: torch.Tensor,
    *, maximum_reject_fraction: float = 0.70,
) -> tuple[dict, dict]:
    fixed = metrics_from_histogram(
        cache["fixed_s2_histograms"][calibration_image_mask].sum(0)
    )
    fallback = {
        "metric_threshold": 0.0,
        "harm_threshold": 1.0,
        "reject_fraction": 0.0,
        "miou": fixed["valid_miou"],
        "damaged_f1": fixed["damaged_f1"],
        "fallback_fixed_s2": True,
    }
    diagnostics = {
        "arm": arm,
        "candidate_total": 0,
        "nontrivial_candidates": 0,
        "over_reject_cap": 0,
        "damaged_f1_below_fixed": 0,
        "miou_not_above_fixed": 0,
        "safe_nontrivial_candidates": 0,
        "useful_nontrivial_candidates": 0,
    }
    if arm == "b0_fixed_s2":
        diagnostics["selected_fallback_fixed_s2"] = True
        return fallback, diagnostics

    safe_candidates = []
    local_metric = metric_scores[calibration_cluster_mask]
    local_harm = harm_scores[calibration_cluster_mask]
    for metric_threshold, harm_threshold in threshold_pairs(arm):
        diagnostics["candidate_total"] += 1
        local_accept = arm_acceptance(
            arm, local_metric, local_harm, metric_threshold, harm_threshold
        )
        reject_fraction = float((~local_accept).float().mean())
        nontrivial = reject_fraction > 0
        diagnostics["nontrivial_candidates"] += int(nontrivial)
        if reject_fraction > maximum_reject_fraction:
            diagnostics["over_reject_cap"] += 1
            continue
        accept = torch.ones(len(metric_scores), dtype=torch.bool)
        accept[calibration_cluster_mask] = local_accept
        selected_by_image = per_image_selective_histograms(cache, accept)
        report = metrics_from_histogram(
            selected_by_image[calibration_image_mask].sum(0)
        )
        damaged_unsafe = report["damaged_f1"] < fixed["damaged_f1"]
        not_better = report["valid_miou"] <= fixed["valid_miou"]
        diagnostics["damaged_f1_below_fixed"] += int(damaged_unsafe)
        diagnostics["miou_not_above_fixed"] += int(not_better)
        if damaged_unsafe or not nontrivial:
            continue
        diagnostics["safe_nontrivial_candidates"] += 1
        diagnostics["useful_nontrivial_candidates"] += int(not not_better)
        safe_candidates.append({
            "metric_threshold": metric_threshold,
            "harm_threshold": harm_threshold,
            "reject_fraction": reject_fraction,
            "miou": report["valid_miou"],
            "damaged_f1": report["damaged_f1"],
            "fallback_fixed_s2": False,
        })
    selected = max(
        [fallback, *safe_candidates],
        key=lambda row: (row["miou"], row["damaged_f1"], -row["reject_fraction"]),
    )
    diagnostics["selected_fallback_fixed_s2"] = selected["fallback_fixed_s2"]
    diagnostics["selected_policy"] = selected
    return selected, diagnostics


def apply_arm_policy(
    arm: str,
    metric_scores: torch.Tensor,
    harm_scores: torch.Tensor,
    policy: dict,
) -> torch.Tensor:
    return arm_acceptance(
        arm, metric_scores, harm_scores,
        policy["metric_threshold"], policy["harm_threshold"],
    )


__all__ = [
    "ARMS", "apply_arm_policy", "arm_acceptance", "select_arm_policy",
    "threshold_pairs",
]
