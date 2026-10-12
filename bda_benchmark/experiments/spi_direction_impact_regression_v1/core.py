from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    per_image_selective_histograms,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram


VARIANTS = (
    "c1_continuous",
    "c2_direction",
    "c3_impact",
    "c4_direction_impact",
)
REJECTION_GRID = (0.0, .01, .02, .05, .10)


class DualUtilityRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.head(features)
        return values[:, 0], values[:, 1]


def uses_direction(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"unknown E2 variant: {variant}")
    return variant in {"c2_direction", "c4_direction_impact"}


def uses_impact(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"unknown E2 variant: {variant}")
    return variant in {"c3_impact", "c4_direction_impact"}


def direction_one_hot(direction_code: torch.Tensor) -> torch.Tensor:
    code = direction_code.long()
    if bool(((code < 0) | (code >= 16)).any()):
        raise ValueError("direction code must be in [0,15]")
    return F.one_hot(code, num_classes=16).float()


def target_scale(utility: torch.Tensor, selected: torch.Tensor) -> float:
    values = utility[selected].abs()
    values = values[values > 0]
    return float(torch.quantile(values.float(), .5).clamp_min(1e-8)) if len(values) else 1.0


def signed_log_transform(utility: torch.Tensor, scale: float) -> torch.Tensor:
    values = utility.float()
    return torch.sign(values) * torch.log1p(values.abs() / max(float(scale), 1e-8))


def event_balanced_weights(cache: dict, selected: torch.Tensor) -> torch.Tensor:
    image_ids = cache["image_index"].long()[selected]
    events = [cache["image_events"][int(index)] for index in image_ids]
    counts = Counter(events)
    weights = torch.tensor([1.0 / counts[event] for event in events], dtype=torch.float32)
    return weights / weights.mean().clamp_min(1e-8)


def impact_weights(
    metric_utility: torch.Tensor,
    damaged_utility: torch.Tensor,
    selected: torch.Tensor,
    metric_scale: float,
    damaged_scale: float,
    maximum: float,
) -> torch.Tensor:
    metric = metric_utility[selected].abs() / max(metric_scale, 1e-8)
    damaged = damaged_utility[selected].abs() / max(damaged_scale, 1e-8)
    impact = torch.maximum(metric, damaged).sqrt().clamp(0.25, float(maximum))
    return impact / impact.mean().clamp_min(1e-8)


def train_regressor(
    features: torch.Tensor,
    cache: dict,
    selected: torch.Tensor,
    metric_target: torch.Tensor,
    damaged_target: torch.Tensor,
    metric_utility: torch.Tensor,
    damaged_utility: torch.Tensor,
    *, impact: bool, metric_scale: float, damaged_scale: float,
    impact_cap: float, steps: int, batch_size: int,
    learning_rate: float, seed: int,
) -> DualUtilityRegressor:
    ids_all = torch.nonzero(selected, as_tuple=False).flatten()
    if not len(ids_all):
        raise RuntimeError("E2 fold has no training clusters")
    torch.manual_seed(seed)
    model = DualUtilityRegressor(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weights = event_balanced_weights(cache, selected)
    if impact:
        weights = weights * impact_weights(
            metric_utility, damaged_utility, selected,
            metric_scale, damaged_scale, impact_cap,
        )
        weights = weights / weights.mean().clamp_min(1e-8)
    generator = torch.Generator().manual_seed(seed + 1)
    size = min(int(batch_size), len(ids_all))
    for step in range(1, int(steps) + 1):
        local = torch.randint(len(ids_all), (size,), generator=generator)
        ids = ids_all[local]
        metric_prediction, damaged_prediction = model(features[ids])
        metric_loss = F.smooth_l1_loss(
            metric_prediction, metric_target[ids], reduction="none", beta=.5
        )
        damaged_loss = F.smooth_l1_loss(
            damaged_prediction, damaged_target[ids], reduction="none", beta=.5
        )
        loss = ((metric_loss + damaged_loss) * weights[local]).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite E2 loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite E2 gradient at step {step}")
        optimizer.step()
    return model


@torch.inference_mode()
def predict(model: DualUtilityRegressor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    return model(features)


def risk_score(metric_prediction: torch.Tensor, damaged_prediction: torch.Tensor) -> torch.Tensor:
    """Positive means stronger evidence that keeping S2 is harmful."""
    return torch.maximum(-metric_prediction, -damaged_prediction)


def choose_conservative_policy(
    cache: dict,
    risk: torch.Tensor,
    cluster_mask: torch.Tensor,
    image_mask: torch.Tensor,
    *, maximum_reject_fraction: float,
) -> tuple[dict, dict]:
    fixed = metrics_from_histogram(cache["fixed_s2_histograms"][image_mask].sum(0))
    fallback = {
        "risk_threshold": None, "target_reject_fraction": 0.0,
        "actual_reject_fraction": 0.0, "fallback_fixed_s2": True,
        "miou": fixed["valid_miou"], "damaged_f1": fixed["damaged_f1"],
    }
    diagnostics = {"candidate_total": 0, "over_reject_cap": 0,
                   "damaged_unsafe": 0, "miou_not_above_fixed": 0,
                   "safe_nontrivial": 0, "useful_nontrivial": 0}
    local_risk = risk[cluster_mask]
    candidates = []
    for fraction in REJECTION_GRID:
        if fraction == 0:
            continue
        diagnostics["candidate_total"] += 1
        threshold = float(torch.quantile(local_risk.float(), 1.0 - fraction))
        local_accept = local_risk < threshold
        reject_fraction = float((~local_accept).float().mean())
        if reject_fraction > maximum_reject_fraction + 1e-8:
            diagnostics["over_reject_cap"] += 1
            continue
        accepted = torch.ones(len(risk), dtype=torch.bool)
        accepted[cluster_mask] = local_accept
        selected = per_image_selective_histograms(cache, accepted)
        report = metrics_from_histogram(selected[image_mask].sum(0))
        damaged_unsafe = report["damaged_f1"] < fixed["damaged_f1"]
        not_better = report["valid_miou"] <= fixed["valid_miou"]
        diagnostics["damaged_unsafe"] += int(damaged_unsafe)
        diagnostics["miou_not_above_fixed"] += int(not_better)
        if damaged_unsafe:
            continue
        diagnostics["safe_nontrivial"] += 1
        diagnostics["useful_nontrivial"] += int(not not_better)
        candidates.append({
            "risk_threshold": threshold,
            "target_reject_fraction": fraction,
            "actual_reject_fraction": reject_fraction,
            "fallback_fixed_s2": False,
            "miou": report["valid_miou"], "damaged_f1": report["damaged_f1"],
        })
    selected = max(
        [fallback, *candidates],
        key=lambda row: (row["miou"], row["damaged_f1"], -row["actual_reject_fraction"]),
    )
    diagnostics["selected_fallback_fixed_s2"] = selected["fallback_fixed_s2"]
    diagnostics["selected_policy"] = selected
    return selected, diagnostics


def apply_policy(risk: torch.Tensor, policy: dict) -> torch.Tensor:
    if policy["fallback_fixed_s2"]:
        return torch.ones(len(risk), dtype=torch.bool)
    return risk < float(policy["risk_threshold"])


__all__ = [
    "DualUtilityRegressor", "REJECTION_GRID", "VARIANTS", "apply_policy",
    "choose_conservative_policy", "direction_one_hot", "impact_weights",
    "predict", "risk_score", "signed_log_transform", "target_scale",
    "train_regressor", "uses_direction", "uses_impact",
]
