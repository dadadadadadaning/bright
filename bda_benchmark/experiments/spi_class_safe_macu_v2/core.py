from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_macu_failure_audit_d0_v1.core import (
    class_f1_gradient, linear_utility,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    metric_utilities, miou_gradient, per_image_selective_histograms,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram


class ClassSafeLinearVerifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(values)
        return logits[:, 0], logits[:, 1]  # metric benefit, Damaged harm risk


def cluster_events(cache: dict) -> list[str]:
    return [
        cache["image_events"][int(index)]
        for index in cache["image_index"].long().tolist()
    ]


def image_event_mask(cache: dict, held_event: str, *, equal: bool) -> torch.Tensor:
    values = torch.tensor([event == held_event for event in cache["image_events"]])
    return values if equal else ~values


def cluster_event_mask(cache: dict, held_event: str, *, equal: bool) -> torch.Tensor:
    values = torch.tensor([event == held_event for event in cluster_events(cache)])
    return values if equal else ~values


def frozen_gradients(cache: dict, training_image_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    histogram = cache["fixed_s2_histograms"][training_image_mask].sum(0)
    return miou_gradient(histogram), class_f1_gradient(histogram, 2)


def dual_targets(
    cache: dict, metric_gradient: torch.Tensor, damaged_gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metric_utility = metric_utilities(cache["delta_hist"], metric_gradient).float()
    damaged_utility = linear_utility(cache["delta_hist"], damaged_gradient).float()
    metric_label = (metric_utility > 0).float()
    harm_label = (damaged_utility < 0).float()
    return metric_label, harm_label, metric_utility, damaged_utility


def event_impact_weights(
    cache: dict, selected: torch.Tensor, utility: torch.Tensor,
) -> torch.Tensor:
    events = [event for event, keep in zip(cluster_events(cache), selected.tolist()) if keep]
    counts = Counter(events)
    event_weight = torch.tensor([1.0 / counts[event] for event in events])
    impact = utility[selected].abs().float()
    nonzero = impact[impact > 0]
    scale = nonzero.median() if len(nonzero) else torch.tensor(1.0)
    impact = (impact / scale.clamp_min(1e-12)).sqrt().clamp(0.25, 8.0)
    result = event_weight * impact
    return result / result.mean().clamp_min(1e-8)


def train_dual_linear(
    features: torch.Tensor,
    cache: dict,
    training_cluster_mask: torch.Tensor,
    metric_label: torch.Tensor,
    harm_label: torch.Tensor,
    metric_utility: torch.Tensor,
    damaged_utility: torch.Tensor,
    *, steps: int, batch_size: int, learning_rate: float, seed: int,
) -> ClassSafeLinearVerifier:
    ids_all = torch.nonzero(training_cluster_mask, as_tuple=False).flatten()
    if len(ids_all) == 0:
        raise RuntimeError("LOEO fold has no training clusters")
    torch.manual_seed(seed)
    model = ClassSafeLinearVerifier(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    metric_pos_weight = (
        (metric_label[ids_all] == 0).sum()
        / metric_label[ids_all].sum().clamp_min(1.0)
    ).float()
    harm_pos_weight = (
        (harm_label[ids_all] == 0).sum()
        / harm_label[ids_all].sum().clamp_min(1.0)
    ).float()
    metric_weight = event_impact_weights(cache, training_cluster_mask, metric_utility)
    harm_weight = event_impact_weights(cache, training_cluster_mask, damaged_utility)
    generator = torch.Generator().manual_seed(seed + 1)
    size = min(int(batch_size), len(ids_all))
    for step in range(1, int(steps) + 1):
        local = torch.randint(len(ids_all), (size,), generator=generator)
        ids = ids_all[local]
        metric_logits, harm_logits = model(features[ids])
        metric_loss = F.binary_cross_entropy_with_logits(
            metric_logits, metric_label[ids], reduction="none",
            pos_weight=metric_pos_weight,
        )
        harm_loss = F.binary_cross_entropy_with_logits(
            harm_logits, harm_label[ids], reduction="none",
            pos_weight=harm_pos_weight,
        )
        loss = (
            metric_loss * metric_weight[local]
            + harm_loss * harm_weight[local]
        ).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite CS-MACU loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite CS-MACU gradient at step {step}")
        optimizer.step()
    return model


def predict(model: ClassSafeLinearVerifier, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        metric, harm = model(features)
        return metric.sigmoid(), harm.sigmoid()


def choose_thresholds(
    cache: dict,
    metric_scores: torch.Tensor,
    harm_scores: torch.Tensor,
    calibration_cluster_mask: torch.Tensor,
    calibration_image_mask: torch.Tensor,
    *, maximum_reject_fraction: float = 0.70,
) -> dict:
    fixed_by_image = cache["fixed_s2_histograms"].double()
    fixed = metrics_from_histogram(fixed_by_image[calibration_image_mask].sum(0))
    candidates = [{
        "metric_threshold": 0.0,
        "harm_threshold": 1.0,
        "reject_fraction": 0.0,
        "miou": fixed["valid_miou"],
        "damaged_f1": fixed["damaged_f1"],
    }]
    grid = torch.linspace(0.30, 0.70, 9)
    for metric_threshold in grid:
        for harm_threshold in grid:
            local_accept = (
                (metric_scores[calibration_cluster_mask] >= metric_threshold)
                & (harm_scores[calibration_cluster_mask] < harm_threshold)
            )
            reject_fraction = float((~local_accept).float().mean())
            if reject_fraction > maximum_reject_fraction:
                continue
            accept = torch.ones(len(metric_scores), dtype=torch.bool)
            accept[calibration_cluster_mask] = local_accept
            selected_by_image = per_image_selective_histograms(cache, accept)
            report = metrics_from_histogram(
                selected_by_image[calibration_image_mask].sum(0)
            )
            if report["damaged_f1"] < fixed["damaged_f1"]:
                continue
            candidates.append({
                "metric_threshold": float(metric_threshold),
                "harm_threshold": float(harm_threshold),
                "reject_fraction": reject_fraction,
                "miou": report["valid_miou"],
                "damaged_f1": report["damaged_f1"],
            })
    return max(
        candidates,
        key=lambda row: (row["miou"], row["damaged_f1"], -row["reject_fraction"]),
    )


def apply_thresholds(
    metric_scores: torch.Tensor, harm_scores: torch.Tensor, policy: dict,
) -> torch.Tensor:
    return (
        (metric_scores >= float(policy["metric_threshold"]))
        & (harm_scores < float(policy["harm_threshold"]))
    )


__all__ = [
    "ClassSafeLinearVerifier", "apply_thresholds", "choose_thresholds",
    "cluster_event_mask", "cluster_events", "dual_targets", "event_impact_weights",
    "frozen_gradients", "image_event_mask", "predict", "train_dual_linear",
]
