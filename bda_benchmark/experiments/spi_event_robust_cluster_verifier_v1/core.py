from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    per_image_selective_histograms,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    metrics_from_histogram,
)


OBJECTIVES = ("erm", "event_balanced", "groupdro", "vrex")


def cluster_events(cache: dict) -> list[str]:
    """Return one event name per stored modification cluster."""
    image_index = cache["image_index"].long()
    image_events = cache["image_events"]
    return [str(image_events[int(index)]) for index in image_index]


def impact_weights(
    utility_pp: torch.Tensor,
    *,
    minimum: float = 0.25,
    maximum: float = 8.0,
) -> torch.Tensor:
    """Metric-impact weights without event-frequency compensation.

    Event balancing is performed explicitly by the objective.  Keeping event
    frequency out of this function prevents accidental double compensation.
    """
    impact = utility_pp.abs().float()
    positive = impact[impact > 0]
    scale = positive.median() if len(positive) else torch.tensor(1.0)
    return (impact / scale.clamp_min(1e-12)).sqrt().clamp(minimum, maximum)


def event_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    event_ids: torch.Tensor,
    event_count: int,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a normalized weighted BCE for every event in a balanced batch."""
    rows = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none", pos_weight=pos_weight,
    )
    losses = []
    for event_id in range(event_count):
        selected = event_ids == event_id
        if not bool(selected.any()):
            raise ValueError(f"balanced batch is missing event_id={event_id}")
        denominator = weights[selected].sum().clamp_min(1e-12)
        losses.append((rows[selected] * weights[selected]).sum() / denominator)
    return torch.stack(losses)


@dataclass
class GroupDROState:
    log_weights: torch.Tensor
    ema_losses: torch.Tensor
    decay: float = 0.9
    eta: float = 0.05

    @classmethod
    def create(
        cls, event_count: int, *, decay: float = 0.9, eta: float = 0.05,
    ) -> "GroupDROState":
        return cls(
            log_weights=torch.zeros(event_count),
            ema_losses=torch.zeros(event_count),
            decay=decay,
            eta=eta,
        )

    def update(self, losses: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            detached = losses.detach().cpu()
            self.ema_losses.mul_(self.decay).add_(detached, alpha=1.0 - self.decay)
            self.log_weights.add_(self.eta * self.ema_losses)
            self.log_weights.sub_(torch.logsumexp(self.log_weights, dim=0))
        return self.log_weights.softmax(0).to(losses.device)


def robust_loss(
    losses: torch.Tensor,
    objective: str,
    *,
    groupdro: GroupDROState | None = None,
    vrex_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate event losses and return the effective event weights."""
    if objective not in OBJECTIVES[1:]:
        raise ValueError("robust_loss expects event_balanced, groupdro, or vrex")
    uniform = torch.full_like(losses, 1.0 / len(losses))
    if objective == "event_balanced":
        return losses.mean(), uniform
    if objective == "vrex":
        return losses.mean() + vrex_lambda * losses.var(unbiased=False), uniform
    if groupdro is None:
        raise ValueError("groupdro state is required")
    weights = groupdro.update(losses)
    return (weights * losses).sum(), weights


def draw_balanced_indices(
    rows_by_event: list[torch.Tensor],
    per_event: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the same number of cluster rows from every source event."""
    sampled = []
    event_ids = []
    for event_id, rows in enumerate(rows_by_event):
        if not len(rows):
            raise ValueError(f"source event {event_id} has no eligible clusters")
        offsets = torch.randint(len(rows), (per_event,), generator=generator)
        sampled.append(rows[offsets])
        event_ids.append(torch.full((per_event,), event_id, dtype=torch.long))
    return torch.cat(sampled), torch.cat(event_ids)


def _image_ids(cache: dict, events: set[str]) -> torch.Tensor:
    return torch.tensor([
        index for index, event in enumerate(cache["image_events"])
        if str(event) in events
    ], dtype=torch.long)


def _cluster_mask(cache: dict, events: set[str]) -> torch.Tensor:
    return torch.tensor([event in events for event in cluster_events(cache)])


def event_metric_rows(
    cache: dict,
    selected_by_image: torch.Tensor,
    events: Iterable[str],
) -> dict[str, dict]:
    rows = {}
    fixed = cache["fixed_s2_histograms"].double()
    for event in sorted(set(events)):
        ids = _image_ids(cache, {event})
        chosen = metrics_from_histogram(selected_by_image[ids].sum(0))
        baseline = metrics_from_histogram(fixed[ids].sum(0))
        rows[event] = {
            "gain_vs_fixed_s2_miou_pp": 100.0 * (
                chosen["valid_miou"] - baseline["valid_miou"]
            ),
            "miou": chosen["valid_miou"],
            "damaged_f1": chosen["damaged_f1"],
            "fixed_s2_miou": baseline["valid_miou"],
            "fixed_s2_damaged_f1": baseline["damaged_f1"],
        }
    return rows


def select_source_threshold(
    cache: dict,
    scores: torch.Tensor,
    source_events: Iterable[str],
    *,
    maximum_reject_fraction: float = 0.70,
    minimum_nonnegative_fraction: float = 0.70,
) -> dict:
    """Choose one global threshold without observing the held-out event."""
    source_events = set(source_events)
    cluster_mask = _cluster_mask(cache, source_events)
    image_ids = _image_ids(cache, source_events)
    fixed_hist = cache["fixed_s2_histograms"][image_ids].sum(0).double()
    fixed = metrics_from_histogram(fixed_hist)
    required_nonnegative = min(
        len(source_events), math.ceil(minimum_nonnegative_fraction * len(source_events)),
    )
    feasible = []
    fallbacks = []
    for threshold in torch.linspace(0.05, 0.95, 91):
        accept = scores >= threshold
        rejection = float((~accept[cluster_mask]).float().mean())
        if rejection > maximum_reject_fraction:
            continue
        selected = per_image_selective_histograms(cache, accept)
        pooled = metrics_from_histogram(selected[image_ids].sum(0))
        events = event_metric_rows(cache, selected, source_events)
        gains = [row["gain_vs_fixed_s2_miou_pp"] for row in events.values()]
        row = {
            "threshold": float(threshold),
            "reject_fraction": rejection,
            "pooled_miou": pooled["valid_miou"],
            "pooled_gain_pp": 100.0 * (pooled["valid_miou"] - fixed["valid_miou"]),
            "damaged_f1": pooled["damaged_f1"],
            "fixed_s2_damaged_f1": fixed["damaged_f1"],
            "event_macro_gain_pp": float(np.mean(gains)),
            "event_worst_gain_pp": float(np.min(gains)),
            "event_nonnegative": sum(value >= 0 for value in gains),
            "event_total": len(gains),
        }
        fallbacks.append(row)
        if (
            row["damaged_f1"] >= row["fixed_s2_damaged_f1"]
            and row["event_nonnegative"] >= required_nonnegative
        ):
            feasible.append(row)
    candidates = feasible if feasible else fallbacks
    if not candidates:
        raise RuntimeError("no threshold satisfies the rejection cap")
    selected = max(candidates, key=lambda row: (
        row["event_macro_gain_pp"], row["pooled_gain_pp"],
        row["damaged_f1"], -row["reject_fraction"],
    ))
    selected["source_constraints_met"] = bool(feasible)
    selected["required_nonnegative_events"] = required_nonnegative
    return selected


def evaluate_heldout_event(
    cache: dict,
    scores: torch.Tensor,
    event: str,
    threshold: float,
) -> tuple[dict, torch.Tensor]:
    accept = scores >= threshold
    selected = per_image_selective_histograms(cache, accept)
    ids = _image_ids(cache, {event})
    cluster_mask = _cluster_mask(cache, {event})
    chosen = metrics_from_histogram(selected[ids].sum(0))
    fixed = metrics_from_histogram(cache["fixed_s2_histograms"][ids].sum(0))
    return {
        "event": event,
        "threshold": threshold,
        "gain_vs_fixed_s2_miou_pp": 100.0 * (
            chosen["valid_miou"] - fixed["valid_miou"]
        ),
        "miou": chosen["valid_miou"],
        "damaged_f1": chosen["damaged_f1"],
        "fixed_s2_miou": fixed["valid_miou"],
        "fixed_s2_damaged_f1": fixed["damaged_f1"],
        "reject_fraction": float((~accept[cluster_mask]).float().mean()),
        "images": int(len(ids)),
        "clusters": int(cluster_mask.sum()),
    }, selected[ids]


def bootstrap_event_mean(
    gains: list[float], iterations: int, seed: int,
) -> dict:
    values = torch.tensor(gains, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(iterations):
        ids = torch.randint(len(values), (len(values),), generator=generator)
        rows.append(float(values[ids].mean()))
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(rows, 0.025)),
        "ci95_high_pp": float(np.quantile(rows, 0.975)),
        "iterations": iterations,
        "resampling_unit": "event",
    }


__all__ = [
    "OBJECTIVES", "GroupDROState", "bootstrap_event_mean", "cluster_events",
    "draw_balanced_indices", "evaluate_heldout_event", "event_losses",
    "event_metric_rows", "impact_weights", "robust_loss",
    "select_source_threshold",
]
