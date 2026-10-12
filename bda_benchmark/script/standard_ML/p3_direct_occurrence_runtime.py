# -*- coding: utf-8 -*-
"""Losses and admission checks for the isolated P3-O1 experiment.

P3-O1 does not introduce a new inference head.  It derives the binary
Intact/Affected decision directly from the existing four-class logits:

    logit(Affected) - logit(Intact)
      = logsumexp(logit(Damaged), logit(Destroyed)) - logit(Intact).

Background and ignored pixels never enter the auxiliary objective.  Intact
and Affected are averaged separately so the natural Intact majority cannot
silence the rare Affected gradient.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def affected_logit(damage_logits: torch.Tensor) -> torch.Tensor:
    """Return the exact binary Affected-vs-Intact log-odds.

    Class order is Background, Intact, Damaged, Destroyed.  Background is
    intentionally absent because occurrence is defined only for buildings.
    """

    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ValueError("damage_logits must be [B,4,H,W]")
    affected = torch.logsumexp(damage_logits[:, 2:4].float(), dim=1)
    intact = damage_logits[:, 1].float()
    return affected - intact


def balanced_occurrence_bce(
    damage_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, Dict[str, float | int | None]]:
    """Balanced binary occurrence loss on GT building pixels only.

    Intact (class 1) is the negative class and Damaged/Destroyed (2/3) form
    the positive Affected class.  Each present side contributes one half,
    avoiding a hidden class-prior change when a mini-batch is Intact-heavy.
    """

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must be [B,H,W] or [B,1,H,W]")
    if tuple(target.shape) != (
        damage_logits.shape[0],
        damage_logits.shape[2],
        damage_logits.shape[3],
    ):
        raise ValueError("target and damage_logits spatial shapes must match")

    margin = affected_logit(damage_logits)
    valid = target != int(ignore_index)
    intact_mask = valid & (target == 1)
    affected_mask = valid & ((target == 2) | (target == 3))

    terms = []
    intact_loss = margin.sum() * 0.0
    affected_loss = margin.sum() * 0.0
    if bool(intact_mask.any()):
        intact_loss = F.softplus(margin[intact_mask]).mean()
        terms.append(intact_loss)
    if bool(affected_mask.any()):
        affected_loss = F.softplus(-margin[affected_mask]).mean()
        terms.append(affected_loss)
    if not terms:
        loss = margin.sum() * 0.0
    else:
        loss = torch.stack(terms).mean()

    with torch.no_grad():
        intact_count = int(intact_mask.sum().item())
        affected_count = int(affected_mask.sum().item())
        intact_correct = (
            float((margin[intact_mask] < 0).float().mean().item())
            if intact_count
            else None
        )
        affected_correct = (
            float((margin[affected_mask] > 0).float().mean().item())
            if affected_count
            else None
        )
    diagnostics: Dict[str, float | int | None] = {
        "intact_pixels": intact_count,
        "affected_pixels": affected_count,
        "intact_loss": float(intact_loss.detach()),
        "affected_loss": float(affected_loss.detach()),
        "intact_gate_accuracy": intact_correct,
        "affected_gate_recall": affected_correct,
    }
    return loss, diagnostics


def region_occurrence_margin(
    damage_logits: torch.Tensor,
    component: torch.Tensor | np.ndarray,
    *,
    batch_index: int,
) -> torch.Tensor:
    """Mean Affected-vs-Intact log-odds for one exact building region."""

    if not 0 <= int(batch_index) < damage_logits.shape[0]:
        raise IndexError("batch_index is outside damage_logits")
    mask = torch.as_tensor(
        np.asarray(component, dtype=np.bool_), device=damage_logits.device
    )
    if mask.ndim != 2 or tuple(mask.shape) != tuple(damage_logits.shape[-2:]):
        raise ValueError("component must match the logit spatial resolution")
    if not bool(mask.any()):
        raise ValueError("component is empty")
    return affected_logit(damage_logits)[int(batch_index)][mask].mean()


def hard_region_boundary_loss(
    damaged_margin: torch.Tensor,
    intact_margin: torch.Tensor,
    *,
    required_margin: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Push missed Damaged above, and risky Intact below, the gate."""

    if damaged_margin.ndim != 1 or intact_margin.ndim != 1:
        raise ValueError("region margins must be one-dimensional")
    if damaged_margin.numel() == 0 or intact_margin.numel() == 0:
        raise ValueError("both hard-region classes must be present")
    threshold = float(required_margin)
    damaged_loss = F.softplus(threshold - damaged_margin).mean()
    intact_loss = F.softplus(threshold + intact_margin).mean()
    loss = 0.5 * (damaged_loss + intact_loss)
    diagnostics = {
        "damaged_margin_mean": float(damaged_margin.detach().mean()),
        "intact_margin_mean": float(intact_margin.detach().mean()),
        "damaged_boundary_loss": float(damaged_loss.detach()),
        "intact_boundary_loss": float(intact_loss.detach()),
    }
    return loss, diagnostics


def select_hard_region_indices(
    labels: torch.Tensor,
    base_predictions: torch.Tensor,
    base_probabilities: torch.Tensor,
    events: torch.Tensor,
    *,
    cap_per_event: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Build Train-only D->I and high-risk correct-Intact pools.

    Region labels/predictions use the CMR zero-based order I/D/X.  Features
    are deliberately not accepted by this function: O2 is not retrieval.
    """

    labels = labels.long().flatten().cpu()
    base_predictions = base_predictions.long().flatten().cpu()
    events = events.long().flatten().cpu()
    probabilities = base_probabilities.float().cpu()
    if not (
        labels.shape == base_predictions.shape == events.shape
        and probabilities.ndim == 2
        and probabilities.shape[0] == labels.numel()
        and probabilities.shape[1] == 3
    ):
        raise ValueError("hard-region cache tensors do not align")
    if int(cap_per_event) <= 0:
        raise ValueError("cap_per_event must be positive")

    generator = torch.Generator().manual_seed(int(seed))
    positive_rows = []
    negative_rows = []
    positive_counts: Dict[str, int] = {}
    negative_counts: Dict[str, int] = {}
    affected_probability = probabilities[:, 1:3].sum(dim=1)
    for event in sorted(torch.unique(events).tolist()):
        event_mask = events == int(event)
        positive = torch.nonzero(
            event_mask & (labels == 1) & (base_predictions == 0),
            as_tuple=False,
        ).flatten()
        if positive.numel() > int(cap_per_event):
            order = torch.randperm(positive.numel(), generator=generator)
            positive = positive[order[: int(cap_per_event)]]
        negative = torch.nonzero(
            event_mask & (labels == 0) & (base_predictions == 0),
            as_tuple=False,
        ).flatten()
        if negative.numel() > 0:
            order = torch.argsort(
                affected_probability[negative], descending=True, stable=True
            )
            negative = negative[order[: int(cap_per_event)]]
        if positive.numel():
            positive_rows.append(positive)
        if negative.numel():
            negative_rows.append(negative)
        positive_counts[str(int(event))] = int(positive.numel())
        negative_counts[str(int(event))] = int(negative.numel())

    if not positive_rows or not negative_rows:
        raise RuntimeError("hard-region mining produced an empty class")
    positive_indices = torch.cat(positive_rows)
    negative_indices = torch.cat(negative_rows)
    audit: Dict[str, object] = {
        "positive_definition": "gt_damaged_and_frozen_p3_intact",
        "negative_definition": (
            "gt_intact_and_frozen_p3_intact_highest_affected_probability"
        ),
        "positive_regions": int(positive_indices.numel()),
        "negative_regions": int(negative_indices.numel()),
        "positive_event_counts": positive_counts,
        "negative_event_counts": negative_counts,
        "positive_events": sum(value > 0 for value in positive_counts.values()),
        "negative_events": sum(value > 0 for value in negative_counts.values()),
        "cap_per_event": int(cap_per_event),
        "uses_feature_similarity": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    return positive_indices, negative_indices, audit


def o1_validation_checks(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    minimum_damaged_gain_pp: float = 0.30,
    maximum_guard_drop_pp: float = 0.20,
    minimum_nonnegative_events: int = 10,
) -> tuple[Dict[str, bool], Dict[str, object]]:
    """Predeclared P3-O1 gate; every check is required for admission."""

    event_base = baseline["event_miou_by_name"]
    event_candidate = candidate["event_miou_by_name"]
    common = sorted(set(event_base) & set(event_candidate))
    event_delta_pp = {
        event: 100.0 * (
            float(event_candidate[event]) - float(event_base[event])
        )
        for event in common
    }
    nonnegative = sum(value >= -1e-12 for value in event_delta_pp.values())
    damaged_gain_pp = 100.0 * (
        float(candidate["damaged_iou"]) - float(baseline["damaged_iou"])
    )
    intact_drop_pp = 100.0 * (
        float(baseline["intact_iou"]) - float(candidate["intact_iou"])
    )
    destroyed_drop_pp = 100.0 * (
        float(baseline["destroyed_iou"])
        - float(candidate["destroyed_iou"])
    )
    diagnostics: Dict[str, object] = {
        "damaged_gain_pp": damaged_gain_pp,
        "intact_drop_pp": intact_drop_pp,
        "destroyed_drop_pp": destroyed_drop_pp,
        "damaged_to_intact_delta_pp": 100.0 * (
            float(candidate["damaged_to_intact"])
            - float(baseline["damaged_to_intact"])
        ),
        "damaged_gate_miss_delta_pp": 100.0 * (
            float(candidate["damaged_gate_miss"])
            - float(baseline["damaged_gate_miss"])
        ),
        "affected_recall_delta_pp": 100.0 * (
            float(candidate["affected_recall"])
            - float(baseline["affected_recall"])
        ),
        "nonnegative_events": nonnegative,
        "comparable_events": len(common),
        "event_delta_pp": event_delta_pp,
    }
    checks = {
        "all_14_events_present": len(common) == 14,
        "valid_miou_nondecreasing": float(candidate["valid_miou"])
        >= float(baseline["valid_miou"]),
        "damaged_iou_plus_declared_gain": damaged_gain_pp
        >= float(minimum_damaged_gain_pp),
        "damaged_to_intact_decreasing": float(candidate["damaged_to_intact"])
        < float(baseline["damaged_to_intact"]),
        "damaged_gate_miss_decreasing": float(candidate["damaged_gate_miss"])
        < float(baseline["damaged_gate_miss"]),
        "affected_recall_nondecreasing": float(candidate["affected_recall"])
        >= float(baseline["affected_recall"]),
        "intact_iou_drop_within_limit": intact_drop_pp
        <= float(maximum_guard_drop_pp),
        "destroyed_iou_drop_within_limit": destroyed_drop_pp
        <= float(maximum_guard_drop_pp),
        "event_worst_nondecreasing": float(
            candidate["event_worst_official_style_4class_miou"]
        )
        >= float(baseline["event_worst_official_style_4class_miou"]),
        "nonnegative_events_at_least_declared": nonnegative
        >= int(minimum_nonnegative_events),
    }
    return checks, diagnostics


__all__ = [
    "affected_logit",
    "balanced_occurrence_bce",
    "hard_region_boundary_loss",
    "o1_validation_checks",
    "region_occurrence_margin",
    "select_hard_region_indices",
]
