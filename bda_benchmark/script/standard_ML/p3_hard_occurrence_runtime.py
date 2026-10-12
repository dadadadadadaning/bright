# -*- coding: utf-8 -*-
"""Shared replay/training utilities for CMR-H1 Train-only supervision."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from model.DamageFormer_S8_MICB_DSMIL import connected_components_4


def component_bbox(component: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(component)
    if ys.size == 0:
        raise ValueError("component is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def match_cached_component(
    building_mask: torch.Tensor | np.ndarray,
    expected_bbox: Sequence[int],
    expected_area: int,
) -> np.ndarray | None:
    """Find the exact frozen-P3 component used by the CMR cache."""

    expected = tuple(int(value) for value in expected_bbox)
    for component in connected_components_4(building_mask):
        if int(component.sum()) != int(expected_area):
            continue
        if component_bbox(component) == expected:
            return component
    return None


def differentiable_region_mean_key(
    fused_roi: torch.Tensor,
    component: torch.Tensor | np.ndarray,
    *,
    batch_index: int,
) -> torch.Tensor:
    """Reproduce CMR Mean pooling while keeping gradients to ``fused_roi``."""

    if fused_roi.ndim != 4:
        raise ValueError("fused_roi must be [B,C,H,W]")
    if not 0 <= int(batch_index) < fused_roi.shape[0]:
        raise IndexError("batch_index is outside fused_roi")
    mask = torch.as_tensor(
        np.asarray(component, dtype=np.float32),
        device=fused_roi.device,
        dtype=torch.float32,
    )
    weights = F.interpolate(
        mask[None, None],
        size=fused_roi.shape[-2:],
        mode="area",
    )[0, 0]
    denominator = weights.sum().clamp_min(1e-6)
    key = (fused_roi[int(batch_index)].float() * weights[None]).sum(
        dim=(1, 2)
    ) / denominator
    return F.normalize(key, dim=0, eps=1e-8)


def region_conditional_probability(
    damage_logits: torch.Tensor,
    component: torch.Tensor | np.ndarray,
    *,
    batch_index: int,
) -> torch.Tensor:
    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ValueError("damage_logits must be [B,4,H,W]")
    mask = torch.as_tensor(
        np.asarray(component, dtype=np.bool_), device=damage_logits.device
    )
    probability = torch.softmax(damage_logits[int(batch_index)].float(), dim=0)
    region = probability[:, mask].mean(dim=1)
    conditional = region[1:4]
    return conditional / conditional.sum().clamp_min(1e-8)


def occurrence_info_nce(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if query.ndim != 2 or positive.shape != query.shape:
        raise ValueError("query and positive must be [B,C]")
    if negatives.ndim != 3 or negatives.shape[0] != query.shape[0] or negatives.shape[2] != query.shape[1]:
        raise ValueError("negatives must be [B,N,C]")
    query = F.normalize(query.float(), dim=1, eps=1e-8)
    positive = F.normalize(positive.float(), dim=1, eps=1e-8)
    negatives = F.normalize(negatives.float(), dim=2, eps=1e-8)
    positive_logit = (query * positive).sum(dim=1, keepdim=True)
    negative_logits = torch.einsum("bc,bnc->bn", query, negatives)
    logits = torch.cat((positive_logit, negative_logits), dim=1) / float(temperature)
    target = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, target)


def protected_spatial_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 2.0,
    protected_classes: Iterable[int] = (1, 3),
    ignore_index: int = 255,
) -> torch.Tensor:
    """Distil only Intact/Destroyed pixels; Damaged remains free to improve."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits must have identical shape")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    mask = torch.zeros_like(target, dtype=torch.bool)
    for class_id in protected_classes:
        mask |= target == int(class_id)
    mask &= target != int(ignore_index)
    if not bool(mask.any()):
        return student_logits.float().sum() * 0.0
    t = float(temperature)
    student = F.log_softmax(student_logits.float() / t, dim=1)
    teacher = F.softmax(teacher_logits.detach().float() / t, dim=1)
    per_pixel = F.kl_div(student, teacher, reduction="none").sum(dim=1) * (t * t)
    return per_pixel[mask].mean()


def h1_validation_checks(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    minimum_damaged_gain_pp: float = 0.30,
    maximum_guard_drop_pp: float = 0.20,
    minimum_nonnegative_events: int = 10,
) -> tuple[Dict[str, bool], Dict[str, float | int]]:
    event_base = baseline["event_miou_by_name"]
    event_candidate = candidate["event_miou_by_name"]
    common = sorted(set(event_base) & set(event_candidate))
    deltas = {
        event: 100.0 * (float(event_candidate[event]) - float(event_base[event]))
        for event in common
    }
    nonnegative = sum(value >= -1e-12 for value in deltas.values())
    damaged_gain = 100.0 * (
        float(candidate["damaged_iou"]) - float(baseline["damaged_iou"])
    )
    intact_drop = 100.0 * (
        float(baseline["intact_iou"]) - float(candidate["intact_iou"])
    )
    destroyed_drop = 100.0 * (
        float(baseline["destroyed_iou"]) - float(candidate["destroyed_iou"])
    )
    diagnostics: Dict[str, float | int] = {
        "damaged_gain_pp": damaged_gain,
        "intact_drop_pp": intact_drop,
        "destroyed_drop_pp": destroyed_drop,
        "damaged_gate_miss_delta_pp": 100.0 * (
            float(candidate["damaged_gate_miss"]) - float(baseline["damaged_gate_miss"])
        ),
        "affected_recall_delta_pp": 100.0 * (
            float(candidate["affected_recall"]) - float(baseline["affected_recall"])
        ),
        "nonnegative_events": nonnegative,
        "comparable_events": len(common),
    }
    checks = {
        "all_14_events_present": len(common) == 14,
        "valid_miou_nondecreasing": float(candidate["valid_miou"]) >= float(baseline["valid_miou"]),
        "damaged_iou_plus_declared_gain": damaged_gain >= float(minimum_damaged_gain_pp),
        "damaged_gate_miss_decreasing": float(candidate["damaged_gate_miss"]) < float(baseline["damaged_gate_miss"]),
        "affected_recall_nondecreasing": float(candidate["affected_recall"]) >= float(baseline["affected_recall"]),
        "intact_iou_drop_within_limit": intact_drop <= float(maximum_guard_drop_pp),
        "destroyed_iou_drop_within_limit": destroyed_drop <= float(maximum_guard_drop_pp),
        "event_worst_nondecreasing": float(candidate["event_worst_official_style_4class_miou"]) >= float(baseline["event_worst_official_style_4class_miou"]),
        "nonnegative_events_at_least_declared": nonnegative >= int(minimum_nonnegative_events),
    }
    return checks, {**diagnostics, "event_delta_pp": deltas}


__all__ = [
    "component_bbox",
    "differentiable_region_mean_key",
    "h1_validation_checks",
    "match_cached_component",
    "occurrence_info_nce",
    "protected_spatial_kl",
    "region_conditional_probability",
]
