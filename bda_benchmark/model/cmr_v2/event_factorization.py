# -*- coding: utf-8 -*-
"""Event-factorized features and counterfactual event experts for CMR-V2.

Event identity is used only to estimate/remove an imaging-domain component or
to define a diagnostic expert.  It is never a damage target.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


FEATURE_PROTOCOL = "CMR_V2_event_class_balanced_factorization_v1"


def _event_tensor(events: Sequence[str], event_order: Sequence[str]) -> torch.Tensor:
    mapping = {str(event): index for index, event in enumerate(event_order)}
    try:
        return torch.tensor([mapping[str(event)] for event in events], dtype=torch.long)
    except KeyError as error:
        raise RuntimeError(f"unknown event while encoding CMR-V2 features: {error}") from error


def fit_event_factorization(
    features: torch.Tensor,
    labels: torch.Tensor,
    events: Sequence[str],
    *,
    scale_floor: float = 1e-3,
) -> Dict[str, object]:
    """Estimate class-balanced event centers and diagonal residual scales.

    For every event, the three class means receive exactly 1/3 weight.  Events
    missing any class are marked unreliable and use the global class-balanced
    fallback for query transformation; they remain visible in diagnostics.
    """
    x = features.detach().float().cpu()
    y = labels.detach().long().flatten().cpu()
    event_names = list(map(str, events))
    if x.ndim != 2 or x.shape[0] != y.numel() or len(event_names) != y.numel():
        raise ValueError("features, labels and events do not align")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    classes = (0, 1, 2)
    global_means = []
    for cls in classes:
        mask = y == cls
        if not bool(mask.any()):
            raise RuntimeError(f"training cache has no class {cls}")
        global_means.append(x[mask].mean(dim=0))
    global_center = torch.stack(global_means).mean(dim=0)
    global_second = torch.stack([
        (x[y == cls] - global_center).square().mean(dim=0) for cls in classes
    ]).mean(dim=0)
    global_scale = global_second.clamp_min(float(scale_floor) ** 2).sqrt()

    event_order = sorted(set(event_names))
    event_ids = _event_tensor(event_names, event_order)
    centers, scales, reliable, counts = [], [], [], []
    for event_index, _ in enumerate(event_order):
        class_masks = [(event_ids == event_index) & (y == cls) for cls in classes]
        class_counts = [int(mask.sum()) for mask in class_masks]
        is_reliable = all(count > 0 for count in class_counts)
        if is_reliable:
            class_means = torch.stack([x[mask].mean(dim=0) for mask in class_masks])
            center = class_means.mean(dim=0)
            second = torch.stack([
                (x[mask] - center).square().mean(dim=0) for mask in class_masks
            ]).mean(dim=0)
            scale = second.clamp_min(float(scale_floor) ** 2).sqrt()
        else:
            center, scale = global_center.clone(), global_scale.clone()
        centers.append(center)
        scales.append(scale)
        reliable.append(is_reliable)
        counts.append(class_counts)
    return {
        "protocol": FEATURE_PROTOCOL,
        "event_order": event_order,
        "centers": torch.stack(centers),
        "scales": torch.stack(scales),
        "reliable": torch.tensor(reliable, dtype=torch.bool),
        "class_counts": torch.tensor(counts, dtype=torch.long),
        "global_center": global_center,
        "global_scale": global_scale,
        "scale_floor": float(scale_floor),
        "center_definition": "equal_mean_of_intact_damaged_destroyed_class_means",
    }


def transform_features(
    features: torch.Tensor,
    events: Sequence[str],
    factorization: Mapping[str, object],
    variant: str,
) -> torch.Tensor:
    if factorization.get("protocol") != FEATURE_PROTOCOL:
        raise RuntimeError("unknown CMR-V2 factorization protocol")
    x = features.detach().float().cpu()
    variant = str(variant).lower()
    if variant == "raw":
        return F.normalize(x, dim=1, eps=1e-8)
    event_order = list(map(str, factorization["event_order"]))
    mapping = {name: index for index, name in enumerate(event_order)}
    centers, scales, reliable = (
        factorization["centers"].float(),
        factorization["scales"].float(),
        factorization["reliable"].bool(),
    )
    encoded = torch.tensor([mapping.get(str(event), -1) for event in events], dtype=torch.long)
    valid = encoded >= 0
    valid &= torch.where(valid, reliable[encoded.clamp_min(0)], torch.zeros_like(valid))
    center_rows = factorization["global_center"].float()[None, :].expand_as(x).clone()
    scale_rows = factorization["global_scale"].float()[None, :].expand_as(x).clone()
    if bool(valid.any()):
        center_rows[valid] = centers[encoded[valid]]
        scale_rows[valid] = scales[encoded[valid]]
    residual = x - center_rows
    if variant == "centered":
        output = residual
    elif variant == "whitened":
        output = residual / scale_rows.clamp_min(1e-6)
    else:
        raise ValueError(f"unknown feature variant: {variant}")
    return F.normalize(output, dim=1, eps=1e-8)


def _top_logmeanexp(similarity: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    take = min(max(int(k), 1), int(similarity.shape[1]))
    values = similarity.topk(take, dim=1).values
    if temperature <= 0:
        return values[:, 0]
    scaled = values / float(temperature)
    return float(temperature) * (torch.logsumexp(scaled, dim=1) - math.log(float(take)))


def frozen_topk_delta(
    candidate_scores: torch.Tensor,
    candidate_labels: torch.Tensor,
    *,
    case_k: int = 5,
    temperature: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """EC0 counterfactual score inside an already frozen candidate matrix."""
    scores = candidate_scores.float().cpu()
    labels = candidate_labels.long().cpu()
    output = torch.full((scores.shape[0],), float("nan"))
    intact_score = torch.full_like(output, float("nan"))
    damaged_score = torch.full_like(output, float("nan"))
    for row in range(scores.shape[0]):
        intact = scores[row, labels[row] == 0][None, :]
        damaged = scores[row, labels[row] == 1][None, :]
        if intact.numel() == 0 or damaged.numel() == 0:
            continue
        intact_score[row] = _top_logmeanexp(intact, case_k, temperature)[0]
        damaged_score[row] = _top_logmeanexp(damaged, case_k, temperature)[0]
        output[row] = damaged_score[row] - intact_score[row]
    return {"delta": output, "intact_score": intact_score, "damaged_score": damaged_score}


@torch.no_grad()
def event_expert_deltas(
    query_features: torch.Tensor,
    query_events: Sequence[str],
    query_source_rows: torch.Tensor,
    memory_features: torch.Tensor,
    memory_labels: torch.Tensor,
    memory_events: Sequence[str],
    memory_source_rows: torch.Tensor,
    *,
    protocol: str,
    case_k: int = 5,
    temperature: float = 0.10,
    device: str | torch.device = "cpu",
    query_chunk: int = 2048,
    exclude_identical_source: bool = False,
) -> Dict[str, object]:
    """Score I/D counterfactuals independently for each training event."""
    protocol = str(protocol).lower()
    if protocol not in {"same_event", "exclude_same_event"}:
        raise ValueError("protocol must be same_event or exclude_same_event")
    q = F.normalize(query_features.float(), dim=1, eps=1e-8).cpu()
    m = F.normalize(memory_features.float(), dim=1, eps=1e-8).cpu()
    my = memory_labels.long().flatten().cpu()
    q_rows = query_source_rows.long().flatten().cpu()
    m_rows = memory_source_rows.long().flatten().cpu()
    q_events = list(map(str, query_events))
    m_events = list(map(str, memory_events))
    if q.shape[0] != len(q_events) or m.shape[0] != len(m_events):
        raise ValueError("event expert inputs do not align")
    expert_order = sorted(set(m_events))
    deltas = torch.full((q.shape[0], len(expert_order)), float("nan"))
    intact_scores = torch.full_like(deltas, float("nan"))
    damaged_scores = torch.full_like(deltas, float("nan"))
    run_device = torch.device(device)
    for expert_index, expert_event in enumerate(expert_order):
        support_by_class = []
        for cls in (0, 1):
            indices = torch.tensor(
                [i for i, event in enumerate(m_events) if event == expert_event and int(my[i]) == cls],
                dtype=torch.long,
            )
            support_by_class.append(indices)
        if any(indices.numel() == 0 for indices in support_by_class):
            continue
        allowed = torch.tensor([
            event == expert_event if protocol == "same_event" else event != expert_event
            for event in q_events
        ], dtype=torch.bool)
        allowed_rows = torch.nonzero(allowed, as_tuple=False).flatten()
        for start in range(0, allowed_rows.numel(), int(query_chunk)):
            query_indices = allowed_rows[start : start + int(query_chunk)]
            query_batch = q[query_indices].to(run_device)
            class_values = []
            for support_indices in support_by_class:
                similarity = query_batch @ m[support_indices].to(run_device).t()
                if exclude_identical_source:
                    same_source = (
                        q_rows[query_indices, None].to(run_device)
                        == m_rows[support_indices][None, :].to(run_device)
                    )
                    similarity.masked_fill_(same_source, -1e4)
                class_values.append(_top_logmeanexp(similarity, case_k, temperature).cpu())
            intact_scores[query_indices, expert_index] = class_values[0]
            damaged_scores[query_indices, expert_index] = class_values[1]
            deltas[query_indices, expert_index] = class_values[1] - class_values[0]
    valid = torch.isfinite(deltas)
    count = valid.sum(dim=1)
    safe = torch.where(valid, deltas, torch.zeros_like(deltas))
    mean = safe.sum(dim=1) / count.clamp_min(1)
    centered = torch.where(valid, deltas - mean[:, None], torch.zeros_like(deltas))
    std = (centered.square().sum(dim=1) / count.clamp_min(1)).sqrt()
    positive = ((deltas > 0) & valid).sum(dim=1)
    fraction = positive.float() / count.clamp_min(1).float()
    mean[count == 0] = float("nan")
    std[count == 0] = float("nan")
    fraction[count == 0] = float("nan")
    return {
        "expert_order": expert_order,
        "deltas": deltas,
        "intact_scores": intact_scores,
        "damaged_scores": damaged_scores,
        "mean_delta": mean,
        "expert_std": std,
        "positive_experts": positive,
        "positive_fraction": fraction,
        "expert_count": count,
        "protocol": protocol,
    }


__all__ = [
    "FEATURE_PROTOCOL",
    "event_expert_deltas",
    "fit_event_factorization",
    "frozen_topk_delta",
    "transform_features",
]
