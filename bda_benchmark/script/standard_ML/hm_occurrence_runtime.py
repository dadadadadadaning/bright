# -*- coding: utf-8 -*-
"""Pure utilities for the O-K0 cross-event Occurrence-key audit."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Dict

import torch
import torch.nn.functional as F

from script.standard_ML.im1_matching_reliability_runtime import event_macro_auroc
from script.standard_ML.rm_region_retrieval_runtime import binary_auroc


def event_class_capped_indices(
    labels: torch.Tensor,
    events: torch.Tensor,
    *,
    cap_per_event_class: int,
    seed: int,
    excluded_event: int | None = None,
) -> torch.Tensor:
    """Sample each event x Occurrence cell without replacement."""

    y = labels.long().flatten().cpu()
    e = events.long().flatten().cpu()
    if y.numel() != e.numel() or y.numel() == 0:
        raise ValueError("labels/events must be aligned and non-empty")
    if not set(torch.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("Occurrence labels must be binary")
    if int(cap_per_event_class) <= 0:
        raise ValueError("cap_per_event_class must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for event in sorted(int(value) for value in torch.unique(e).tolist()):
        if excluded_event is not None and event == int(excluded_event):
            continue
        for label in (0, 1):
            indices = torch.nonzero(
                (e == event) & (y == label), as_tuple=False
            ).flatten()
            if indices.numel() == 0:
                continue
            take = min(int(cap_per_event_class), int(indices.numel()))
            order = torch.randperm(indices.numel(), generator=generator)[:take]
            selected.append(indices[order])
    if not selected:
        raise RuntimeError("event-class cap selected no rows")
    result = torch.cat(selected)
    if torch.unique(result).numel() != result.numel():
        raise AssertionError("event-class cap sampled a row more than once")
    return result


def fit_binary_centres(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    x = F.normalize(features.detach().float().cpu(), dim=1, eps=1e-8)
    y = labels.long().flatten().cpu()
    index = indices.long().flatten().cpu()
    if x.shape[0] != y.numel() or index.numel() == 0:
        raise ValueError("centre fitting inputs are invalid")
    centres = []
    for label in (0, 1):
        members = x[index[y[index] == label]]
        if members.numel() == 0:
            raise RuntimeError(f"Occurrence centre {label} has no support")
        centres.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-8))
    return torch.stack(centres)


def binary_centre_scores(features: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    x = F.normalize(features.detach().float().cpu(), dim=1, eps=1e-8)
    c = F.normalize(centres.detach().float().cpu(), dim=1, eps=1e-8)
    if x.ndim != 2 or c.shape != (2, x.shape[1]):
        raise ValueError("binary-centre feature dimensions differ")
    similarity = x @ c.t()
    return similarity[:, 1] - similarity[:, 0]


def loeo_occurrence_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    *,
    cap_per_event_class: int,
    seed: int,
) -> torch.Tensor:
    """Score every Train row using centres fitted without its event."""

    x = features.detach().float().cpu()
    y = labels.long().flatten().cpu()
    e = events.long().flatten().cpu()
    if x.shape[0] != y.numel() or y.numel() != e.numel():
        raise ValueError("LOEO inputs are not aligned")
    scores = torch.full((y.numel(),), float("nan"), dtype=torch.float32)
    for held_event in sorted(int(value) for value in torch.unique(e).tolist()):
        query_mask = e == held_event
        fit_indices = event_class_capped_indices(
            y,
            e,
            cap_per_event_class=cap_per_event_class,
            seed=int(seed),
            excluded_event=held_event,
        )
        centres = fit_binary_centres(x, y, fit_indices)
        scores[query_mask] = binary_centre_scores(x[query_mask], centres)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("LOEO left non-finite scores")
    return scores


def occurrence_auc_report(
    scores: torch.Tensor,
    labels: torch.Tensor,
    event_names: Sequence[str],
) -> Dict[str, object]:
    target = labels.bool().flatten().cpu()
    event_report = event_macro_auroc(scores, target, event_names)
    event_values = event_report["event_auroc"]
    return {
        "global_auroc": float(binary_auroc(scores, target)),
        **event_report,
        "direction_consistent_events": sum(
            float(value) >= 0.5 for value in event_values.values()
        ),
    }


def fit_event_centres(
    features: torch.Tensor,
    events: torch.Tensor,
    *,
    cap_per_event: int,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    """Fit capped event centroids for the event-leakage probe."""

    x = F.normalize(features.detach().float().cpu(), dim=1, eps=1e-8)
    e = events.long().flatten().cpu()
    if x.shape[0] != e.numel() or int(cap_per_event) <= 0:
        raise ValueError("event-centre inputs are invalid")
    generator = torch.Generator().manual_seed(int(seed))
    centres, codes = [], []
    for event in sorted(int(value) for value in torch.unique(e).tolist()):
        indices = torch.nonzero(e == event, as_tuple=False).flatten()
        take = min(int(cap_per_event), int(indices.numel()))
        order = torch.randperm(indices.numel(), generator=generator)[:take]
        centre = F.normalize(x[indices[order]].mean(dim=0), dim=0, eps=1e-8)
        centres.append(centre)
        codes.append(event)
    return torch.stack(centres), codes


def event_probe_report(
    train_features: torch.Tensor,
    train_events: torch.Tensor,
    query_features: torch.Tensor,
    query_events: torch.Tensor,
    *,
    cap_per_event: int,
    seed: int,
) -> Dict[str, object]:
    centres, codes = fit_event_centres(
        train_features, train_events, cap_per_event=cap_per_event, seed=seed
    )
    q = F.normalize(query_features.detach().float().cpu(), dim=1, eps=1e-8)
    prediction_local = (q @ centres.t()).argmax(dim=1)
    code_tensor = torch.tensor(codes, dtype=torch.long)
    prediction = code_tensor[prediction_local]
    target = query_events.long().flatten().cpu()
    if prediction.numel() != target.numel():
        raise ValueError("event probe inputs are not aligned")
    per_event = {}
    for event in sorted(int(value) for value in torch.unique(target).tolist()):
        mask = target == event
        per_event[str(event)] = float((prediction[mask] == target[mask]).float().mean())
    return {
        "accuracy": float((prediction == target).float().mean()),
        "balanced_accuracy": sum(per_event.values()) / max(len(per_event), 1),
        "per_event_accuracy": per_event,
        "event_centres": len(codes),
    }


__all__ = [
    "binary_centre_scores",
    "event_class_capped_indices",
    "event_probe_report",
    "fit_binary_centres",
    "fit_event_centres",
    "loeo_occurrence_scores",
    "occurrence_auc_report",
]
