# -*- coding: utf-8 -*-
"""Utilities for train-locked, one-way CMR Damaged rescue."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict

import torch

from script.standard_ML.cmr_pairwise_utils import candidate_metrics


def promote_selected_to_front(
    raw_ranked_labels: torch.Tensor,
    selected_indices: torch.Tensor,
    accept: torch.Tensor,
) -> torch.Tensor:
    """Stable-promote one candidate for accepted rows; leave all others exact."""

    raw_ranked_labels = raw_ranked_labels.long()
    selected_indices = selected_indices.long().flatten()
    accept = accept.bool().flatten()
    if raw_ranked_labels.ndim != 2:
        raise ValueError("raw_ranked_labels must be [N,K]")
    if selected_indices.numel() != raw_ranked_labels.shape[0] or accept.numel() != raw_ranked_labels.shape[0]:
        raise ValueError("promotion controls must align with ranking rows")
    if selected_indices.numel() and (
        int(selected_indices.min()) < 0
        or int(selected_indices.max()) >= raw_ranked_labels.shape[1]
    ):
        raise ValueError("selected candidate index is out of range")
    result = raw_ranked_labels.clone()
    rows = torch.nonzero(accept, as_tuple=False).flatten()
    for row in rows.tolist():
        position = int(selected_indices[row])
        if position == 0:
            continue
        original = result[row].clone()
        result[row, 0] = original[position]
        result[row, 1 : position + 1] = original[:position]
    return result


def one_way_accept_mask(
    raw_ranked_labels: torch.Tensor,
    model_ranked_labels: torch.Tensor,
    proposal_margin: torch.Tensor,
    *,
    threshold: float,
    damaged_class: int = 1,
) -> torch.Tensor:
    """Only Non-D raw -> D model proposals can ever be accepted."""

    if raw_ranked_labels.ndim != 2 or model_ranked_labels.shape != raw_ranked_labels.shape:
        raise ValueError("raw/model rankings must be aligned [N,K]")
    proposal_margin = proposal_margin.float().flatten()
    if proposal_margin.numel() != raw_ranked_labels.shape[0]:
        raise ValueError("proposal_margin must align with ranking rows")
    return (
        (raw_ranked_labels[:, 0] != int(damaged_class))
        & (model_ranked_labels[:, 0] == int(damaged_class))
        & (proposal_margin >= float(threshold))
    )


def rescue_complement(
    raw_ranked_labels: torch.Tensor,
    final_ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
    query_events: torch.Tensor,
    accept: torch.Tensor,
    *,
    damaged_class: int = 1,
) -> Dict[str, object]:
    """Audit corrections relative to Raw Mean, including event-weighted counts."""

    raw_top = raw_ranked_labels[:, 0].long().cpu()
    final_top = final_ranked_labels[:, 0].long().cpu()
    truth = query_labels.long().flatten().cpu()
    events = query_events.long().flatten().cpu()
    accept = accept.bool().flatten().cpu()
    if not (raw_top.numel() == final_top.numel() == truth.numel() == events.numel() == accept.numel()):
        raise ValueError("rescue audit tensors must align")
    raw_correct = raw_top == truth
    final_correct = final_top == truth
    rescued = (~raw_correct) & final_correct
    harmed = raw_correct & (~final_correct)
    event_net: Dict[str, int] = {}
    event_damaged_rescue: Dict[str, int] = {}
    for event in sorted(map(int, torch.unique(events).tolist())):
        mask = events == event
        event_net[str(event)] = int(rescued[mask].sum() - harmed[mask].sum())
        event_damaged_rescue[str(event)] = int(
            (rescued & (truth == int(damaged_class)) & mask).sum()
        )
    accepted = int(accept.sum())
    damage_rescued = int((rescued & (truth == int(damaged_class))).sum())
    intact_harmed = int((harmed & (truth == 0)).sum())
    destroyed_harmed = int((harmed & (truth == 2)).sum())
    return {
        "proposed_changes": int(
            ((raw_top != int(damaged_class)) & (final_top == int(damaged_class))).sum()
        ),
        "accepted": accepted,
        "rescued": int(rescued.sum()),
        "harmed": int(harmed.sum()),
        "net_correction": int(rescued.sum() - harmed.sum()),
        "damaged_rescued": damage_rescued,
        "intact_harmed": intact_harmed,
        "destroyed_harmed": destroyed_harmed,
        "accepted_precision": damage_rescued / max(accepted, 1),
        "event_net_correction": event_net,
        "event_damaged_rescue": event_damaged_rescue,
        "nonnegative_events": int(sum(value >= 0 for value in event_net.values())),
        "positive_events": int(sum(value > 0 for value in event_net.values())),
        "damaged_positive_events": int(
            sum(value > 0 for value in event_damaged_rescue.values())
        ),
    }


def threshold_checks(
    complement: Mapping[str, object],
    *,
    minimum_rescued: int,
    minimum_precision: float,
    minimum_nonnegative_events: int,
    minimum_positive_events: int,
) -> Dict[str, bool]:
    return {
        "damaged_rescued_at_least_declared": int(complement["damaged_rescued"]) >= int(minimum_rescued),
        "accepted_precision_at_least_declared": float(complement["accepted_precision"]) >= float(minimum_precision),
        "overall_net_correction_positive": int(complement["net_correction"]) > 0,
        "nonnegative_events_at_least_declared": int(complement["nonnegative_events"]) >= int(minimum_nonnegative_events),
        "positive_events_at_least_declared": int(complement["positive_events"]) >= int(minimum_positive_events),
    }


def select_locked_threshold(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Prefer a fully passing Train point, then net benefit and coverage."""

    if not candidates:
        raise ValueError("threshold candidate list is empty")
    return max(
        candidates,
        key=lambda item: (
            bool(item["passed"]),
            int(item["complement"]["net_correction"]),
            int(item["complement"]["damaged_rescued"]),
            float(item["complement"]["accepted_precision"]),
            -int(item["complement"]["accepted"]),
        ),
    )


def metric_bundle(ranked_labels: torch.Tensor, query_labels: torch.Tensor) -> Dict[str, object]:
    return candidate_metrics(ranked_labels, query_labels)

