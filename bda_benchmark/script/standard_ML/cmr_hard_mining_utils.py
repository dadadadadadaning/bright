# -*- coding: utf-8 -*-
"""Utilities for CMR-H0 Train-only hard-example mining.

The module is deliberately independent from inference-time retrieval.  It
uses labels from the Train split to construct fixed cross-event triplets for
later P3 representation training.  No Validation/Test label or feature is
accepted by this protocol.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict

import torch
import torch.nn.functional as F


def select_missed_damaged_anchors(
    labels: torch.Tensor,
    base_predictions: torch.Tensor,
) -> torch.Tensor:
    """Return Train regions for which GT=Damaged but frozen P3 predicts Intact."""

    labels = labels.long().flatten().cpu()
    base_predictions = base_predictions.long().flatten().cpu()
    if labels.shape != base_predictions.shape:
        raise ValueError("labels and base_predictions must have identical shape")
    if labels.numel() == 0:
        raise ValueError("cannot mine an empty cache")
    return torch.nonzero(
        (labels == 1) & (base_predictions == 0), as_tuple=False
    ).flatten()


@torch.no_grad()
def event_diverse_cross_event_topk(
    query_feature: torch.Tensor,
    query_events: torch.Tensor,
    reference_feature: torch.Tensor,
    reference_events: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    topk: int,
    max_per_event: int,
    chunk_size: int = 256,
    device: torch.device | str = "cpu",
    print_prefix: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find exact cosine neighbours with same-event exclusion and event caps.

    ``reference_indices`` maps rows in ``reference_feature`` back to the full
    Train cache.  The third return value records the number of distinct source
    events represented in each selected row.
    """

    if min(int(topk), int(max_per_event), int(chunk_size)) <= 0:
        raise ValueError("topk, max_per_event and chunk_size must be positive")
    query = F.normalize(query_feature.float().cpu(), dim=1, eps=1e-8)
    reference = F.normalize(reference_feature.float().cpu(), dim=1, eps=1e-8)
    query_events = query_events.long().flatten().cpu()
    reference_events = reference_events.long().flatten().cpu()
    reference_indices = reference_indices.long().flatten().cpu()
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference features must be aligned 2-D matrices")
    if query.shape[0] != query_events.numel():
        raise ValueError("query features/events do not align")
    if reference.shape[0] != reference_events.numel() or reference.shape[0] != reference_indices.numel():
        raise ValueError("reference features/events/indices do not align")

    unique_events = sorted(int(item) for item in torch.unique(reference_events).tolist())
    device = torch.device(device)
    reference_device = reference.to(device)
    event_pools = {
        event: torch.nonzero(reference_events == event, as_tuple=False)
        .flatten()
        .to(device)
        for event in unique_events
    }
    output_indices = []
    output_scores = []
    output_unique_events = []
    for start in range(0, query.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), query.shape[0])
        q = query[start:end].to(device)
        qe = query_events[start:end].to(device)
        similarity = q @ reference_device.t()
        candidate_scores = []
        candidate_local_indices = []
        for event, pool in event_pools.items():
            take = min(int(max_per_event), int(pool.numel()))
            values, local = torch.topk(similarity[:, pool], k=take, dim=1)
            values[qe == event] = float("-inf")
            candidate_scores.append(values)
            candidate_local_indices.append(pool[local])
        scores = torch.cat(candidate_scores, dim=1)
        local_indices = torch.cat(candidate_local_indices, dim=1)
        selected_scores, selected_columns = torch.topk(
            scores, k=int(topk), dim=1, largest=True, sorted=True
        )
        selected_local = torch.gather(local_indices, 1, selected_columns)
        if not bool(torch.isfinite(selected_scores).all()):
            raise RuntimeError(
                "at least one query lacks enough cross-event references for declared topk"
            )
        selected_events = reference_events[selected_local.cpu()]
        unique_counts = torch.tensor(
            [torch.unique(row).numel() for row in selected_events], dtype=torch.long
        )
        output_indices.append(reference_indices[selected_local.cpu()])
        output_scores.append(selected_scores.cpu())
        output_unique_events.append(unique_counts)
        if print_prefix and (
            start == 0 or end == query.shape[0] or (start // int(chunk_size)) % 10 == 0
        ):
            print(f"{print_prefix}: {end}/{query.shape[0]}", flush=True)
    return (
        torch.cat(output_indices),
        torch.cat(output_scores),
        torch.cat(output_unique_events),
    )


def _event_diverse_random_indices(
    query_events: torch.Tensor,
    reference_events: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    topk: int,
    max_per_event: int,
    seed: int,
) -> torch.Tensor:
    """Sample a deterministic random graph under the same event constraints."""

    query_events = query_events.long().flatten().cpu()
    reference_events = reference_events.long().flatten().cpu()
    reference_indices = reference_indices.long().flatten().cpu()
    pools = {
        int(event): torch.nonzero(reference_events == int(event), as_tuple=False).flatten()
        for event in torch.unique(reference_events).tolist()
    }
    generator = torch.Generator().manual_seed(int(seed))
    rows = []
    for query_event in query_events.tolist():
        chosen = []
        event_order = [event for event in sorted(pools) if event != int(query_event)]
        event_order = [event_order[i] for i in torch.randperm(len(event_order), generator=generator).tolist()]
        for event in event_order:
            pool = pools[event]
            take = min(int(max_per_event), int(pool.numel()))
            local = torch.randperm(pool.numel(), generator=generator)[:take]
            chosen.extend(reference_indices[pool[local]].tolist())
            if len(chosen) >= int(topk):
                break
        if len(chosen) < int(topk):
            raise RuntimeError("random graph cannot satisfy declared cross-event topk")
        rows.append(chosen[: int(topk)])
    return torch.tensor(rows, dtype=torch.long)


def hard_mining_audit(
    feature: torch.Tensor,
    labels: torch.Tensor,
    base_predictions: torch.Tensor,
    events: torch.Tensor,
    anchor_indices: torch.Tensor,
    positive_indices: torch.Tensor,
    positive_scores: torch.Tensor,
    positive_unique_events: torch.Tensor,
    negative_indices: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_unique_events: torch.Tensor,
    *,
    max_per_event: int,
    seed: int,
) -> Dict[str, object]:
    """Summarise hard-pair coverage without using Validation/Test."""

    feature = F.normalize(feature.float().cpu(), dim=1, eps=1e-8)
    labels = labels.long().flatten().cpu()
    base_predictions = base_predictions.long().flatten().cpu()
    events = events.long().flatten().cpu()
    anchor_indices = anchor_indices.long().flatten().cpu()
    positive_indices = positive_indices.long().cpu()
    negative_indices = negative_indices.long().cpu()
    positive_scores = positive_scores.float().cpu()
    negative_scores = negative_scores.float().cpu()
    if anchor_indices.numel() == 0:
        raise RuntimeError("CMR-H0 found no P3-missed Damaged anchors")
    if positive_indices.shape != positive_scores.shape or negative_indices.shape != negative_scores.shape:
        raise ValueError("neighbour indices and scores must align")
    if positive_indices.shape[0] != anchor_indices.numel() or negative_indices.shape[0] != anchor_indices.numel():
        raise ValueError("anchor/neighbour rows must align")
    if not bool((labels[anchor_indices] == 1).all()) or not bool((base_predictions[anchor_indices] == 0).all()):
        raise RuntimeError("anchor contract must be GT Damaged and frozen-P3 Intact")
    if not bool((labels[positive_indices] == 1).all()):
        raise RuntimeError("positive graph contains a non-Damaged item")
    if not bool((labels[negative_indices] == 0).all()):
        raise RuntimeError("negative graph contains a non-Intact item")
    if bool((events[positive_indices] == events[anchor_indices, None]).any()):
        raise RuntimeError("positive graph contains same-event leakage")
    if bool((events[negative_indices] == events[anchor_indices, None]).any()):
        raise RuntimeError("negative graph contains same-event leakage")

    damaged_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    random_positive_indices = _event_diverse_random_indices(
        events[anchor_indices],
        events[damaged_indices],
        damaged_indices,
        topk=positive_indices.shape[1],
        max_per_event=max_per_event,
        seed=seed,
    )
    random_positive_scores = (
        feature[anchor_indices, None, :] * feature[random_positive_indices]
    ).sum(dim=2)
    best_positive = positive_scores[:, 0]
    hardest_negative = negative_scores[:, 0]
    pair_margin = best_positive - hardest_negative
    anchor_event_counts = Counter(events[anchor_indices].tolist())
    all_event_counts = {
        str(int(event)): int(anchor_event_counts.get(int(event), 0))
        for event in torch.unique(events).tolist()
    }
    total_damaged = int((labels == 1).sum())
    return {
        "train_regions": int(labels.numel()),
        "total_damaged_regions": total_damaged,
        "missed_damaged_anchors": int(anchor_indices.numel()),
        "p3_damaged_to_intact_fraction": float(anchor_indices.numel() / max(total_damaged, 1)),
        "anchor_events": int(len(anchor_event_counts)),
        "anchor_event_counts": all_event_counts,
        "maximum_anchor_event_share": float(max(anchor_event_counts.values()) / anchor_indices.numel()),
        "positive_topk": int(positive_indices.shape[1]),
        "negative_topk": int(negative_indices.shape[1]),
        "positive_cosine_mean": float(positive_scores.mean()),
        "random_damaged_cosine_mean": float(random_positive_scores.mean()),
        "positive_similarity_gain_over_random": float(positive_scores.mean() - random_positive_scores.mean()),
        "hard_intact_cosine_mean": float(negative_scores.mean()),
        "top1_positive_minus_hard_intact_mean": float(pair_margin.mean()),
        "top1_positive_beats_hard_intact_fraction": float((pair_margin > 0).float().mean()),
        "positive_unique_events_mean": float(positive_unique_events.float().mean()),
        "negative_unique_events_mean": float(negative_unique_events.float().mean()),
        "complete_positive_rows_fraction": float(torch.isfinite(positive_scores).all(dim=1).float().mean()),
        "complete_negative_rows_fraction": float(torch.isfinite(negative_scores).all(dim=1).float().mean()),
    }


def hard_mining_gate(audit: Dict[str, object]) -> Dict[str, bool]:
    """Predeclared H0 gate. Pair margin is diagnostic, not a pass criterion."""

    return {
        "at_least_100_missed_damaged_anchors": int(audit["missed_damaged_anchors"]) >= 100,
        "anchors_cover_at_least_10_events": int(audit["anchor_events"]) >= 10,
        "positive_graph_complete": float(audit["complete_positive_rows_fraction"]) >= 1.0,
        "negative_graph_complete": float(audit["complete_negative_rows_fraction"]) >= 1.0,
        "positive_neighbours_cover_at_least_3_events": float(audit["positive_unique_events_mean"]) >= 3.0,
        "positive_similarity_plus_0_03_vs_random": float(audit["positive_similarity_gain_over_random"]) >= 0.03,
        "maximum_anchor_event_share_at_most_0_35": float(audit["maximum_anchor_event_share"]) <= 0.35,
    }
