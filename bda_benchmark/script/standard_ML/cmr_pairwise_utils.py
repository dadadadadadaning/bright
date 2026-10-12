# -*- coding: utf-8 -*-
"""Protocol-locked candidate utilities for CMR-M3."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict

import torch
import torch.nn.functional as F

from script.standard_ML.cmr_multiview_utils import (
    _finalise_accumulator,
    _new_accumulator,
    _update_accumulator,
)


@torch.no_grad()
def frozen_topk_candidates(
    memory: torch.Tensor,
    memory_events: torch.Tensor,
    query: torch.Tensor,
    query_events: torch.Tensor,
    *,
    topk: int = 20,
    query_chunk: int = 1024,
    device: torch.device | str = "cuda:0",
    print_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Raw Mean Top-K after exact same-event exclusion."""

    if int(topk) <= 0 or int(query_chunk) <= 0:
        raise ValueError("topk and query_chunk must be positive")
    device = torch.device(device)
    memory = F.normalize(memory.float(), dim=1, eps=1e-8).to(device)
    query = F.normalize(query.float(), dim=1, eps=1e-8).cpu()
    memory_events = memory_events.long().flatten().to(device)
    query_events = query_events.long().flatten().cpu()
    if memory.shape[0] != memory_events.numel() or query.shape[0] != query_events.numel():
        raise ValueError("candidate tensors do not align")
    take = min(int(topk), memory.shape[0])
    all_indices, all_scores = [], []
    for start in range(0, query.shape[0], int(query_chunk)):
        end = min(start + int(query_chunk), query.shape[0])
        q = query[start:end].to(device, non_blocking=True)
        event = query_events[start:end].to(device, non_blocking=True)
        similarity = q @ memory.t()
        similarity.masked_fill_(event[:, None] == memory_events[None, :], float("-inf"))
        values, indices = torch.topk(similarity, k=take, dim=1, sorted=True)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("at least one query has fewer than Top-K cross-event candidates")
        all_indices.append(indices.cpu())
        all_scores.append(values.cpu())
        chunk = start // int(query_chunk)
        if print_progress and (start == 0 or end == query.shape[0] or chunk % 10 == 0):
            print(f"CMR-M3 candidates: {end}/{query.shape[0]}", flush=True)
    return torch.cat(all_indices), torch.cat(all_scores)


def candidate_metrics(
    ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    k_values: Sequence[int] = (1, 5, 10),
    ap_k: int = 10,
) -> Dict[str, object]:
    """Evaluate an already-ranked candidate matrix."""

    ranked_labels = ranked_labels.long().cpu()
    query_labels = query_labels.long().flatten().cpu()
    if ranked_labels.ndim != 2 or ranked_labels.shape[0] != query_labels.numel():
        raise ValueError("ranked labels and truth do not align")
    classes = sorted(map(int, torch.unique(query_labels).tolist()))
    accumulator = _new_accumulator(classes, k_values)
    _update_accumulator(
        accumulator,
        ranked_labels,
        torch.ones_like(ranked_labels, dtype=torch.bool),
        query_labels,
        k_values=k_values,
        ap_k=ap_k,
        num_classes=max(classes) + 1,
    )
    return _finalise_accumulator(accumulator, k_values=k_values, ap_k=ap_k)


def oracle_ranking(candidate_labels: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
    """Upper bound only: stable-promote correct candidates using GT labels."""

    candidate_labels = candidate_labels.long()
    query_labels = query_labels.long().flatten()
    correct = candidate_labels == query_labels[:, None]
    # Stable sorting retains Raw Mean order within correct/incorrect groups.
    order = torch.argsort(correct.long(), dim=1, descending=True, stable=True)
    return torch.gather(candidate_labels, 1, order)


def oracle_gate(
    baseline: Mapping[str, object],
    oracle: Mapping[str, object],
    *,
    damaged_headroom: float = 0.05,
    minimum_damaged_coverage: float = 0.95,
) -> Dict[str, bool]:
    return {
        "damaged_oracle_r1_headroom": (
            float(oracle["per_class"][1]["R@1"])
            >= float(baseline["per_class"][1]["R@1"]) + float(damaged_headroom)
        ),
        "damaged_topk_coverage": (
            float(oracle["per_class"][1]["R@1"]) >= float(minimum_damaged_coverage)
        ),
        "all_classes_have_candidates": all(
            float(oracle["per_class"][cls]["R@1"]) > 0.0 for cls in (0, 1, 2)
        ),
    }


def per_event_nonnegative_count(
    baseline_ranked_labels: torch.Tensor,
    candidate_ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
    query_events: torch.Tensor,
) -> tuple[int, Dict[str, float]]:
    """Count events whose class-balanced R@1 does not fall below Raw Mean."""

    deltas: Dict[str, float] = {}
    for event in sorted(map(int, torch.unique(query_events).tolist())):
        mask = query_events == event
        class_deltas = []
        for cls in (0, 1, 2):
            cls_mask = mask & (query_labels == cls)
            if not bool(cls_mask.any()):
                continue
            raw = (baseline_ranked_labels[cls_mask, 0] == cls).float().mean()
            new = (candidate_ranked_labels[cls_mask, 0] == cls).float().mean()
            class_deltas.append(float(new - raw))
        deltas[str(event)] = sum(class_deltas) / max(len(class_deltas), 1)
    return sum(value >= -1e-12 for value in deltas.values()), deltas


def per_event_r1_diagnostics(
    baseline_ranked_labels: torch.Tensor,
    candidate_ranked_labels: torch.Tensor,
    query_labels: torch.Tensor,
    query_events: torch.Tensor,
    *,
    damaged_class: int = 1,
) -> Dict[str, object]:
    """Report both macro and Damaged-specific event R@1 deltas."""

    tensors = (
        baseline_ranked_labels,
        candidate_ranked_labels,
        query_labels,
        query_events,
    )
    baseline_ranked_labels, candidate_ranked_labels, query_labels, query_events = (
        item.cpu() for item in tensors
    )
    macro_count, macro_delta = per_event_nonnegative_count(
        baseline_ranked_labels,
        candidate_ranked_labels,
        query_labels,
        query_events,
    )
    damaged_delta: Dict[str, float] = {}
    damaged_details: Dict[str, object] = {}
    comparable = 0
    for event in sorted(map(int, torch.unique(query_events).tolist())):
        mask = (query_events == event) & (query_labels == int(damaged_class))
        if not bool(mask.any()):
            continue
        raw = (baseline_ranked_labels[mask, 0] == int(damaged_class)).float().mean()
        new = (candidate_ranked_labels[mask, 0] == int(damaged_class)).float().mean()
        delta = float(new - raw)
        count = int(mask.sum())
        raw_correct = int((baseline_ranked_labels[mask, 0] == int(damaged_class)).sum())
        new_correct = int((candidate_ranked_labels[mask, 0] == int(damaged_class)).sum())
        damaged_delta[str(event)] = delta
        damaged_details[str(event)] = {
            "queries": count,
            "raw_correct": raw_correct,
            "candidate_correct": new_correct,
            "net_correct": new_correct - raw_correct,
            "delta_r1": delta,
        }
        comparable += 1
    return {
        "macro_nonnegative_events": int(macro_count),
        "macro_event_delta_r1": macro_delta,
        "damaged_nonnegative_events": int(
            sum(value >= -1e-12 for value in damaged_delta.values())
        ),
        "damaged_comparable_events": int(comparable),
        "damaged_event_delta_r1": damaged_delta,
        "damaged_event_details": damaged_details,
    }
