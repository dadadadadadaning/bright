# -*- coding: utf-8 -*-
"""Pure utilities for M-CB0 fixed-review-budget complement auditing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict

import torch


METHOD_NAMES = ("p3_probability", "memory_knn", "p3_80_memory_20")


def candidate_table(cache: Mapping[str, object]) -> Dict[str, object]:
    """Build the label-free P3-Intact proposal universe.

    Ground truth is returned separately and is used only after selection for
    diagnostic accounting.  It never participates in any ranking.
    """

    required = (
        "labels",
        "base_predictions",
        "base_probabilities",
        "events",
        "image_ids",
        "key_variants",
    )
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"M-CB0 cache is missing: {missing}")
    truth = torch.as_tensor(cache["labels"]).long().flatten().cpu()
    base = torch.as_tensor(cache["base_predictions"]).long().flatten().cpu()
    probability = torch.as_tensor(cache["base_probabilities"]).float().cpu()
    mean = torch.as_tensor(cache["key_variants"]["mean"]).float().cpu()
    events = list(map(str, cache["events"]))
    image_ids = list(map(str, cache["image_ids"]))
    count = truth.numel()
    if probability.shape != (count, 3):
        raise ValueError("M-CB0 expects P3 probabilities [N,3]")
    if mean.shape[0] != count or base.numel() != count:
        raise ValueError("M-CB0 tensor rows are not aligned")
    if len(events) != count or len(image_ids) != count:
        raise ValueError("M-CB0 metadata rows are not aligned")
    if not set(torch.unique(truth).tolist()).issubset({0, 1, 2}):
        raise ValueError("M-CB0 expects labels 0=I,1=D,2=X")

    candidate_indices = torch.nonzero(base == 0, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        raise RuntimeError("M-CB0 found no P3-Intact candidates")
    return {
        "indices": candidate_indices,
        "truth": truth[candidate_indices],
        "events": [events[int(index)] for index in candidate_indices],
        "image_ids": [image_ids[int(index)] for index in candidate_indices],
        "keys": mean[candidate_indices],
        "p3_scores": probability[candidate_indices, 1:3].sum(dim=1),
        "candidate_count": int(candidate_indices.numel()),
    }


def top_budget_indices(scores: torch.Tensor, budget: int) -> torch.Tensor:
    values = scores.float().flatten().cpu()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("M-CB0 scores must be finite and non-empty")
    take = min(max(int(budget), 0), int(values.numel()))
    if take <= 0:
        raise ValueError("M-CB0 budget must be positive")
    return torch.argsort(values, descending=True, stable=True)[:take]


def blended_budget_indices(
    p3_scores: torch.Tensor,
    memory_scores: torch.Tensor,
    *,
    budget: int,
    p3_fraction: float,
) -> Dict[str, torch.Tensor]:
    """Allocate a fixed budget to P3 first, then unique Memory proposals."""

    p3_scores = p3_scores.float().flatten().cpu()
    memory_scores = memory_scores.float().flatten().cpu()
    if p3_scores.shape != memory_scores.shape or p3_scores.numel() == 0:
        raise ValueError("M-CB0 blended scores must be aligned and non-empty")
    if not 0.0 <= float(p3_fraction) <= 1.0:
        raise ValueError("M-CB0 P3 fraction must be in [0,1]")
    total = min(int(budget), int(p3_scores.numel()))
    if total <= 0:
        raise ValueError("M-CB0 budget must be positive")
    p3_take = min(total, int(total * float(p3_fraction)))
    p3_order = torch.argsort(p3_scores, descending=True, stable=True)
    p3_selected = p3_order[:p3_take]
    occupied = torch.zeros(p3_scores.numel(), dtype=torch.bool)
    occupied[p3_selected] = True
    memory_order = torch.argsort(memory_scores, descending=True, stable=True)
    memory_unique = memory_order[~occupied[memory_order]][: total - p3_take]
    selected = torch.cat((p3_selected, memory_unique))
    if selected.numel() != total or torch.unique(selected).numel() != total:
        raise AssertionError("M-CB0 blend did not preserve exact unique workload")
    return {
        "selected": selected,
        "p3_selected": p3_selected,
        "memory_unique_selected": memory_unique,
    }


def selection_summary(
    selected: torch.Tensor,
    truth: torch.Tensor,
    events: Sequence[str],
) -> Dict[str, object]:
    selected = selected.long().flatten().cpu()
    truth = truth.long().flatten().cpu()
    if truth.numel() != len(events):
        raise ValueError("M-CB0 truth/events are not aligned")
    chosen_truth = truth[selected]
    damaged = int((chosen_truth == 1).sum())
    destroyed = int((chosen_truth == 2).sum())
    intact = int((chosen_truth == 0).sum())
    affected = damaged + destroyed
    rescued_by_event = {event: 0 for event in sorted(set(map(str, events)))}
    damaged_by_event = {event: 0 for event in sorted(set(map(str, events)))}
    for index in selected.tolist():
        label = int(truth[index])
        if label > 0:
            rescued_by_event[str(events[index])] += 1
        if label == 1:
            damaged_by_event[str(events[index])] += 1
    positive_values = [value for value in rescued_by_event.values() if value > 0]
    return {
        "workload": int(selected.numel()),
        "affected_rescued": affected,
        "damaged_rescued": damaged,
        "destroyed_rescued": destroyed,
        "intact_reviewed": intact,
        "precision_at_budget": affected / max(int(selected.numel()), 1),
        "rescue_events": sum(value > 0 for value in rescued_by_event.values()),
        "damaged_rescue_events": sum(value > 0 for value in damaged_by_event.values()),
        "maximum_rescue_event_share": (
            max(positive_values) / max(sum(positive_values), 1)
            if positive_values else 0.0
        ),
        "affected_rescued_by_event": rescued_by_event,
        "damaged_rescued_by_event": damaged_by_event,
        "selected_local_indices": selected.tolist(),
    }


def selection_complement(
    proposed: torch.Tensor,
    baseline: torch.Tensor,
    truth: torch.Tensor,
    events: Sequence[str],
) -> Dict[str, object]:
    """Measure proposals uniquely contributed relative to a same-budget baseline."""

    proposed = proposed.long().flatten().cpu()
    baseline = baseline.long().flatten().cpu()
    truth = truth.long().flatten().cpu()
    proposed_set = set(proposed.tolist())
    baseline_set = set(baseline.tolist())
    unique = torch.tensor(sorted(proposed_set - baseline_set), dtype=torch.long)
    displaced = torch.tensor(sorted(baseline_set - proposed_set), dtype=torch.long)
    overlap = len(proposed_set.intersection(baseline_set))

    unique_truth = truth[unique] if unique.numel() else torch.empty(0, dtype=torch.long)
    displaced_truth = (
        truth[displaced] if displaced.numel() else torch.empty(0, dtype=torch.long)
    )
    unique_damaged_by_event = {event: 0 for event in sorted(set(map(str, events)))}
    for index in unique.tolist():
        if int(truth[index]) == 1:
            unique_damaged_by_event[str(events[index])] += 1
    unique_damaged_values = [
        value for value in unique_damaged_by_event.values() if value > 0
    ]
    return {
        "overlap": overlap,
        "overlap_ratio": overlap / max(int(baseline.numel()), 1),
        "unique_selected": int(unique.numel()),
        "displaced_selected": int(displaced.numel()),
        "unique_affected_rescued": int((unique_truth > 0).sum()),
        "unique_damaged_rescued": int((unique_truth == 1).sum()),
        "unique_destroyed_rescued": int((unique_truth == 2).sum()),
        "unique_intact_reviewed": int((unique_truth == 0).sum()),
        "displaced_affected": int((displaced_truth > 0).sum()),
        "displaced_damaged": int((displaced_truth == 1).sum()),
        "displaced_destroyed": int((displaced_truth == 2).sum()),
        "unique_damaged_events": sum(
            value > 0 for value in unique_damaged_by_event.values()
        ),
        "maximum_unique_damaged_event_share": (
            max(unique_damaged_values) / max(sum(unique_damaged_values), 1)
            if unique_damaged_values else 0.0
        ),
        "unique_damaged_by_event": unique_damaged_by_event,
        "unique_local_indices": unique.tolist(),
        "displaced_local_indices": displaced.tolist(),
    }


def evaluate_budgets(
    p3_scores: torch.Tensor,
    memory_scores: torch.Tensor,
    truth: torch.Tensor,
    events: Sequence[str],
    *,
    budgets: Sequence[int],
    p3_fraction: float,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for budget in budgets:
        p3 = top_budget_indices(p3_scores, int(budget))
        memory = top_budget_indices(memory_scores, int(budget))
        blend = blended_budget_indices(
            p3_scores,
            memory_scores,
            budget=int(budget),
            p3_fraction=p3_fraction,
        )
        selected = {
            "p3_probability": p3,
            "memory_knn": memory,
            "p3_80_memory_20": blend["selected"],
        }
        summaries = {
            name: selection_summary(indices, truth, events)
            for name, indices in selected.items()
        }
        summaries["p3_80_memory_20"]["p3_quota"] = int(
            blend["p3_selected"].numel()
        )
        summaries["p3_80_memory_20"]["memory_unique_quota"] = int(
            blend["memory_unique_selected"].numel()
        )
        results[str(int(budget))] = {
            "methods": summaries,
            "memory_vs_p3": selection_complement(memory, p3, truth, events),
            "combined_vs_p3": selection_complement(
                blend["selected"], p3, truth, events
            ),
        }
    return results


__all__ = [
    "METHOD_NAMES",
    "blended_budget_indices",
    "candidate_table",
    "evaluate_budgets",
    "selection_complement",
    "selection_summary",
    "top_budget_indices",
]
