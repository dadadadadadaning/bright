# -*- coding: utf-8 -*-
"""Deterministic, no-training diagnostics for Damaged multi-prototypes.

This module deliberately operates on frozen CMR keys.  It does not modify P3,
does not expose validation labels to a predictor, and does not implement an
inference-time Memory classifier.  Its only job is to test whether the broad
``Damaged`` label contains reproducible cross-event modes.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def spherical_kmeans(
    features: torch.Tensor,
    clusters: int,
    *,
    seed: int = 3407,
    iterations: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster L2-normalised rows with deterministic farthest-first seeds."""

    x = F.normalize(features.detach().float().cpu(), dim=1, eps=1e-8)
    if x.ndim != 2 or x.shape[0] < int(clusters) or int(clusters) < 2:
        raise ValueError("spherical_kmeans needs N >= clusters >= 2")
    generator = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(x.shape[0], (1,), generator=generator))
    chosen = [first]
    best_distance = 1.0 - x @ x[first]
    for _ in range(1, int(clusters)):
        candidate = int(best_distance.argmax())
        chosen.append(candidate)
        best_distance = torch.minimum(best_distance, 1.0 - x @ x[candidate])
    centres = x[torch.tensor(chosen)].clone()

    assignment = torch.full((x.shape[0],), -1, dtype=torch.long)
    for _ in range(int(iterations)):
        updated_assignment = (x @ centres.t()).argmax(dim=1)
        if torch.equal(updated_assignment, assignment):
            break
        assignment = updated_assignment
        new_centres = []
        for cluster in range(int(clusters)):
            members = x[assignment == cluster]
            if members.numel() == 0:
                # Re-seed an empty cluster with the currently worst represented row.
                nearest = (x @ centres.t()).max(dim=1).values
                centre = x[int(nearest.argmin())]
            else:
                centre = members.mean(dim=0)
            new_centres.append(F.normalize(centre, dim=0, eps=1e-8))
        centres = torch.stack(new_centres)
    return assignment, centres


def assign_to_centres(features: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    x = F.normalize(features.detach().float().cpu(), dim=1, eps=1e-8)
    c = F.normalize(centres.detach().float().cpu(), dim=1, eps=1e-8)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError("prototype assignment dimensions differ")
    return (x @ c.t()).argmax(dim=1)


def event_capped_indices(
    events: torch.Tensor,
    *,
    cap_per_event: int,
    seed: int = 3407,
    excluded_event: int | None = None,
) -> torch.Tensor:
    """Sample each event without replacement, never upsampling small events."""

    event = events.long().flatten().cpu()
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for value in sorted(int(item) for item in torch.unique(event).tolist()):
        if excluded_event is not None and value == int(excluded_event):
            continue
        indices = torch.nonzero(event == value, as_tuple=False).flatten()
        take = min(int(cap_per_event), int(indices.numel()))
        order = torch.randperm(indices.numel(), generator=generator)[:take]
        selected.append(indices[order])
    if not selected:
        raise RuntimeError("event-capped sampler selected no rows")
    result = torch.cat(selected)
    if torch.unique(result).numel() != result.numel():
        raise AssertionError("event-capped sampling repeated a row")
    return result


def adjusted_rand_index(first: torch.Tensor, second: torch.Tensor) -> float:
    """Adjusted Rand Index without a scikit-learn dependency."""

    a = first.long().flatten().cpu()
    b = second.long().flatten().cpu()
    if a.numel() != b.numel() or a.numel() < 2:
        raise ValueError("ARI inputs must have the same length >= 2")
    _, ai = torch.unique(a, sorted=True, return_inverse=True)
    _, bi = torch.unique(b, sorted=True, return_inverse=True)
    contingency = torch.zeros(
        int(ai.max()) + 1, int(bi.max()) + 1, dtype=torch.float64
    )
    contingency.index_put_((ai, bi), torch.ones_like(ai, dtype=torch.float64), accumulate=True)

    choose2 = lambda value: value * (value - 1.0) / 2.0
    sum_cells = choose2(contingency).sum()
    sum_rows = choose2(contingency.sum(dim=1)).sum()
    sum_cols = choose2(contingency.sum(dim=0)).sum()
    total = choose2(torch.tensor(float(a.numel())))
    expected = sum_rows * sum_cols / total.clamp_min(1e-12)
    maximum = 0.5 * (sum_rows + sum_cols)
    denominator = maximum - expected
    if float(denominator.abs()) <= 1e-12:
        return 1.0 if torch.equal(ai, bi) else 0.0
    return float((sum_cells - expected) / denominator)


def normalised_mutual_information(labels: torch.Tensor, events: torch.Tensor) -> float:
    """Symmetric NMI between cluster assignments and disaster events."""

    labels = labels.long().flatten().cpu()
    events = events.long().flatten().cpu()
    if labels.numel() != events.numel() or labels.numel() == 0:
        raise ValueError("NMI inputs must be aligned and non-empty")
    _, li = torch.unique(labels, sorted=True, return_inverse=True)
    _, ei = torch.unique(events, sorted=True, return_inverse=True)
    table = torch.zeros(int(li.max()) + 1, int(ei.max()) + 1, dtype=torch.float64)
    table.index_put_((li, ei), torch.ones_like(li, dtype=torch.float64), accumulate=True)
    joint = table / table.sum()
    pl = joint.sum(dim=1, keepdim=True)
    pe = joint.sum(dim=0, keepdim=True)
    valid = joint > 0
    mutual = (joint[valid] * torch.log(joint[valid] / (pl @ pe)[valid])).sum()
    hl = -(pl[pl > 0] * torch.log(pl[pl > 0])).sum()
    he = -(pe[pe > 0] * torch.log(pe[pe > 0])).sum()
    denominator = torch.sqrt(hl * he)
    return float(mutual / denominator.clamp_min(1e-12))


def cross_event_similarity_gain(
    features: torch.Tensor,
    assignments: torch.Tensor,
    events: torch.Tensor,
    *,
    seed: int = 3407,
    pairs: int = 50000,
) -> Dict[str, float | int]:
    """Compare same-prototype cross-event similarity with random D--D pairs."""

    x = F.normalize(features.detach().float().cpu(), dim=1, eps=1e-8)
    assignment = assignments.long().flatten().cpu()
    event = events.long().flatten().cpu()
    if not (x.shape[0] == assignment.numel() == event.numel()):
        raise ValueError("feature/assignment/event rows do not align")
    generator = torch.Generator().manual_seed(int(seed))
    first = torch.randint(x.shape[0], (int(pairs) * 4,), generator=generator)
    second = torch.randint(x.shape[0], (int(pairs) * 4,), generator=generator)
    cross = event[first] != event[second]
    first, second = first[cross], second[cross]
    if first.numel() < int(pairs):
        raise RuntimeError("not enough cross-event pairs for prototype audit")
    first, second = first[: int(pairs)], second[: int(pairs)]
    similarity = (x[first] * x[second]).sum(dim=1)
    same = assignment[first] == assignment[second]
    if int(same.sum()) < 10:
        raise RuntimeError("too few within-prototype cross-event pairs")
    return {
        "random_cross_event_similarity": float(similarity.mean()),
        "within_prototype_cross_event_similarity": float(similarity[same].mean()),
        "similarity_gain": float(similarity[same].mean() - similarity.mean()),
        "sampled_cross_event_pairs": int(similarity.numel()),
        "within_prototype_pairs": int(same.sum()),
    }


def prototype_event_diagnostics(
    assignments: torch.Tensor,
    events: torch.Tensor,
    *,
    clusters: int,
) -> list[Dict[str, object]]:
    result = []
    assignments = assignments.long().flatten().cpu()
    events = events.long().flatten().cpu()
    for cluster in range(int(clusters)):
        member_events = events[assignments == cluster]
        count = int(member_events.numel())
        if count == 0:
            result.append({"prototype": cluster, "samples": 0, "events": 0, "max_event_share": 1.0})
            continue
        histogram = torch.bincount(member_events)
        result.append({
            "prototype": int(cluster),
            "samples": count,
            "events": int((histogram > 0).sum()),
            "max_event_share": float(histogram.max() / count),
            "event_counts": {str(i): int(v) for i, v in enumerate(histogram.tolist()) if v},
        })
    return result


def mean_pairwise_ari(assignments: Sequence[torch.Tensor]) -> float:
    pairs = [adjusted_rand_index(a, b) for a, b in combinations(assignments, 2)]
    return float(sum(pairs) / len(pairs)) if pairs else 1.0
