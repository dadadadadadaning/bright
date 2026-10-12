# -*- coding: utf-8 -*-
"""Train-only global Damaged-neighbour graph utilities for CMR-M1BG."""
from __future__ import annotations

from collections import Counter
from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_event_diverse_graph(
    feature: torch.Tensor,
    events: torch.Tensor,
    *,
    topk: int = 20,
    max_per_event: int = 2,
    chunk_size: int = 512,
    device: torch.device | str = "cpu",
    print_progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact cosine graph with a per-source-event neighbour cap.

    Inputs contain Damaged regions only. Same-event candidates are excluded.
    For every source event we first retain its best ``max_per_event`` items,
    then select the global top-k from that event-balanced candidate union.
    """

    if int(topk) <= 0 or int(max_per_event) <= 0 or int(chunk_size) <= 0:
        raise ValueError("topk, max_per_event and chunk_size must be positive")
    x_cpu = F.normalize(feature.float().cpu(), dim=1, eps=1e-8)
    events_cpu = events.long().flatten().cpu()
    if x_cpu.ndim != 2 or x_cpu.shape[0] != events_cpu.numel():
        raise ValueError("feature and events must align")
    unique_events = torch.unique(events_cpu).tolist()
    available_slots = (len(unique_events) - 1) * int(max_per_event)
    if available_slots < int(topk):
        raise RuntimeError(
            f"event cap offers only {available_slots} slots for topk={topk}"
        )
    device = torch.device(device)
    reference = x_cpu.to(device)
    event_pools = {
        int(event): torch.nonzero(events_cpu == int(event), as_tuple=False)
        .flatten()
        .to(device)
        for event in unique_events
    }
    graph_indices = []
    graph_scores = []
    for start in range(0, x_cpu.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), x_cpu.shape[0])
        query = x_cpu[start:end].to(device)
        query_events = events_cpu[start:end].to(device)
        similarity = query @ reference.t()
        candidate_scores = []
        candidate_indices = []
        for event, pool in event_pools.items():
            take = min(int(max_per_event), int(pool.numel()))
            values, local = torch.topk(similarity[:, pool], k=take, dim=1)
            indices = pool[local]
            same_event_query = query_events == int(event)
            values[same_event_query] = float("-inf")
            candidate_scores.append(values)
            candidate_indices.append(indices)
        scores = torch.cat(candidate_scores, dim=1)
        indices = torch.cat(candidate_indices, dim=1)
        selected_scores, selected_local = torch.topk(
            scores, k=int(topk), dim=1, largest=True, sorted=True
        )
        selected_indices = torch.gather(indices, 1, selected_local)
        if not bool(torch.isfinite(selected_scores).all()):
            raise RuntimeError("at least one Damaged anchor lacks a complete graph")
        graph_indices.append(selected_indices.cpu())
        graph_scores.append(selected_scores.cpu())
        if print_progress and (start == 0 or end == x_cpu.shape[0] or start // int(chunk_size) % 10 == 0):
            print(f"CMR-M1BG graph: {end}/{x_cpu.shape[0]}", flush=True)
    return torch.cat(graph_indices), torch.cat(graph_scores)


def _event_diverse_random_graph(
    events: torch.Tensor,
    *,
    topk: int,
    max_per_event: int,
    seed: int,
) -> torch.Tensor:
    """Random graph with the same cross-event and event-cap policy."""

    events = events.long().flatten().cpu()
    pools = {
        int(event): torch.nonzero(events == int(event), as_tuple=False).flatten()
        for event in torch.unique(events).tolist()
    }
    generator = torch.Generator().manual_seed(int(seed))
    rows = []
    for anchor_event in events.tolist():
        candidates = []
        available_events = [event for event in sorted(pools) if event != int(anchor_event)]
        order = torch.randperm(len(available_events), generator=generator).tolist()
        for position in order:
            pool = pools[available_events[position]]
            take = min(int(max_per_event), int(pool.numel()))
            chosen = torch.randperm(pool.numel(), generator=generator)[:take]
            candidates.extend(pool[chosen].tolist())
        if len(candidates) < int(topk):
            raise RuntimeError("random graph cannot satisfy declared topk")
        rows.append(candidates[: int(topk)])
    return torch.tensor(rows, dtype=torch.long)


def graph_audit(
    feature: torch.Tensor,
    events: torch.Tensor,
    neighbours: torch.Tensor,
    similarities: torch.Tensor,
    *,
    max_per_event: int,
    seed: int,
) -> Dict[str, object]:
    """Measure separation, event diversity and mutual-neighbour stability."""

    x = F.normalize(feature.float().cpu(), dim=1, eps=1e-8)
    events = events.long().flatten().cpu()
    neighbours = neighbours.long().cpu()
    similarities = similarities.float().cpu()
    if neighbours.shape != similarities.shape or neighbours.shape[0] != x.shape[0]:
        raise ValueError("graph tensors do not align")
    random_graph = _event_diverse_random_graph(
        events,
        topk=neighbours.shape[1],
        max_per_event=max_per_event,
        seed=seed,
    )
    random_similarity = (x[:, None, :] * x[random_graph]).sum(dim=2)
    anchor_mean = similarities.mean(dim=1)
    random_anchor_mean = random_similarity.mean(dim=1)
    unique_event_counts = torch.tensor(
        [torch.unique(events[row]).numel() for row in neighbours], dtype=torch.float32
    )
    dominant_shares = []
    for row in neighbours:
        counts = Counter(events[row].tolist())
        dominant_shares.append(max(counts.values()) / max(len(row), 1))
    dominant_shares = torch.tensor(dominant_shares)

    neighbour_sets = [set(row.tolist()) for row in neighbours]
    mutual = sum(
        int(anchor in neighbour_sets[target])
        for anchor, row in enumerate(neighbours.tolist())
        for target in row
    ) / max(neighbours.numel(), 1)
    random_sets = [set(row.tolist()) for row in random_graph]
    random_mutual = sum(
        int(anchor in random_sets[target])
        for anchor, row in enumerate(random_graph.tolist())
        for target in row
    ) / max(random_graph.numel(), 1)

    event_summary = {}
    for event in torch.unique(events).tolist():
        mask = events == int(event)
        event_summary[str(int(event))] = {
            "anchors": int(mask.sum()),
            "mean_topk_cosine": float(anchor_mean[mask].mean()),
            "mean_random_cosine": float(random_anchor_mean[mask].mean()),
            "mean_unique_neighbour_events": float(unique_event_counts[mask].mean()),
        }
    return {
        "anchors": int(x.shape[0]),
        "topk": int(neighbours.shape[1]),
        "topk_cosine_mean": float(similarities.mean()),
        "topk_anchor_mean_p10": float(torch.quantile(anchor_mean, 0.10)),
        "random_cosine_mean": float(random_similarity.mean()),
        "similarity_gap": float(similarities.mean() - random_similarity.mean()),
        "low_similarity_anchor_fraction": float(
            (anchor_mean <= random_anchor_mean.mean()).float().mean()
        ),
        "unique_neighbour_events_mean": float(unique_event_counts.mean()),
        "anchors_with_at_least_5_events": float((unique_event_counts >= 5).float().mean()),
        "dominant_event_share_max": float(dominant_shares.max()),
        "mutual_neighbour_rate": float(mutual),
        "random_mutual_neighbour_rate": float(random_mutual),
        "mutual_neighbour_gain": float(mutual - random_mutual),
        "event_summary": event_summary,
    }


def graph_gate(audit: Dict[str, object]) -> Dict[str, bool]:
    return {
        "similarity_gap_at_least_0_05": float(audit["similarity_gap"]) >= 0.05,
        "complete_graph_for_at_least_95pct": int(audit["anchors"]) > 0,
        "at_least_5_events_for_95pct": float(audit["anchors_with_at_least_5_events"]) >= 0.95,
        "dominant_event_share_at_most_0_25": float(audit["dominant_event_share_max"]) <= 0.25,
        "mutual_rate_plus_0_02_vs_random": float(audit["mutual_neighbour_gain"]) >= 0.02,
    }


def damaged_anchor_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """InfoNCE evaluated only for Damaged anchors."""

    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if anchor.ndim != 2 or positive.shape != anchor.shape:
        raise ValueError("anchor and positive must be aligned [B,C] tensors")
    if negatives.ndim != 3 or negatives.shape[:1] != anchor.shape[:1] or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must have shape [B,N,C]")
    anchor = F.normalize(anchor.float(), dim=1, eps=1e-8)
    positive = F.normalize(positive.float(), dim=1, eps=1e-8)
    negatives = F.normalize(negatives.float(), dim=2, eps=1e-8)
    positive_logit = (anchor * positive).sum(dim=1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", anchor, negatives)
    logits = torch.cat((positive_logit, negative_logits), dim=1) / float(temperature)
    target = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, target)


def cosine_preservation_loss(
    adapted: torch.Tensor,
    frozen_raw: torch.Tensor,
) -> torch.Tensor:
    """Keep guard-class embeddings aligned with their frozen Raw Mean keys."""

    if adapted.ndim != 2 or adapted.shape != frozen_raw.shape:
        raise ValueError("adapted and frozen_raw must have identical [N,C] shape")
    adapted = F.normalize(adapted.float(), dim=1, eps=1e-8)
    frozen_raw = F.normalize(frozen_raw.detach().float(), dim=1, eps=1e-8)
    return (1.0 - (adapted * frozen_raw).sum(dim=1)).mean()
