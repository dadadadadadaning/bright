from __future__ import annotations

from collections import defaultdict

import torch


def random_cross_event_support(
    query_events: list[str], memory_events: list[str], topk: int, seed: int,
) -> torch.Tensor:
    """Return deterministic random memory-local indices, excluding the query event."""
    output = torch.full((len(query_events), topk), -1, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    memory_by_query_event = {
        event: torch.tensor([i for i, candidate in enumerate(memory_events) if candidate != event])
        for event in sorted(set(query_events))
    }
    for row, event in enumerate(query_events):
        candidates = memory_by_query_event[event]
        if not len(candidates):
            continue
        selected = torch.randint(len(candidates), (topk,), generator=generator)
        output[row] = candidates[selected]
    return output


def shuffle_within_event(value: torch.Tensor, events: list[str], seed: int) -> torch.Tensor:
    """Break query-to-support correspondence while preserving each event's distribution."""
    output = value.clone()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        groups[event].append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        index = torch.tensor(indices)
        permutation = torch.randperm(len(index), generator=generator)
        output[index] = value[index[permutation]]
    return output


def p3_uncertainty_features(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp_min(1e-8)
    ordered = probability.sort(1, descending=True).values
    entropy = -(probability * probability.log()).sum(1, keepdim=True)
    margin = ordered[:, :1] - ordered[:, 1:2]
    affected = probability[:, 2:4].sum(1, keepdim=True)
    severity_margin = (probability[:, 3:4] - probability[:, 2:3]).abs()
    return torch.cat((probability, probability.max(1, keepdim=True).values,
                      entropy, margin, affected, severity_margin), 1)


def route_probability(
    p3_probability: torch.Tensor, expert_probability: torch.Tensor,
    score: torch.Tensor, threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    routed = score >= threshold
    output = p3_probability.clone()
    output[routed] = expert_probability[routed]
    return output, routed


def event_nonnegative(left: dict, right: dict) -> tuple[int, int]:
    names = sorted(set(left["events"]) & set(right["events"]))
    return sum(left["events"][name] >= right["events"][name] for name in names), len(names)


def gain_pp(left: dict, right: dict) -> float:
    return (left["macro_f1"] - right["macro_f1"]) * 100.0
