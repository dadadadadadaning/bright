from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def p3_diagnostics(probability: torch.Tensor) -> torch.Tensor:
    """P3-only control features. They must be present in every learned arm."""
    p = probability.float().clamp_min(1e-6)
    ordered = p.sort(1, descending=True).values
    entropy = -(p * p.log()).sum(1, keepdim=True)
    margin = ordered[:, :1] - ordered[:, 1:2]
    affected = p[:, 2:4].sum(1, keepdim=True)
    return torch.cat((p, p.log(), entropy, margin, affected), 1)


def fit_standardizer(train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = train.float()
    mean = value.mean(0)
    std = value.std(0, unbiased=False).clamp_min(1e-5)
    return mean, std


def standardize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num((value.float() - mean) / std, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8, 8)


class ResidualCalibrator(nn.Module):
    """Identical residual architecture for P3-only, true-card, and shuffled-card arms."""

    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.delta = nn.Linear(hidden, 4)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, value: torch.Tensor, base_probability: torch.Tensor) -> torch.Tensor:
        base = base_probability.float().clamp_min(1e-6).log()
        return base + self.delta(self.encoder(value.float()))


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = -(target.float() * logits.log_softmax(1)).sum(1)
    return (loss * weight.float()).sum() / weight.float().sum().clamp_min(1)


def bounded_event_class_rows(labels: torch.Tensor, events: list[str], maximum: int, seed: int) -> torch.Tensor:
    """Keep a bounded, reproducible memory. More rows are not automatically better."""
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for event in sorted(set(events)):
        event_mask = torch.tensor([name == event for name in events])
        for label in range(4):
            rows = torch.nonzero(event_mask & (labels.long() == label), as_tuple=False).flatten()
            if len(rows) > maximum:
                rows = rows[torch.randperm(len(rows), generator=generator)[:maximum]]
            chosen.append(rows)
    return torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long)


def _cosine(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    return F.normalize(query.float(), dim=1, eps=1e-8) @ F.normalize(memory.float(), dim=1, eps=1e-8).T


@torch.no_grad()
def hierarchical_cross_event_topk(
    query_structure: torch.Tensor,
    memory_structure: torch.Tensor,
    query_damage: torch.Tensor,
    memory_damage: torch.Tensor,
    query_quality: torch.Tensor,
    memory_quality: torch.Tensor,
    query_events: list[str],
    memory_events: list[str],
    *,
    coarse_k: int = 64,
    topk: int = 8,
    structure_weight: float = .25,
    damage_weight: float = .65,
    quality_weight: float = .10,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB geometry retrieves candidates; SAR evidence and quality re-rank them."""
    if coarse_k < topk:
        raise ValueError("coarse_k must be >= topk")
    m_events = list(memory_events)
    all_indices, all_scores = [], []
    for start in range(0, len(query_structure), chunk):
        stop = min(start + chunk, len(query_structure))
        coarse = _cosine(query_structure[start:stop], memory_structure)
        for local, query_event in enumerate(query_events[start:stop]):
            forbidden = torch.tensor([event == query_event for event in m_events], dtype=torch.bool)
            coarse[local, forbidden] = -torch.inf
        k0 = min(coarse_k, memory_structure.shape[0])
        coarse_score, coarse_index = coarse.topk(k0, dim=1)
        q_damage = F.normalize(query_damage[start:stop].float(), dim=1, eps=1e-8)
        m_damage = F.normalize(memory_damage.float(), dim=1, eps=1e-8)
        damage_score = (m_damage[coarse_index] * q_damage[:, None]).sum(2)
        q_quality = query_quality[start:stop].float()
        quality_distance = ((memory_quality.float()[coarse_index] - q_quality[:, None]) ** 2).mean(2).sqrt()
        quality_score = torch.exp(-quality_distance / math.sqrt(max(query_quality.shape[1], 1)))
        score = structure_weight * coarse_score + damage_weight * damage_score + quality_weight * quality_score
        k = min(topk, k0)
        selected_score, selected_local = score.topk(k, dim=1)
        selected_index = coarse_index.gather(1, selected_local)
        all_indices.append(selected_index.cpu())
        all_scores.append(selected_score.cpu())
    return torch.cat(all_indices), torch.cat(all_scores)


def memory_residual_probability(
    base_probability: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    memory_target_probability: torch.Tensor,
    memory_p3_probability: torch.Tensor,
    *,
    beta: float = .25,
    temperature: float = .10,
) -> torch.Tensor:
    """Retrieve how P3 was wrong on neighbours, not their raw class label."""
    weight = torch.softmax(scores.float() / temperature, 1)
    residual = memory_target_probability.float() - memory_p3_probability.float()
    delta = (residual[indices] * weight[..., None]).sum(1)
    output = (base_probability.float() + beta * delta).clamp_min(1e-6)
    return output / output.sum(1, keepdim=True)

