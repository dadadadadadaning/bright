from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.core import (
    balanced_event_folds,
    retrieval_diverse_topk,
    retrieval_relative_gap,
    threshold_candidates,
    valid_miou,
)
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    cross_event_supervised_contrastive,
)


SOURCE_VARIANTS = {
    "c0_base": ("base_change",),
    "c1_diff": ("explicit_diff",),
    "c2_enhanced": ("enhanced_change",),
    "c3_dual": ("base_change", "explicit_diff"),
    "c4_structural": ("rgb_structure", "base_change", "explicit_diff"),
}
POOL_VARIANTS = ("p0_mean", "p1_stats", "p2_attn_skip", "p3_attn_tail")
RAG_VARIANTS = ("x0_keys", "x1_label_geometry", "x2_label_geometry_event")


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def masked_std(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(value, mask)
    weight = mask.to(value.dtype).unsqueeze(-1)
    variance = ((value - mean[:, None]) ** 2 * weight).sum(1)
    variance = variance / weight.sum(1).clamp_min(1.0)
    return variance.clamp_min(0.0).sqrt()


def masked_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values.nan_to_num()


def select_region_positions(
    score: torch.Tensor,
    mask: torch.Tensor,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically keep both rare anomalies and representative pixels."""
    positions = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    if positions.numel() == 0:
        return torch.zeros(maximum, dtype=torch.long, device=mask.device), torch.zeros(
            maximum, dtype=torch.bool, device=mask.device
        )
    take = min(int(maximum), int(positions.numel()))
    anomaly_take = min((take + 1) // 2, int(positions.numel()))
    local_score = score.flatten()[positions].float()
    anomaly = positions[local_score.topk(anomaly_take).indices]
    remaining = positions[~torch.isin(positions, anomaly)]
    representative_take = take - anomaly_take
    if representative_take > 0 and remaining.numel() > 0:
        grid = torch.linspace(
            0, remaining.numel() - 1, representative_take, device=mask.device
        ).round().long()
        chosen = torch.cat((anomaly, remaining[grid]))
    else:
        chosen = anomaly
    chosen = torch.unique(chosen, sorted=True)[:take]
    valid = torch.zeros(maximum, dtype=torch.bool, device=mask.device)
    output = torch.zeros(maximum, dtype=torch.long, device=mask.device)
    output[: chosen.numel()] = chosen
    valid[: chosen.numel()] = True
    if chosen.numel() < maximum:
        output[chosen.numel():] = chosen[-1]
    return output, valid


def sample_region_evidence(
    features: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    maximum: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    score = features["enhanced_change"].float().norm(dim=0)
    positions, valid = select_region_positions(score, mask, maximum)
    output = {}
    for name, feature in features.items():
        values = feature.permute(1, 2, 0).reshape(-1, feature.shape[0])
        output[name] = values[positions]
    return output, valid


class EvidencePooler(nn.Module):
    def __init__(
        self,
        source_dims: Mapping[str, int],
        source_variant: str,
        pool_variant: str,
        hidden_dim: int = 192,
        key_dim: int = 128,
    ) -> None:
        super().__init__()
        if source_variant not in SOURCE_VARIANTS:
            raise ValueError(source_variant)
        if pool_variant not in POOL_VARIANTS:
            raise ValueError(pool_variant)
        self.source_variant = source_variant
        self.pool_variant = pool_variant
        self.names = SOURCE_VARIANTS[source_variant]
        per_source = max(32, hidden_dim // len(self.names))
        self.source_projection = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(int(source_dims[name])),
                nn.Linear(int(source_dims[name]), per_source),
                nn.GELU(),
            ) for name in self.names
        })
        token_dim = per_source * len(self.names)
        if pool_variant == "p0_mean":
            main_dim = token_dim
        elif pool_variant == "p1_stats":
            main_dim = token_dim * 3
        elif pool_variant == "p2_attn_skip":
            main_dim = token_dim * 2
        else:
            main_dim = token_dim * 3
        self.attention = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, max(16, token_dim // 2)),
            nn.GELU(), nn.Linear(max(16, token_dim // 2), 1),
        )
        self.main = nn.Sequential(
            nn.LayerNorm(main_dim), nn.Linear(main_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden_dim, key_dim),
        )
        self.skip = nn.Sequential(
            nn.LayerNorm(token_dim * 2), nn.Linear(token_dim * 2, key_dim)
        )
        self.skip_scale = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(key_dim, 3)

    def forward(
        self,
        tokens: Mapping[str, torch.Tensor],
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = torch.cat(
            [self.source_projection[name](tokens[name].float()) for name in self.names],
            dim=-1,
        )
        mean = masked_mean(hidden, token_mask)
        std = masked_std(hidden, token_mask)
        maximum = masked_max(hidden, token_mask)
        score = self.attention(hidden).squeeze(-1).masked_fill(~token_mask, -torch.inf)
        weight = torch.softmax(score.float(), dim=1).to(hidden.dtype)
        attended = (hidden * weight.unsqueeze(-1)).sum(1)
        top_count = max(1, int(math.ceil(hidden.shape[1] * 0.25)))
        top_index = score.topk(top_count, dim=1).indices
        top_mask = torch.zeros_like(token_mask).scatter(1, top_index, True) & token_mask
        tail = masked_mean(hidden, top_mask)
        if self.pool_variant == "p0_mean":
            main_input = mean
        elif self.pool_variant == "p1_stats":
            main_input = torch.cat((mean, std, maximum), dim=1)
        elif self.pool_variant == "p2_attn_skip":
            main_input = torch.cat((attended, maximum), dim=1)
        else:
            main_input = torch.cat((attended, maximum, tail), dim=1)
        main = self.main(main_input)
        if self.pool_variant in ("p2_attn_skip", "p3_attn_tail"):
            key_raw = main + torch.tanh(self.skip_scale) * self.skip(torch.cat((mean, std), dim=1))
        else:
            key_raw = main
        key = F.normalize(key_raw.float(), dim=1)
        return key, self.classifier(key), {
            "attention_entropy": (-(weight * weight.clamp_min(1e-8).log()).sum(1)),
            "skip_scale": torch.tanh(self.skip_scale).detach().expand(key.shape[0]),
        }


def tokens_at(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device):
    base = payload["tokens"]
    gamma = float(payload["metadata"]["diff_gamma_effective"])
    output = {
        "rgb_structure": base["rgb_structure"][indices].to(device).float(),
        "base_change": base["base_change"][indices].to(device).float(),
        "explicit_diff": base["explicit_diff"][indices].to(device).float(),
    }
    output["enhanced_change"] = output["base_change"] + gamma * output["explicit_diff"]
    return output, payload["token_mask"][indices].to(device)


class BalancedEventClassSampler:
    def __init__(self, labels: torch.Tensor, events: Sequence[str], seed: int) -> None:
        self.rng = random.Random(int(seed))
        buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, (label, event) in enumerate(zip(labels.tolist(), events)):
            buckets[(int(label), str(event))].append(index)
        self.by_class = {
            label: {event: rows for (cls, event), rows in buckets.items() if cls == label}
            for label in (1, 2, 3)
        }

    def draw(self, batch_size: int) -> torch.Tensor:
        output = []
        for offset in range(int(batch_size)):
            label = 1 + offset % 3
            if not self.by_class[label]:
                raise RuntimeError(f"class {label} has no records")
            event = self.rng.choice(sorted(self.by_class[label]))
            output.append(self.rng.choice(self.by_class[label][event]))
        self.rng.shuffle(output)
        return torch.tensor(output, dtype=torch.long)


class RetrievalAugmentedClassifier(nn.Module):
    def __init__(
        self,
        key_dim: int,
        variant: str,
        geometry_dim: int,
        event_count: int,
        hidden_dim: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if variant not in RAG_VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.label_embedding = nn.Embedding(4, key_dim)
        self.geometry = nn.Sequential(nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, key_dim))
        self.event_embedding = nn.Embedding(max(1, event_count + 1), key_dim)
        self.attention = nn.MultiheadAttention(key_dim, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(key_dim * 3), nn.Linear(key_dim * 3, hidden_dim), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        query: torch.Tensor,
        neighbours: torch.Tensor,
        neighbour_labels: torch.Tensor,
        neighbour_geometry: torch.Tensor,
        neighbour_events: torch.Tensor,
    ) -> torch.Tensor:
        value = neighbours.float()
        if self.variant in ("x1_label_geometry", "x2_label_geometry_event"):
            value = value + self.label_embedding(neighbour_labels.long())
            value = value + self.geometry(neighbour_geometry.float())
        if self.variant == "x2_label_geometry_event":
            value = value + self.event_embedding(neighbour_events.long())
        attended, _ = self.attention(query[:, None].float(), value, value, need_weights=False)
        attended = attended[:, 0]
        return self.fuse(torch.cat((query.float(), attended, query.float() - attended), dim=1))


__all__ = [
    "BalancedEventClassSampler", "EvidencePooler", "POOL_VARIANTS", "RAG_VARIANTS",
    "RetrievalAugmentedClassifier", "SOURCE_VARIANTS", "balanced_event_folds",
    "cross_event_supervised_contrastive", "retrieval_diverse_topk",
    "retrieval_relative_gap", "sample_region_evidence", "threshold_candidates",
    "tokens_at", "valid_miou",
]
