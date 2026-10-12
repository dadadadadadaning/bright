"""Train-only retrieval and a direct region+Memory Transformer classifier."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class JointMemoryReader(nn.Module):
    """Pre-LN cross-attention; outputs class logits directly, never A1 residuals."""

    def __init__(self, dim: int = 128, num_events: int = 1, heads: int = 4, slots: int = 8):
        super().__init__()
        if slots < 1:
            raise ValueError("At least one support slot is required")
        self.slots = int(slots)
        self.label_embedding = nn.Embedding(4, dim)
        self.disaster_embedding = nn.Embedding(8, dim)
        self.event_embedding = nn.Embedding(num_events + 1, dim)
        self.geometry_embedding = nn.Linear(5, dim)
        self.learned_slots = nn.Parameter(torch.zeros(self.slots, dim))
        self.learned_label_logits = nn.Parameter(torch.zeros(self.slots, 4))
        nn.init.normal_(self.learned_slots, std=0.02)
        with torch.no_grad():
            self.learned_label_logits[torch.arange(self.slots), torch.arange(self.slots) % 4] = 1.0
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.classifier_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 4)

    def _fallback(self, query: torch.Tensor):
        distribution = self.learned_label_logits.softmax(-1)
        slots = self.learned_slots + distribution @ self.label_embedding.weight
        slots = slots[None].expand(len(query), -1, -1)
        valid = torch.ones(slots.shape[:2], dtype=torch.bool, device=query.device)
        return slots, slots, valid

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor | None = None,
        distribution: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        mode: str = "memory",
        metadata: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        fallback_key, fallback_value, fallback_valid = self._fallback(query)
        if mode == "nomemory":
            keys, values, attention_valid = fallback_key, fallback_value, fallback_valid
        elif mode == "memory":
            if support is None or distribution is None or valid is None:
                raise ValueError("Memory mode requires support, distribution and valid mask")
            if support.shape[1] != self.slots or valid.shape[1] != self.slots:
                raise ValueError("Memory support count must match learned control slots")
            memory_keys = support
            memory_values = support + distribution @ self.label_embedding.weight
            if metadata is not None:
                memory_values = memory_values + self.disaster_embedding(metadata["disasters"])
                memory_values = memory_values + self.event_embedding(metadata["events"] + 1)
                memory_values = memory_values + self.geometry_embedding(metadata["geometry"].float())
            # Invalid/missing historical supports are replaced slot-for-slot.
            keys = torch.where(valid[..., None], memory_keys, fallback_key)
            values = torch.where(valid[..., None], memory_values, fallback_value)
            attention_valid = fallback_valid
        else:
            raise ValueError(f"Unknown reader mode: {mode}")

        normalized_query = self.query_norm(query)[:, None]
        context, _ = self.cross_attention(
            normalized_query,
            self.key_norm(keys),
            self.value_norm(values),
            key_padding_mask=~attention_valid,
            need_weights=False,
        )
        hidden = query + context[:, 0]
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return self.classifier(self.classifier_norm(hidden))


def retrieve(
    query: torch.Tensor,
    bank: Dict[str, torch.Tensor],
    event: int,
    image: int,
    topk: int = 8,
    intervention: str = "normal",
    generator: torch.Generator | None = None,
    return_metadata: bool = False,
    policy: str = "global",
    exclusion: str = "cross_event",
):
    """Cosine retrieval with conservative same-event and same-image exclusion."""
    if topk < 1 or len(bank["tokens"]) < 1:
        raise ValueError("Non-empty bank and positive topk required")
    if exclusion == "same_image":
        allowed = bank["images"] != image
    elif exclusion == "cross_event":
        allowed = (bank["events"] != event) & (bank["images"] != image)
    else:
        raise ValueError(exclusion)
    scores = F.normalize(query.float(), dim=-1) @ F.normalize(bank["tokens"].float(), dim=-1).T
    if intervention == "random":
        scores = torch.rand(scores.shape, device=query.device, generator=generator)
    scores = scores.masked_fill(~allowed[None], -torch.inf)
    if policy == "global":
        values, indices = scores.topk(min(topk, len(bank["tokens"])), dim=1)
        if indices.shape[1] < topk:
            padding = topk - indices.shape[1]
            values = torch.cat((values, values.new_full((len(values), padding), -torch.inf)), dim=1)
            indices = torch.cat((indices, indices.new_zeros((len(indices), padding))), dim=1)
        valid = torch.isfinite(values)
    elif policy == "per_class":
        if topk % 4:
            raise ValueError("per_class policy requires topk divisible by four")
        each = topk // 4
        dominant = bank["distribution"].argmax(-1)
        selected_values, selected_indices = [], []
        for class_index in range(4):
            class_scores = scores.masked_fill(dominant[None] != class_index, -torch.inf)
            values, indices = class_scores.topk(min(each, len(bank["tokens"])), dim=1)
            if indices.shape[1] < each:
                pad = each - indices.shape[1]
                values = torch.cat((values, values.new_full((len(values), pad), -torch.inf)), dim=1)
                indices = torch.cat((indices, indices.new_zeros((len(indices), pad))), dim=1)
            selected_values.append(values)
            selected_indices.append(indices)
        values, indices = torch.cat(selected_values, dim=1), torch.cat(selected_indices, dim=1)
        valid = torch.isfinite(values)
    else:
        raise ValueError(policy)
    support = bank["tokens"][indices]
    distribution = bank["distribution"][indices]
    if intervention == "shuffle_labels":
        permutation = torch.randperm(len(bank["tokens"]), device=query.device, generator=generator)
        distribution = bank["distribution"][permutation[indices]]
    if intervention == "off":
        valid = torch.zeros_like(valid)
    if return_metadata:
        metadata = {name: bank[name][indices] for name in ("geometry", "events", "disasters")}
        return support, distribution, valid, metadata
    return support, distribution, valid

