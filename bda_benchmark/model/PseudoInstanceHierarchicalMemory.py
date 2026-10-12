"""Pseudo-instance hierarchical semantic memory.

The module is deliberately independent from DamageFormer.  A caller supplies
child-region features, pseudo-building ids and provenance metadata.  Memory is
read-only during validation/test and can exclude the query image or event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalOutput:
    keys: torch.Tensor
    labels: torch.Tensor
    similarity: torch.Tensor
    valid: torch.Tensor
    metadata: torch.Tensor


def parent_mean_context(
    child: torch.Tensor, parent_id: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Broadcast the mean feature of each pseudo-building to its children."""
    if child.ndim != 2 or parent_id.ndim != 1 or len(child) != len(parent_id):
        raise ValueError("expected child [N,C] and parent_id [N]")
    if valid is None:
        valid = torch.ones(len(child), dtype=torch.bool, device=child.device)
    valid = valid.bool()
    output = torch.zeros_like(child)
    for identity in torch.unique(parent_id[valid]):
        members = valid & (parent_id == identity)
        output[members] = child[members].mean(0)
    return output


def semantic_labels_from_delta(
    direction_code: torch.Tensor, delta_hist: torch.Tensor
) -> torch.Tensor:
    """Recover the dominant GT class of a constant-transition cluster.

    A transition b->a adds its GT counts to delta_hist[:, a] and subtracts
    them from delta_hist[:, b], so the positive column recovers the histogram.
    Background-dominant clusters are returned as -1 and excluded from memory.
    Output labels are Intact/Damaged/Destroyed = 0/1/2.
    """
    direction_code = direction_code.long()
    after = direction_code.remainder(4)
    rows = torch.arange(len(after), device=delta_hist.device)
    counts = delta_hist.long()[rows, :, after].clamp_min(0)
    label4 = counts.argmax(1)
    label = label4 - 1
    label[counts.sum(1) == 0] = -1
    label[label4 == 0] = -1
    return label


def stable_string_hash(values: list[str]) -> torch.Tensor:
    """Stable signed-63-bit FNV-1a ids (unlike Python's salted hash)."""
    result = []
    for value in values:
        number = 1469598103934665603
        for byte in value.encode("utf-8"):
            number ^= byte
            number = (number * 1099511628211) & ((1 << 63) - 1)
        result.append(number)
    return torch.tensor(result, dtype=torch.long)


class ProvenanceBalancedMemory(nn.Module):
    """Equal-capacity semantic memory with image/event provenance filters."""

    def __init__(self, classes: int, channels: int, metadata_dim: int,
                 capacity_per_class: int = 2048) -> None:
        super().__init__()
        self.classes = int(classes)
        self.channels = int(channels)
        self.metadata_dim = int(metadata_dim)
        self.capacity_per_class = int(capacity_per_class)
        shape = (classes, capacity_per_class)
        self.register_buffer("keys", torch.zeros(*shape, channels))
        self.register_buffer("metadata", torch.zeros(*shape, metadata_dim))
        self.register_buffer("event_id", torch.full(shape, -1, dtype=torch.long))
        self.register_buffer("image_id", torch.full(shape, -1, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(shape, dtype=torch.bool))
        self.register_buffer("cursor", torch.zeros(classes, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, labels: torch.Tensor,
                event_id: torch.Tensor, image_id: torch.Tensor,
                metadata: torch.Tensor) -> None:
        keys = F.normalize(keys.detach().float(), dim=-1)
        for index in range(len(keys)):
            label = int(labels[index])
            if not 0 <= label < self.classes:
                continue
            slot = int(self.cursor[label]) % self.capacity_per_class
            self.keys[label, slot].copy_(keys[index])
            self.metadata[label, slot].copy_(metadata[index].float())
            self.event_id[label, slot] = event_id[index]
            self.image_id[label, slot] = image_id[index]
            self.valid[label, slot] = True
            self.cursor[label] += 1

    def retrieve(self, query: torch.Tensor, query_event: torch.Tensor,
                 query_image: torch.Tensor, topk_per_class: int = 4,
                 exclude_same_event: bool = True,
                 exclude_same_image: bool = True) -> RetrievalOutput:
        query = F.normalize(query.float(), dim=-1)
        all_keys, all_labels, all_scores, all_valid, all_metadata = [], [], [], [], []
        for class_id in range(self.classes):
            allowed = self.valid[class_id].unsqueeze(0).expand(len(query), -1).clone()
            if exclude_same_event:
                allowed &= self.event_id[class_id].unsqueeze(0) != query_event[:, None]
            if exclude_same_image:
                allowed &= self.image_id[class_id].unsqueeze(0) != query_image[:, None]
            score = query @ self.keys[class_id].T
            score = score.masked_fill(~allowed, -1.0e4)
            count = min(int(topk_per_class), self.capacity_per_class)
            top_score, top_index = score.topk(count, dim=1)
            all_keys.append(self.keys[class_id][top_index])
            all_metadata.append(self.metadata[class_id][top_index])
            all_scores.append(top_score)
            all_valid.append(top_score > -9.0e3)
            all_labels.append(torch.full_like(top_index, class_id))
        return RetrievalOutput(
            keys=torch.cat(all_keys, 1), labels=torch.cat(all_labels, 1),
            similarity=torch.cat(all_scores, 1), valid=torch.cat(all_valid, 1),
            metadata=torch.cat(all_metadata, 1),
        )

    @property
    def size(self) -> int:
        return int(self.valid.sum())


class PseudoInstanceHierarchicalMemory(nn.Module):
    """Child-parent query encoder plus occurrence/severity semantic memories."""

    def __init__(self, input_dim: int, hidden: int = 128,
                 capacity_per_class: int = 2048, topk: int = 4,
                 direction_classes: int = 16) -> None:
        super().__init__()
        self.topk = int(topk)
        self.direction_embedding = nn.Embedding(direction_classes, 16)
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim * 3 + 16, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.metadata_encoder = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.label_embedding = nn.Embedding(3, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.occurrence_head = nn.Linear(hidden, 2)
        self.severity_head = nn.Linear(hidden, 2)
        self.retrieval_gate = nn.Parameter(torch.zeros(()))
        self.semantic_memory = ProvenanceBalancedMemory(
            3, hidden, 2, capacity_per_class
        )

    def encode(self, child: torch.Tensor, parent_id: torch.Tensor,
               direction_code: torch.Tensor) -> torch.Tensor:
        parent = parent_mean_context(child, parent_id)
        direction = self.direction_embedding(direction_code.long().clamp(0, 15))
        query = torch.cat((child, parent, child - parent, direction), 1)
        return F.normalize(self.query_encoder(query), dim=1)

    def forward(self, child: torch.Tensor, parent_id: torch.Tensor,
                direction_code: torch.Tensor, cluster_area: torch.Tensor,
                parent_area: torch.Tensor, event_id: torch.Tensor,
                image_id: torch.Tensor, use_memory: bool = True) -> Dict[str, torch.Tensor]:
        node = self.encode(child, parent_id, direction_code)
        valid_fraction = node.new_zeros(())
        mean_similarity = node.new_zeros(())
        if use_memory and self.semantic_memory.size:
            found = self.semantic_memory.retrieve(
                node, event_id, image_id, self.topk, True, True
            )
            support = found.keys + self.label_embedding(found.labels)
            support = support + self.metadata_encoder(found.metadata)
            valid = found.valid
            none = ~valid.any(1)
            if none.any():
                valid = valid.clone(); support = support.clone()
                valid[none, 0] = True; support[none, 0] = 0
            retrieved, _ = self.attention(
                node[:, None], support, support,
                key_padding_mask=~valid, need_weights=False,
            )
            node = node + torch.tanh(self.retrieval_gate) * retrieved[:, 0]
            valid_fraction = found.valid.float().mean()
            if found.valid.any():
                mean_similarity = found.similarity[found.valid].mean()
        return {
            "embedding": node,
            "occurrence_logits": self.occurrence_head(node),
            "severity_logits": self.severity_head(node),
            "retrieval_gate": torch.tanh(self.retrieval_gate),
            "memory_size": node.new_tensor(float(self.semantic_memory.size)),
            "retrieval_valid_fraction": valid_fraction,
            "retrieval_mean_similarity": mean_similarity,
        }

    @staticmethod
    def metadata(cluster_area: torch.Tensor, parent_area: torch.Tensor) -> torch.Tensor:
        return torch.stack((cluster_area.float().log1p(),
                            cluster_area.float() / parent_area.float().clamp_min(1)), 1)

