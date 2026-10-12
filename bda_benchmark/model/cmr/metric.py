# -*- coding: utf-8 -*-
"""Isolated CMR-M1 cross-event metric learning components.

This module never imports or mutates the accepted P3 model.  It adapts cached
building keys only, and its residual branch is exactly zero initialised so the
step-zero embedding is the frozen P0 key.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CMRResidualMetric(nn.Module):
    """Identity-safe residual adapter for a frozen CMR pooling key."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("metric dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2 or feature.shape[1] != self.input_dim:
            raise ValueError(
                f"expected [N,{self.input_dim}] features, got {tuple(feature.shape)}"
            )
        base = F.normalize(feature.float(), dim=1, eps=1e-8)
        return F.normalize(base + self.adapter(base), dim=1, eps=1e-8)


def event_aware_supcon_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    event_codes: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Supervised contrastive loss using cross-event positives only.

    Same-class/same-event pairs are ignored.  Same-class/different-event pairs
    are positives and all different-class pairs are negatives.
    """

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    z = F.normalize(embedding.float(), dim=1, eps=1e-8)
    labels = labels.long().flatten().to(z.device)
    event_codes = event_codes.long().flatten().to(z.device)
    if z.shape[0] != labels.numel() or labels.numel() != event_codes.numel():
        raise ValueError("embedding, labels and event codes must align")
    if z.shape[0] < 2:
        raise ValueError("contrastive batch must contain at least two samples")

    identity = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    same_class = labels[:, None] == labels[None, :]
    different_event = event_codes[:, None] != event_codes[None, :]
    positive = same_class & different_event & (~identity)
    negative = (~same_class) & (~identity)
    valid_pair = positive | negative
    valid_anchor = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid_anchor.any()):
        raise RuntimeError("batch has no valid cross-event contrastive anchors")

    logits = (z @ z.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * valid_pair.float()
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    positive_count = positive.sum(dim=1).clamp_min(1)
    positive_log_probability = (
        (log_probability * positive.float()).sum(dim=1) / positive_count
    )
    return -positive_log_probability[valid_anchor].mean()


def selective_damaged_positive_mask(
    raw_feature: torch.Tensor,
    labels: torch.Tensor,
    event_codes: torch.Tensor,
    *,
    damaged_class: int = 1,
    damaged_topk: int = 2,
) -> torch.Tensor:
    """Build the CMR-M1B positive mask without using learned embeddings.

    Intact and Destroyed retain all same-class/cross-event positives.  A
    Damaged anchor retains only its ``damaged_topk`` most similar Damaged
    samples from other events in the frozen raw-Mean space.  Other Damaged
    pairs are ignored by the contrastive loss, not treated as negatives.
    """

    if int(damaged_topk) <= 0:
        raise ValueError("damaged_topk must be positive")
    raw = F.normalize(raw_feature.detach().float(), dim=1, eps=1e-8)
    labels = labels.long().flatten().to(raw.device)
    event_codes = event_codes.long().flatten().to(raw.device)
    if raw.ndim != 2:
        raise ValueError("raw_feature must have shape [N,C]")
    if raw.shape[0] != labels.numel() or labels.numel() != event_codes.numel():
        raise ValueError("raw features, labels and event codes must align")

    count = raw.shape[0]
    identity = torch.eye(count, dtype=torch.bool, device=raw.device)
    same_class = labels[:, None] == labels[None, :]
    different_event = event_codes[:, None] != event_codes[None, :]
    cross_event_same_class = same_class & different_event & (~identity)
    damaged_anchor = labels == int(damaged_class)

    # Non-Damaged classes keep the original all-positive relation.
    positive = cross_event_same_class & (~damaged_anchor[:, None])
    damaged_candidates = cross_event_same_class & damaged_anchor[:, None]
    similarity = raw @ raw.t()
    similarity = similarity.masked_fill(~damaged_candidates, float("-inf"))
    take = min(int(damaged_topk), count)
    values, indices = torch.topk(similarity, k=take, dim=1)
    chosen = torch.zeros_like(damaged_candidates)
    chosen.scatter_(1, indices, torch.isfinite(values))
    positive |= chosen & damaged_candidates
    return positive


def selective_damaged_supcon_loss(
    embedding: torch.Tensor,
    raw_feature: torch.Tensor,
    labels: torch.Tensor,
    event_codes: torch.Tensor,
    *,
    temperature: float = 0.10,
    damaged_class: int = 1,
    damaged_topk: int = 2,
) -> torch.Tensor:
    """Cross-event SupCon with topology-preserving Damaged positives.

    The learned embedding receives gradients, while the positive-neighbour
    selection is locked to the frozen raw Mean geometry.  This prevents the
    loss from declaring visually incompatible Damaged modes as positives.
    """

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    z = F.normalize(embedding.float(), dim=1, eps=1e-8)
    labels = labels.long().flatten().to(z.device)
    event_codes = event_codes.long().flatten().to(z.device)
    raw_feature = raw_feature.to(z.device)
    if z.shape[0] != labels.numel() or labels.numel() != event_codes.numel():
        raise ValueError("embedding, labels and event codes must align")
    if raw_feature.shape != z.shape:
        raise ValueError("raw_feature and embedding must have identical shape")
    if z.shape[0] < 2:
        raise ValueError("contrastive batch must contain at least two samples")

    positive = selective_damaged_positive_mask(
        raw_feature,
        labels,
        event_codes,
        damaged_class=damaged_class,
        damaged_topk=damaged_topk,
    )
    identity = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    same_class = labels[:, None] == labels[None, :]
    negative = (~same_class) & (~identity)
    valid_pair = positive | negative
    valid_anchor = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid_anchor.any()):
        raise RuntimeError("batch has no valid selective contrastive anchors")

    logits = (z @ z.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * valid_pair.float()
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    positive_count = positive.sum(dim=1).clamp_min(1)
    positive_log_probability = (
        (log_probability * positive.float()).sum(dim=1) / positive_count
    )
    return -positive_log_probability[valid_anchor].mean()
