# -*- coding: utf-8 -*-
"""Train-only cross-event metric adapter for frozen P3 region features."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CrossEventResidualMetric(nn.Module):
    """Identity-safe residual metric adapter.

    The last projection is exactly zero initialised, so step zero reproduces
    the original normalized P3 region key.  Unlike a zero transition scalar,
    the last projection still receives a gradient on the first update.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
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
        base = F.normalize(feature.float(), dim=1)
        return F.normalize(base + self.adapter(base), dim=1)


def cross_event_supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    event_codes: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Supervised contrastive loss with cross-event positives only.

    Same-class/same-event pairs are excluded rather than treated as negatives.
    Different-class pairs are negatives regardless of event.  Every retained
    anchor must have at least one same-class/different-event positive.
    """

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    z = F.normalize(embedding.float(), dim=1)
    labels = labels.long().flatten().to(z.device)
    event_codes = event_codes.long().flatten().to(z.device)
    if z.shape[0] != labels.numel() or labels.numel() != event_codes.numel():
        raise ValueError("embedding, labels and event codes must align")
    count = z.shape[0]
    if count < 2:
        raise ValueError("contrastive batch must contain at least two samples")

    identity = torch.eye(count, dtype=torch.bool, device=z.device)
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
    mean_positive_log_probability = (
        (log_probability * positive.float()).sum(dim=1) / positive_count
    )
    return -mean_positive_log_probability[valid_anchor].mean()


def embedding_preservation_loss(
    adapted: torch.Tensor, original: torch.Tensor
) -> torch.Tensor:
    """Squared L2 trust-region penalty between normalized embeddings."""

    if adapted.shape != original.shape or adapted.ndim != 2:
        raise ValueError("adapted and original embeddings must be aligned matrices")
    adapted = F.normalize(adapted.float(), dim=1)
    original = F.normalize(original.float(), dim=1)
    return (adapted - original).square().sum(dim=1).mean()
