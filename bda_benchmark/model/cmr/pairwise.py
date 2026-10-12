# -*- coding: utf-8 -*-
"""Frozen-key pairwise reranker used by the isolated CMR-M3 experiment."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CMRBoundedPairwiseScorer(nn.Module):
    """Score a frozen Query--Candidate pair without changing either key.

    The learned residual is bounded and exactly zero at initialisation, so
    epoch 0 reproduces Raw Mean ranking bit-for-bit (up to floating precision).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        maximum_residual: float = 0.05,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("pairwise dimensions must be positive")
        if float(maximum_residual) <= 0:
            raise ValueError("maximum_residual must be positive")
        self.feature_dim = int(feature_dim)
        self.maximum_residual = float(maximum_residual)
        # Symmetric pair evidence avoids learning a Query/Support ordering
        # shortcut.  The raw cosine is included as a calibration feature.
        input_dim = 2 * self.feature_dim + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if query.shape != candidate.shape or query.ndim not in (2, 3):
            raise ValueError("query/candidate must have identical [N,C] or [B,K,C] shape")
        if query.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}")
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        candidate = F.normalize(candidate.float(), dim=-1, eps=1e-8)
        cosine = (query * candidate).sum(dim=-1, keepdim=True)
        relation = torch.cat((torch.abs(query - candidate), query * candidate, cosine), dim=-1)
        raw = self.network(relation).squeeze(-1)
        return self.maximum_residual * torch.tanh(raw)


class CMRCandidateConditionalPairwiseScorer(nn.Module):
    """Candidate-label conditional variant with a shared relation encoder.

    Memory labels are train-only support metadata and are known at retrieval
    time.  Query labels are never accepted by this module.  Splitting only the
    final scoring heads keeps the M3C change controlled: candidate pool,
    relation features, loss and residual bound are identical to M3.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        maximum_residual: float = 0.05,
        num_candidate_classes: int = 3,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("pairwise dimensions must be positive")
        if float(maximum_residual) <= 0 or int(num_candidate_classes) <= 1:
            raise ValueError("invalid conditional scorer configuration")
        self.feature_dim = int(feature_dim)
        self.maximum_residual = float(maximum_residual)
        self.num_candidate_classes = int(num_candidate_classes)
        input_dim = 2 * self.feature_dim + 1
        self.relation_encoder = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.class_heads = nn.ModuleList(
            nn.Linear(int(hidden_dim), 1)
            for _ in range(self.num_candidate_classes)
        )
        for head in self.class_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        query: torch.Tensor,
        candidate: torch.Tensor,
        candidate_labels: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape != candidate.shape or query.ndim not in (2, 3):
            raise ValueError("query/candidate must have identical [N,C] or [B,K,C] shape")
        if query.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}")
        if candidate_labels.shape != query.shape[:-1]:
            raise ValueError("candidate_labels must match the pair leading dimensions")
        candidate_labels = candidate_labels.long()
        if candidate_labels.numel() and (
            int(candidate_labels.min()) < 0
            or int(candidate_labels.max()) >= self.num_candidate_classes
        ):
            raise ValueError("candidate_labels contains an unsupported class")
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        candidate = F.normalize(candidate.float(), dim=-1, eps=1e-8)
        cosine = (query * candidate).sum(dim=-1, keepdim=True)
        relation = torch.cat((torch.abs(query - candidate), query * candidate, cosine), dim=-1)
        hidden = self.relation_encoder(relation)
        all_scores = torch.cat([head(hidden) for head in self.class_heads], dim=-1)
        raw = torch.gather(all_scores, -1, candidate_labels.unsqueeze(-1)).squeeze(-1)
        return self.maximum_residual * torch.tanh(raw)


def multi_positive_candidate_loss(
    score: torch.Tensor,
    candidate_labels: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Listwise loss: promote every correct-label item inside frozen Top-K."""

    if score.ndim != 2 or candidate_labels.shape != score.shape:
        raise ValueError("score and candidate_labels must be aligned [B,K]")
    query_labels = query_labels.long().flatten()
    if query_labels.numel() != score.shape[0]:
        raise ValueError("query_labels must align with score rows")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    positive = candidate_labels.long() == query_labels[:, None]
    valid = positive.any(dim=1) & (~positive).any(dim=1)
    if not bool(valid.any()):
        raise ValueError("pairwise batch has no query with both positive and negative candidates")
    logits = score[valid] / float(temperature)
    positive = positive[valid]
    numerator = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def class_balanced_hard_top1_loss(
    score: torch.Tensor,
    candidate_labels: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    temperature: float = 0.10,
    margin: float = 0.0,
) -> torch.Tensor:
    """Directly rank the best correct candidate above the hardest incorrect.

    Losses are averaged inside each Query class first and then across classes,
    so post-eligibility batch imbalance cannot let an easy class dominate.
    """

    if score.ndim != 2 or candidate_labels.shape != score.shape:
        raise ValueError("score and candidate_labels must be aligned [B,K]")
    query_labels = query_labels.long().flatten()
    if query_labels.numel() != score.shape[0]:
        raise ValueError("query_labels must align with score rows")
    if float(temperature) <= 0 or float(margin) < 0:
        raise ValueError("temperature must be positive and margin nonnegative")
    positive = candidate_labels.long() == query_labels[:, None]
    valid = positive.any(dim=1) & (~positive).any(dim=1)
    if not bool(valid.any()):
        raise ValueError("top1 batch has no query with both positive and negative candidates")
    valid_score = score[valid]
    valid_positive = positive[valid]
    valid_labels = query_labels[valid]
    best_positive = valid_score.masked_fill(~valid_positive, float("-inf")).max(dim=1).values
    hardest_negative = valid_score.masked_fill(valid_positive, float("-inf")).max(dim=1).values
    per_query = F.softplus(
        (hardest_negative - best_positive + float(margin)) / float(temperature)
    )
    class_losses = []
    for cls in torch.unique(valid_labels, sorted=True).tolist():
        mask = valid_labels == int(cls)
        if bool(mask.any()):
            class_losses.append(per_query[mask].mean())
    if not class_losses:
        raise RuntimeError("top1 class-balanced reduction has no class")
    return torch.stack(class_losses).mean()
