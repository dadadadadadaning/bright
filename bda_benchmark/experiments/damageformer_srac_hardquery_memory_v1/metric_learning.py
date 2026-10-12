"""Frozen-feature, soft-target search learning for heterogeneous Train queries.

Dominant classes are used ONLY to stratify query sampling. The hard-query
objective preserves the complete target distribution, with strict event AND
image exclusion before target normalization and the softmax denominator.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..damageformer_srac_qmetric_v1.metric import RetrievalProjector, encode_keys

NUM_CLASSES = 4


def validate_distribution(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != NUM_CLASSES:
        raise ValueError(f"{name} must be [N, 4]")
    value = value.detach().float()
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    total = value.sum(-1)
    if not torch.allclose(total, torch.ones_like(total), atol=1e-5, rtol=0):
        raise ValueError(f"{name} rows must sum to one")
    return value


@torch.no_grad()
def query_hardness(distribution: torch.Tensor, base_probability: torch.Tensor) -> torch.Tensor:
    target = validate_distribution(distribution, "distribution")
    probability = validate_distribution(base_probability, "base_probability")
    if target.shape != probability.shape:
        raise ValueError("Target and J1 probability dimensions disagree")
    return (1.0 - (target * probability.to(target.device)).sum(-1)).clamp(0, 1)


class HardQuerySampler:
    """Class-balanced draws oversampling the within-class highest-hardness quartile.

    Pools are computed once. Every class with two or more eligible queries has
    both ordinary and difficult queries available, including tied hardness.
    Equal hardness ties are resolved by input order; no GT is used at inference.
    """

    def __init__(self, distribution, base_probability, eligible=None, hard_quantile=.75):
        if not 0 < hard_quantile < 1:
            raise ValueError("Require 0 < hard_quantile < 1")
        self.hard_quantile = float(hard_quantile)
        self.distribution = validate_distribution(distribution, "distribution").cpu()
        self.hardness = query_hardness(self.distribution, base_probability.cpu())
        self.classes = self.distribution.argmax(-1)
        if eligible is None:
            eligible = torch.ones(len(self.classes), dtype=torch.bool)
        if eligible.shape != self.classes.shape:
            raise ValueError("eligible must contain one flag per query")
        self.eligible = eligible.detach().cpu().bool()
        self.pools = []
        self.inventory = {}
        for class_id in range(NUM_CLASSES):
            ids = torch.where(self.eligible & (self.classes == class_id))[0]
            ordered = ids[torch.argsort(self.hardness[ids], stable=True)]
            split = (max(1, min(len(ordered) - 1, int(math.ceil(len(ordered) * hard_quantile))))
                     if len(ordered) > 1 else len(ordered))
            ordinary, difficult = ordered[:split], ordered[split:]
            self.pools.append((ordinary, difficult))
            self.inventory[str(class_id)] = dict(
                count=len(ids), ordinary_count=len(ordinary), difficult_count=len(difficult),
                hardness_mean=float(self.hardness[ids].mean()) if len(ids) else None,
                hardness_min=float(self.hardness[ids].min()) if len(ids) else None,
                hardness_max=float(self.hardness[ids].max()) if len(ids) else None,
                mixed_count=int((self.distribution[ids].max(-1).values < .8).sum()),
                hard_quantile=self.hard_quantile,
                difficult_min_hardness=float(self.hardness[difficult].min()) if len(difficult) else None,
            )

    def sample(self, per_class: int, generator: torch.Generator, hard_fraction=.5):
        if per_class < 1 or not 0 < hard_fraction < 1:
            raise ValueError("Require positive per_class and 0 < hard_fraction < 1")
        if per_class < 2:
            raise ValueError("per_class >= 2 is needed to retain ordinary and difficult queries")
        difficult_count = max(1, min(per_class - 1, int(round(per_class * hard_fraction))))
        ordinary_count = per_class - difficult_count
        selected, rows = [], {}

        def draw(pool, count):
            if not len(pool) or count == 0:
                return pool[:0]
            if len(pool) >= count:
                return pool[torch.randperm(len(pool), generator=generator)[:count]]
            return pool[torch.randint(len(pool), (count,), generator=generator)]

        for class_id, (ordinary, difficult) in enumerate(self.pools):
            # Singleton classes are not silently dropped; the missing stratum is logged.
            ordinary_take = ordinary_count if len(difficult) else per_class
            difficult_take = difficult_count if len(ordinary) else per_class
            easy_ids = draw(ordinary, ordinary_take)
            hard_ids = draw(difficult, difficult_take)
            ids = torch.cat((easy_ids, hard_ids))
            selected.append(ids)
            rows[str(class_id)] = dict(
                ordinary=len(easy_ids), difficult=len(hard_ids), sampled=len(ids),
                mean_hardness=float(self.hardness[ids].mean()) if len(ids) else None,
                ordinary_replacement=ordinary_take > len(ordinary) > 0,
                difficult_replacement=difficult_take > len(difficult) > 0,
            )
        indices = torch.cat(selected)
        if len(indices):
            indices = indices[torch.randperm(len(indices), generator=generator)]
        return indices, dict(by_class=rows, sampled_anchor_count=len(indices),
                             hard_fraction=hard_fraction, hard_quantile=self.hard_quantile)


def soft_cross_event_loss(
    query_tokens: torch.Tensor,
    query_distribution: torch.Tensor,
    query_events: torch.Tensor,
    query_images: torch.Tensor,
    support_tokens: torch.Tensor,
    support_distribution: torch.Tensor,
    support_events: torch.Tensor,
    support_images: torch.Tensor,
    projector: nn.Module,
    temperature: float = .1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """CE against normalized soft-label overlap over permitted reliable supports.

    Raw image/event IDs share the bank's global vocabulary. All feature and GT
    inputs are detached; gradients update only the shared retrieval projector.
    Queries without positive overlap, or with fewer than two permitted supports,
    contribute a graph-connected zero and are explicitly counted.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if query_tokens.ndim != 2 or support_tokens.ndim != 2:
        raise ValueError("tokens must have shape [N, D]")
    if query_tokens.shape[1] != support_tokens.shape[1]:
        raise ValueError("Query/support feature dimensions disagree")
    device = query_tokens.device
    if support_tokens.device != device:
        raise ValueError("Query/support tokens must share a device")
    target = validate_distribution(query_distribution, "query_distribution").to(device)
    supports = validate_distribution(support_distribution, "support_distribution").to(device)
    if len(target) != len(query_tokens) or len(supports) != len(support_tokens):
        raise ValueError("Token and target counts disagree")
    provenance = []
    for ids, count, name in (
        (query_events, len(target), "query_events"),
        (query_images, len(target), "query_images"),
        (support_events, len(supports), "support_events"),
        (support_images, len(supports), "support_images"),
    ):
        if ids.shape != (count,) or ids.is_floating_point():
            raise ValueError(f"{name} must be an integer vector of length {count}")
        provenance.append(ids.detach().to(device).long())
    qe, qi, se, si = provenance
    allowed = (qe[:, None] != se[None, :]) & (qi[:, None] != si[None, :])
    overlap = (target @ supports.T).clamp_min(0).masked_fill(~allowed, 0)
    total = overlap.sum(-1)
    has_positive = total > 0
    enough_candidates = allowed.sum(-1) >= 2
    used = has_positive & enough_candidates
    query_key = encode_keys(query_tokens, projector, detach_input=True)
    support_key = encode_keys(support_tokens, projector, detach_input=True)
    if used.any():
        logits = query_key[used] @ support_key.T / temperature
        candidate_mask = allowed[used]
        log_denominator = torch.logsumexp(logits.masked_fill(~candidate_mask, -torch.inf), dim=-1)
        # Keep excluded entries finite, avoiding 0 * -inf in soft target CE.
        log_probability = (logits - log_denominator[:, None]).masked_fill(~candidate_mask, 0)
        normalized_target = overlap[used] / total[used, None]
        loss = -(normalized_target * log_probability).sum(-1).mean()
        entropy = -(normalized_target * normalized_target.clamp_min(1e-30).log()).sum(-1)
        target_entropy = float(entropy.detach().mean())
    else:
        loss = (query_key.sum() + support_key.sum()) * 0.0
        target_entropy = None
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite soft cross-event loss")
    dominant = target.argmax(-1)
    stats = dict(
        selected_anchor_count=len(target), used_anchor_count=int(used.sum()),
        skipped_selected_no_positive=int((~has_positive).sum()),
        skipped_selected_insufficient_candidates=int((~enough_candidates).sum()),
        allowed_pair_count=int(allowed.sum()), positive_overlap_pairs=int((overlap > 0).sum()),
        zero_overlap_pairs=int((allowed & (overlap == 0)).sum()),
        mean_positive_overlap_candidates=float((overlap[used] > 0).sum(-1).float().mean()) if used.any() else None,
        target_entropy=target_entropy,
        used_by_class=torch.bincount(dominant[used], minlength=NUM_CLASSES).tolist(),
        mixed_used=int((target[used].max(-1).values < .8).sum()),
        zero_loss=not bool(used.any()),
    )
    return loss, stats
