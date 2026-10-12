"""Mine misleading Train retrievals and draw matched metric-learning queries.

The frozen original J3 metric and original full bank only diagnose which Train
queries receive misleading Top-K label evidence. They are never updated here.
The class label of a query is never used to filter candidates at inference.
"""
from __future__ import annotations

import math

import torch

from ..damageformer_srac_hardquery_memory_v1.metric_learning import (
    HardQuerySampler, validate_distribution,
)
from ..damageformer_srac_qmetric_v1.metric import encode_keys


@torch.no_grad()
def mine_reference_risk(
    store,
    bank: dict,
    old_projector,
    device: torch.device,
    *,
    topk: int = 8,
    temperature: float = .1,
    purity: float = .8,
    risk_margin: float = 0.,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return Train-only misleading-Intact and missed-Damaged query masks.

    A misleading Intact query has dominant Train label Intact with purity at
    least ``purity``, is correctly predicted Intact by frozen J1, but its
    frozen original J3 Top-K soft-label vote favors Damaged over Intact. This
    vote is a mining diagnostic, not the J3 Reader output or a performance
    bound. A missed Damaged query has the analogous pure Train label and J1
    predicts a different class. Full-bank ranking, event/image exclusions and
    Top-K match the source J3 retrieval rule.
    """
    if topk < 1 or batch_size < 1 or not 0 <= purity <= 1:
        raise ValueError("Invalid topk, batch_size or purity")
    if (not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(risk_margin) or risk_margin < 0):
        raise ValueError("Temperature must be positive; risk margin must be nonnegative")
    fields = store.fields
    target = validate_distribution(fields["distribution"], "Train distribution").cpu()
    base = validate_distribution(fields["base_probability"], "Train J1 probability").cpu()
    count = len(target)
    if any(fields[name].shape != (count,) for name in ("events", "images")):
        raise ValueError("Invalid Train query provenance")
    bank_target = validate_distribution(bank["distribution"], "bank distribution").to(device)
    if (bank["tokens"].ndim != 2 or len(bank["tokens"]) != len(bank_target)
            or bank["events"].shape != (len(bank_target),)
            or bank["images"].shape != (len(bank_target),)):
        raise ValueError("Invalid full Train bank")
    if not len(bank_target):
        raise ValueError("Empty Train bank")
    bank_keys = encode_keys(bank["tokens"].to(device), old_projector)
    bank_events = bank["events"].to(device)
    bank_images = bank["images"].to(device)
    labels = target.argmax(-1)
    predicted = base.argmax(-1)
    pure_intact_correct = (labels == 1) & (target[:, 1] >= purity) & (predicted == 1)
    missed_damaged = (labels == 2) & (target[:, 2] >= purity) & (predicted != 2)
    intact_candidates = torch.where(pure_intact_correct)[0]
    misleading = torch.zeros(count, dtype=torch.bool)
    valid_vote = 0
    candidate_topk = min(topk, len(bank_target))
    # Iterate in cache order, so each batch generally touches only a few
    # memory-mapped image shards. No per-query score matrix is retained.
    for start in range(0, len(intact_candidates), batch_size):
        chosen = intact_candidates[start:start + batch_size]
        query = store.batch(chosen, device)
        query_key = encode_keys(query["tokens"], old_projector)
        scores = query_key @ bank_keys.T
        allowed = ((query["events"][:, None] != bank_events[None, :])
                   & (query["images"][:, None] != bank_images[None, :]))
        scores.masked_fill_(~allowed, -torch.inf)
        values, ids = scores.topk(candidate_topk, dim=1)
        has_support = torch.isfinite(values).any(-1)
        if not has_support.any():
            continue
        chosen = chosen[has_support.cpu()]
        values, ids = values[has_support], ids[has_support]
        # Softmax is safe now: each retained row has at least one finite score.
        weights = (values / temperature).softmax(-1)
        votes = (weights[:, :, None] * bank_target[ids]).sum(1)
        misleading[chosen] = ((votes[:, 2] - votes[:, 1]) > risk_margin).cpu()
        valid_vote += len(chosen)
    stats = dict(
        query_regions=count,
        pure_intact_j1_correct=int(pure_intact_correct.sum()),
        pure_intact_with_support=valid_vote,
        misleading_intact=int(misleading.sum()),
        pure_damaged_j1_missed=int(missed_damaged.sum()),
        topk=topk,
        vote_temperature=temperature,
        purity=purity,
        risk_margin=risk_margin,
        interpretation="Train-only Top-K soft-label vote, not Reader behavior",
    )
    return misleading, missed_damaged, stats


class RetrievalMisleadSampler:
    """Balanced Train queries with explicit Intact and Damaged risk pools.

    Background/Destroyed retain the original J1-hardness quartile scheme;
    Intact/Damaged split into ordinary and retrieval-risk pools. Classes are
    only used for sampling and supervised metric loss, never for Val retrieval.
    """

    def __init__(self, distribution, base_probability, misleading, missed_damaged,
                 *, hard_quantile: float = .75, require_target: bool = True):
        self.baseline = HardQuerySampler(distribution, base_probability,
                                         hard_quantile=hard_quantile)
        count = len(self.baseline.classes)
        if misleading.shape != (count,) or missed_damaged.shape != (count,):
            raise ValueError("Risk pools must contain one flag per Train query")
        misleading = misleading.detach().cpu().bool()
        missed_damaged = missed_damaged.detach().cpu().bool()
        if (misleading & (self.baseline.classes != 1)).any():
            raise ValueError("Misleading pool contains non-Intact queries")
        if (missed_damaged & (self.baseline.classes != 2)).any():
            raise ValueError("Missed-Damaged pool contains non-Damaged queries")
        self.classes = self.baseline.classes
        self.pools = list(self.baseline.pools)
        for class_id, risk in ((1, misleading), (2, missed_damaged)):
            ids = torch.where(self.classes == class_id)[0]
            ordinary = ids[~risk[ids]]
            difficult = ids[risk[ids]]
            if require_target and (not len(ordinary) or not len(difficult)):
                raise ValueError(f"Train class {class_id} lacks an ordinary or target-risk pool")
            self.pools[class_id] = (ordinary, difficult)
        if require_target and any(not len(ordinary) and not len(difficult)
                                  for ordinary, difficult in self.pools):
            raise ValueError("Formal Train query cache lacks a class")
        self.inventory = {
            str(class_id): dict(ordinary=len(ordinary), target_risk=len(difficult),
                                total=len(ordinary) + len(difficult))
            for class_id, (ordinary, difficult) in enumerate(self.pools)
        }

    def sample(self, per_class: int, generator: torch.Generator, hard_fraction: float = .5):
        if per_class < 2 or not 0 < hard_fraction < 1:
            raise ValueError("Require per_class >= 2 and 0 < hard_fraction < 1")
        hard_count = max(1, min(per_class - 1, round(per_class * hard_fraction)))
        ordinary_count = per_class - hard_count
        selected, statistics = [], {}

        def draw(pool: torch.Tensor, count: int) -> torch.Tensor:
            if not len(pool):
                return pool
            if len(pool) >= count:
                return pool[torch.randperm(len(pool), generator=generator)[:count]]
            return pool[torch.randint(len(pool), (count,), generator=generator)]

        for class_id, (ordinary, difficult) in enumerate(self.pools):
            ordinary_take = ordinary_count if len(difficult) else per_class
            difficult_take = hard_count if len(ordinary) else per_class
            easy_ids = draw(ordinary, ordinary_take)
            hard_ids = draw(difficult, difficult_take)
            ids = torch.cat((easy_ids, hard_ids))
            selected.append(ids)
            statistics[str(class_id)] = dict(ordinary=len(easy_ids), target_risk=len(hard_ids),
                                             sampled=len(ids), risk_replacement=len(difficult) < difficult_take)
        indices = torch.cat(selected)
        if len(indices):
            indices = indices[torch.randperm(len(indices), generator=generator)]
        return indices, dict(by_class=statistics, sampled_anchor_count=len(indices),
                             hard_fraction=hard_fraction)
