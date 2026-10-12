"""Label evidence and a rejectable correction of a frozen public baseline.

Retrieval never accepts query targets.  ``posterior`` is a similarity-weighted
support-label mixture, not a calibrated probability or an oracle prediction.
The reader only returns a correction; the caller owns and freezes J1 and uses
the existing Q unpooling to add that correction to J1's pixel logits.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import (
    encode_keys,
    select_ids,
)


QUALITY_NAMES = (
    "top1_cosine",
    "top1_top2_gap",
    "valid_fraction",
    "label_entropy",
    "mean_support_purity",
    "unique_event_fraction",
)


def _summarize(evidence: dict[str, Any]) -> dict[str, Any]:
    """Recompute every aggregate from valid slots, including after dropout."""
    result = dict(evidence)
    valid = result["valid"].bool()
    scores = torch.where(valid, result["scores"].float(), 0.0)
    distribution = torch.where(valid[..., None], result["support_distribution"].float(), 0.0)
    events = torch.where(valid, result["support_events"], -1)
    count = valid.sum(-1)
    supported = count > 0
    # Explicitly replace an entirely masked row before softmax. Multiplying
    # an all-masked softmax's NaN by zero would not provide exact fallback.
    scaled = (scores / float(result["temperature"])).masked_fill(~valid, -torch.inf)
    scaled = torch.where(supported[:, None], scaled, torch.zeros_like(scaled))
    weights = torch.where(valid, scaled.softmax(-1), torch.zeros_like(scaled))
    posterior = (weights[..., None] * distribution).sum(1)
    sorted_scores = scores.masked_fill(~valid, -torch.inf).topk(min(2, scores.shape[1]), dim=-1).values
    top1 = torch.where(supported, sorted_scores[:, 0], 0.0)
    if scores.shape[1] > 1:
        # Fill the second score before subtraction so even internal values
        # stay finite for rows with fewer than two supports.
        second = torch.where(count > 1, sorted_scores[:, 1], top1)
        gap = top1 - second
    else:
        gap = torch.zeros_like(top1)
    entropy = -(posterior * posterior.clamp_min(1e-8).log()).sum(-1) / math.log(4.0)
    purity = distribution.max(-1).values.sum(-1) / count.clamp_min(1)
    # Mark a slot as unique only when no earlier VALID slot shares its event.
    same_event = events[:, :, None] == events[:, None, :]
    earlier = torch.ones(valid.shape[1], valid.shape[1], dtype=torch.bool, device=valid.device).tril(-1)
    seen_earlier = (same_event & earlier[None] & valid[:, None, :]).any(-1)
    unique_fraction = (valid & ~seen_earlier).sum(-1).float() / count.clamp_min(1)
    quality = torch.stack((top1, gap, count.float() / valid.shape[1], entropy, purity, unique_fraction), -1)
    result.update(
        selected=torch.where(valid, result["selected"], -1),
        scores=scores,
        valid=valid,
        support_distribution=distribution,
        support_events=events,
        weights=weights,
        posterior=posterior,
        supported=supported,
        quality=quality,
    )
    return result


@torch.no_grad()
def retrieve_evidence(
    query: torch.Tensor,
    bank: dict[str, Any],
    event_id: int | torch.Tensor,
    image_id: int | torch.Tensor,
    topk: int = 8,
    projector: nn.Module | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Cross-event AND cross-image top-k, with no query-label-dependent policy.

    Selection uses the original J3 implementation. Scores are recomputed for
    selected keys in the same metric; values are the unchanged soft bank labels.
    Missing/dropped slots have ID -1 and zero score, value and weight. The
    ``valid`` mask, never a zero score, defines whether a support exists.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if query.ndim != 2 or not torch.isfinite(query).all():
        raise ValueError("query must be a finite [N,D] tensor")
    if "distribution" not in bank:
        raise ValueError("bank requires soft label distribution")
    distribution = bank["distribution"]
    if distribution.shape != (len(bank["tokens"]), 4):
        raise ValueError("bank distribution must be [M,4]")
    if distribution.device != query.device:
        raise ValueError("bank distribution and query must use the same device")
    if not torch.isfinite(bank["tokens"]).all() or not torch.isfinite(distribution).all():
        raise ValueError("bank features and distributions must be finite")
    if bool((distribution < 0).any()) or not torch.allclose(
        distribution.float().sum(-1), torch.ones(len(distribution), device=query.device), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("bank distributions must be nonnegative and sum to one")
    selected = select_ids(query, bank, event_id, image_id, topk, projector)
    valid = selected >= 0
    shape = selected.shape
    if len(bank["tokens"]):
        safe = selected.clamp_min(0)
        query_key = encode_keys(query, projector)
        support_key = encode_keys(bank["tokens"][safe], projector)
        scores = (query_key[:, None] * support_key).sum(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("projector produced non-finite scores")
        values = distribution[safe].float()
        events = bank["events"].to(query.device)[safe]
    else:
        scores = query.new_zeros(shape, dtype=torch.float32)
        values = query.new_zeros((*shape, 4), dtype=torch.float32)
        events = selected.new_full(shape, -1)
    return _summarize(dict(
        selected=selected, scores=scores, valid=valid,
        support_distribution=values, support_events=events,
        temperature=float(temperature),
    ))


def drop_support(
    evidence: dict[str, Any],
    probability: float,
    generator: torch.Generator | None = None,
    row_probability: float = 0.0,
) -> dict[str, Any]:
    """Independent Bernoulli dropout per support, preserving the top-k width.

    ``row_probability`` independently drops all supports for a query. No
    replacement retrieval occurs: this simulates unavailable evidence, not
    false labels or adversarial supports. Fully dropped rows exercise exact
    baseline fallback (their masked delta supplies no correction gradient).
    All posterior and quality statistics are recomputed;
    the input dictionary and tensors are left unchanged. For a seeded CPU
    generator, draws happen on CPU then move to the evidence device.
    """
    if not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1:
        raise ValueError("drop probability must lie in [0,1]")
    if not math.isfinite(float(row_probability)) or not 0 <= float(row_probability) <= 1:
        raise ValueError("row drop probability must lie in [0,1]")
    result = dict(evidence)
    valid = evidence["valid"]
    draw_device = valid.device if generator is None else generator.device
    draw = torch.rand(valid.shape, generator=generator, device=draw_device).to(valid.device)
    row_draw = torch.rand((len(valid), 1), generator=generator, device=draw_device).to(valid.device)
    result["valid"] = valid & (draw >= float(probability)) & (row_draw >= float(row_probability))
    return _summarize(result)


class GatedMemoryCorrection(nn.Module):
    """Bounded logit correction with strict reject-to-baseline semantics.

    The delta head is zero-initialized, while the initial gate is 0.5. Thus the
    delta learns immediately; after its first update the gate can also learn.
    There is no un-gated query residual. Baseline parameters are not owned by
    this module and baseline/query inputs are detached at the boundary.
    """

    def __init__(self, dim: int = 128, hidden: int = 64, bound: float = 2.0):
        super().__init__()
        if dim < 1 or hidden < 1 or not math.isfinite(float(bound)) or bound <= 0:
            raise ValueError("dim, hidden and finite bound must be positive")
        self.dim, self.bound = int(dim), float(bound)
        self.encoder = nn.Sequential(
            nn.Linear(self.dim + 4 + 2 + 4 + len(QUALITY_NAMES), int(hidden)),
            nn.GELU(), nn.Linear(int(hidden), int(hidden)), nn.GELU(),
        )
        self.delta_head = nn.Linear(int(hidden), 4)
        self.gate_head = nn.Linear(int(hidden), 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        query: torch.Tensor,
        base_probability: torch.Tensor,
        evidence: dict[str, Any],
        mode: str = "gated",
        force_off: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if mode not in ("gated", "ungated"):
            raise ValueError("mode must be gated or ungated")
        if query.ndim != 2 or query.shape[1] != self.dim:
            raise ValueError("query must be [N,dim]")
        if base_probability.shape != (len(query), 4):
            raise ValueError("base_probability must be [N,4]")
        if evidence["posterior"].shape != (len(query), 4) or evidence["quality"].shape != (len(query), len(QUALITY_NAMES)):
            raise ValueError("evidence posterior/quality shape mismatch")
        if evidence["supported"].shape != (len(query),):
            raise ValueError("evidence supported shape mismatch")
        # Sanitize before arithmetic, including rejection paths: zero times
        # NaN is not zero. Ordinary retrieval raises on corrupt finite inputs.
        clean = lambda x: torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        q = F.normalize(clean(query), dim=-1)
        base = clean(base_probability).clamp_min(0)
        total = base.sum(-1, keepdim=True)
        base = torch.where(total > 0, base / total.clamp_min(1e-8), torch.full_like(base, 0.25))
        entropy = -(base * base.clamp_min(1e-8).log()).sum(-1, keepdim=True) / math.log(4.0)
        leading = base.topk(2, dim=-1).values
        margin = leading[:, :1] - leading[:, 1:2]
        posterior, quality = clean(evidence["posterior"]), clean(evidence["quality"])
        hidden = self.encoder(torch.cat((q, base, entropy, margin, posterior, quality), -1))
        raw_delta = self.delta_head(hidden).tanh() * self.bound
        supported = evidence["supported"].bool() & evidence["valid"].any(-1)
        if force_off:
            supported = torch.zeros_like(supported)
        candidate_gate = self.gate_head(hidden).squeeze(-1).sigmoid() if mode == "gated" else torch.ones_like(raw_delta[:, 0])
        gate = torch.where(supported, candidate_gate, torch.zeros_like(candidate_gate))
        delta = torch.where(supported[:, None], gate[:, None] * raw_delta, torch.zeros_like(raw_delta))
        return delta, gate, dict(supported=supported, raw_delta=raw_delta, posterior=posterior, quality=quality)


__all__ = ["QUALITY_NAMES", "retrieve_evidence", "drop_support", "GatedMemoryCorrection"]
