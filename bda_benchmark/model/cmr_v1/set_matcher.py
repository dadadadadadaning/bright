# -*- coding: utf-8 -*-
"""Set-to-set local case matching for CMR-V1.

All functions are training-free and operate on already extracted, frozen
P3/MICB local tokens.  They are intentionally small so the V1 audit can compare
multiple matching hypotheses without changing the base model.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _valid_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = tokens[mask.bool()].float()
    if x.ndim != 2 or x.shape[0] == 0:
        raise RuntimeError("set matcher received an empty token set")
    return F.normalize(x, dim=1, eps=1e-8)


def cosine_matrix(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    q = _valid_tokens(query_tokens, query_mask)
    s = _valid_tokens(support_tokens, support_mask)
    return q @ s.t()


def symmetric_chamfer_score(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    sim = cosine_matrix(query_tokens, query_mask, support_tokens, support_mask)
    return 0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean())


def dn4_score(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    neighbor_k: int = 3,
) -> torch.Tensor:
    """Local-descriptor image-to-case score inspired by DN4."""
    sim = cosine_matrix(query_tokens, query_mask, support_tokens, support_mask)
    take = min(max(int(neighbor_k), 1), sim.shape[1])
    return sim.topk(take, dim=1, largest=True, sorted=False).values.mean(dim=1).mean()


def sinkhorn_emd_score(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    epsilon: float = 0.07,
    iterations: int = 40,
) -> torch.Tensor:
    """Balanced entropic OT score over local cosine affinity (DeepEMD-style).

    Uniform masses are used deliberately in V1-A.  If this audit works, a later
    stage can replace uniform mass by frozen P3 occurrence/change-energy masses.
    """
    if epsilon <= 0 or iterations <= 0:
        raise ValueError("epsilon and iterations must be positive")
    sim = cosine_matrix(query_tokens, query_mask, support_tokens, support_mask)
    nq, ns = sim.shape
    log_a = sim.new_full((nq,), -math.log(float(nq)))
    log_b = sim.new_full((ns,), -math.log(float(ns)))
    log_kernel = sim / float(epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(iterations)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    transport = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    transport = transport / transport.sum().clamp_min(1e-12)
    return (transport * sim).sum()


def frn_reconstruction_score(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Support-to-query ridge reconstruction score inspired by FRN.

    Higher is better.  The score is negative mean reconstruction error.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    q = _valid_tokens(query_tokens, query_mask)
    s = _valid_tokens(support_tokens, support_mask)
    gram = s @ s.t()
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    # A = Q S^T (S S^T + lambda I)^-1
    rhs = (q @ s.t()).t()  # [Ns, Nq]
    coeff_t = torch.linalg.solve(gram + float(ridge) * identity, rhs)
    coeff = coeff_t.t()
    reconstructed = coeff @ s
    error = (q - reconstructed).square().sum(dim=1).mean()
    return -error


def pair_score(
    method: str,
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    method = str(method).lower()
    if method == "chamfer":
        return symmetric_chamfer_score(query_tokens, query_mask, support_tokens, support_mask)
    if method == "dn4":
        return dn4_score(
            query_tokens, query_mask, support_tokens, support_mask,
            neighbor_k=int(kwargs.get("neighbor_k", 3)),
        )
    if method in {"emd", "sinkhorn", "deepemd"}:
        return sinkhorn_emd_score(
            query_tokens, query_mask, support_tokens, support_mask,
            epsilon=float(kwargs.get("epsilon", 0.07)),
            iterations=int(kwargs.get("iterations", 40)),
        )
    if method in {"frn", "reconstruction"}:
        return frn_reconstruction_score(
            query_tokens, query_mask, support_tokens, support_mask,
            ridge=float(kwargs.get("ridge", 1e-2)),
        )
    raise ValueError(f"unknown set-matching method: {method}")


def class_logmeanexp(scores: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """Smooth class evidence without rewarding classes merely for having more cases."""
    if scores.numel() == 0:
        return scores.new_tensor(float("-inf"))
    if temperature <= 0:
        return scores.max()
    scaled = scores / float(temperature)
    return float(temperature) * (
        torch.logsumexp(scaled, dim=0) - math.log(float(scores.numel()))
    )
