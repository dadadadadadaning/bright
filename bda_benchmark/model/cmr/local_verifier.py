# -*- coding: utf-8 -*-
"""Local-token correlation evidence for the CMR-V0 rescue verifier."""
from __future__ import annotations

import torch
import torch.nn.functional as F


LOCAL_CORRELATION_NAMES = (
    "query_to_support_mean",
    "query_to_support_std",
    "query_to_support_q90",
    "support_to_query_mean",
    "support_to_query_q90",
    "query_peak_margin_mean",
)


def _masked_stats(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=1).clamp_min(1)
    mean = (values * mask).sum(dim=1) / count
    variance = ((values - mean[:, None]).square() * mask).sum(dim=1) / count
    rows = []
    for row in range(values.shape[0]):
        valid = values[row, mask[row]]
        rows.append(torch.quantile(valid, 0.90))
    return mean, variance.clamp_min(0).sqrt(), torch.stack(rows)


def token_correlation_statistics(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Summarise bidirectional local correlation for aligned token sets.

    Inputs describe B independent Query--Support pairs.  Masks prevent padded
    tokens from influencing any statistic.
    """

    q = F.normalize(query_tokens.float(), dim=2, eps=1e-8)
    s = F.normalize(support_tokens.float(), dim=2, eps=1e-8)
    qm = query_mask.bool()
    sm = support_mask.bool()
    if q.ndim != 3 or s.ndim != 3 or q.shape[0] != s.shape[0] or q.shape[2] != s.shape[2]:
        raise ValueError("local verifier expects aligned [B,T,D] token tensors")
    if qm.shape != q.shape[:2] or sm.shape != s.shape[:2]:
        raise ValueError("local verifier masks do not match token tensors")
    if bool((qm.sum(dim=1) == 0).any()) or bool((sm.sum(dim=1) == 0).any()):
        raise ValueError("every local pair needs at least one valid token")

    correlation = torch.einsum("bqd,bsd->bqs", q, s)
    valid_pair = qm[:, :, None] & sm[:, None, :]
    correlation = correlation.masked_fill(~valid_pair, float("-inf"))
    query_best = correlation.max(dim=2).values.masked_fill(~qm, 0)
    support_best = correlation.max(dim=1).values.masked_fill(~sm, 0)
    query_mean, query_std, query_q90 = _masked_stats(query_best, qm)
    support_mean, _, support_q90 = _masked_stats(support_best, sm)

    take = min(2, correlation.shape[2])
    peaks = torch.topk(correlation, k=take, dim=2).values
    if take == 2:
        second = peaks[:, :, 1]
        # A one-token support has no meaningful Top-1/Top-2 margin.
        second = torch.where(sm.sum(dim=1)[:, None] >= 2, second, peaks[:, :, 0])
        margin = peaks[:, :, 0] - second
    else:
        margin = torch.zeros_like(peaks[:, :, 0])
    margin = margin.masked_fill(~qm, 0)
    margin_mean = (margin * qm).sum(dim=1) / qm.sum(dim=1).clamp_min(1)
    result = torch.stack(
        (query_mean, query_std, query_q90, support_mean, support_q90, margin_mean),
        dim=1,
    )
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite local correlation statistic")
    return result
