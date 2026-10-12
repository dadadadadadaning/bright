"""Frozen-Q, fixed-budget cross-group gathering; output routing stays hard.

This module never consumes labels and contains no trainable parameters. SAR
features are used only to choose the distribution of a prescribed gathering
budget, not to modify Q, the building decision, or the output footprint.

The supported-uniform control is deliberately NOT the legacy ``exchange``:
opposite slots empty under the HARD reference route are excluded in both new
controls. This prevents SAR from being compared with a control that can create
new unsupported slots. No-opposite-support pixels fall back to hard gathering.
"""
import math

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import pool
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import route_from_q


@torch.no_grad()
def exchange_pool_route(reference, sar_feature, epsilon, threshold=.5,
                        temperature=.2, mode='uniform'):
    """Return ``(pool_route, scalar_diagnostics)``; caller keeps hard read route.

    reference: q [B,9,H*W], ids [9,H*W] or [B,9,H*W], tokens [B,K,D],
               building [B,1,H,W], shape (H,W).
    sar_feature: a frozen [B,C,H,W] SAR feature in these native coordinates.

    Each pixel keeps mass 1-epsilon in its own predicted group and sends total
    epsilon to reachable, already occupied opposite-group slots. Without such
    a slot it keeps mass 1. Both modes use exactly the same admissible support
    and per-pixel budget; 'uniform' means base-Q-proportional, not equal per ID.
    SAR mode reweights only that opposite-group distribution by cosine affinity
    to centers computed ONCE from the hard reference, never recursively from
    the exchanged content. This is not registration or a physical layover model.
    """
    if mode not in ('uniform', 'sar'):
        raise ValueError('mode must be uniform or sar')
    if not math.isfinite(epsilon) or not 0 <= epsilon <= .5:
        raise ValueError('epsilon must be finite and in [0, 0.5]')
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be finite and in (0, 1)')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')

    q = reference['q'].detach().float()
    tokens = reference['tokens']
    if tokens.ndim != 3 or tokens.shape[0] != q.shape[0]:
        raise ValueError('Expected reference tokens B,K,D')
    hard = route_from_q(q, reference['ids'], tokens.shape[1],
                        reference['shape'], reference['building'],
                        epsilon=0., threshold=threshold)
    b, _, n = q.shape
    h, w = hard['shape']
    if (sar_feature.ndim != 4 or sar_feature.shape[0] != b or
            sar_feature.shape[1] < 1 or tuple(sar_feature.shape[-2:]) != (h, w)):
        raise ValueError('SAR feature must share native B,C,H,W coordinates')
    if sar_feature.device != q.device or not torch.isfinite(sar_feature).all():
        raise ValueError('SAR feature must be finite and on the Q device')

    feature = F.normalize(sar_feature.detach().float(), dim=1, eps=1e-8)
    values = feature.flatten(2).transpose(1, 2)
    centers, mass = pool(values, hard['q'], hard['ids'], hard['k'])
    centers = F.normalize(centers, dim=-1, eps=1e-8)
    group = (reference['building'].detach().flatten(2) >= threshold).long()
    # Hard IDs are ordered background candidates, then building candidates.
    opposite_ids = hard['ids'][:, :9] + (1 - group)
    support = torch.stack([mass.gather(1, opposite_ids[:, j]) > 0
                           for j in range(9)], dim=1) & (q > 0)
    reachable = support.any(1, keepdim=True)
    budget = reachable.to(q.dtype) * float(epsilon)

    if mode == 'uniform':
        distribution = q * support
        distribution = distribution / distribution.sum(1, keepdim=True).clamp_min(
            torch.finfo(q.dtype).tiny)
    else:
        # Iterate candidate axis: never allocate B*9*N*C gathered feature maps.
        affinity = []
        for j in range(9):
            center = centers.gather(1, opposite_ids[:, j, :, None].expand(
                -1, -1, centers.shape[-1]))
            affinity.append((values * center).sum(-1).clamp(-1., 1.))
        similarity = torch.stack(affinity, dim=1).double()
        log_weights = q.double().clamp_min(torch.finfo(torch.float32).tiny).log()
        log_weights = (log_weights + similarity / float(temperature)).masked_fill(
            ~support, -torch.inf)
        maximum = log_weights.max(1, keepdim=True).values
        maximum = torch.where(reachable, maximum, torch.zeros_like(maximum))
        # FP64 exponent computation plus a representability floor keeps the
        # declared support identical in both modes under extremely sharp scores.
        shifted = (log_weights - maximum).clamp_min(-80.)
        weights = shifted.exp() * support
        distribution = (weights / weights.sum(1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny)).float()

    cross = distribution * budget
    exchange = torch.cat([cross * group, cross * (1 - group)], dim=1)
    routed = hard['q'] * (1 - budget) + exchange
    # eps=0 has exact old-hard routing, including zero entries.
    route = dict(hard, q=hard['q'] if epsilon == 0 else routed,
                 epsilon=float(epsilon), exchange_mode=mode)
    if (not torch.isfinite(route['q']).all() or (route['q'] < 0).any() or
            not torch.allclose(route['q'].sum(1), torch.ones_like(q[:, 0]),
                               atol=1e-6, rtol=1e-6)):
        raise RuntimeError('Invalid exchanged gathering normalization')
    opposite = (route['ids'] % 2 != group)
    actual = (route['q'] * opposite).sum(1)
    diagnostics = dict(
        requested_exchange=float(epsilon),
        actual_exchange_mean=float(actual.mean()),
        exchange_budget_max_abs_error=float((actual-budget[:, 0]).abs().max()),
        no_opposite_support_fraction=float((~reachable).float().mean()),
        opposite_candidates_mean=float(support.sum(1).float().mean()),
        hard_occupied_slots_mean=float((mass > 0).sum(1).float().mean()),
    )
    return route, diagnostics
