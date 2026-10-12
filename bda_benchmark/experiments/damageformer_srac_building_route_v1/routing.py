"""Sparse building/background routes for the frozen SRAC association Q.

For split modes each original seed r has two global slots, 2*r and 2*r+1.
Hard/soft modes use predicted background/building probability.  The capacity
control uses a fixed spatial checkerboard, without building probability in its
split.  Both gathering and reading use the same route weights.  Ground truth
is never an input to these functions.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


MODES = ("unsplit", "hard", "soft", "capacity_control")
# The inherited SRAC Q normally places seeds every eight feature-grid cells.
# Fixed four-cell tiles split such cells into two coordinate groups, matching
# the split arms' 2K slots and 18 supports without consulting building labels.
CONTROL_TILE = 4


def _batched_ids(ids: torch.Tensor, batch: int, supports: int, pixels: int) -> torch.Tensor:
    if ids.ndim == 2 and ids.shape == (supports, pixels):
        return ids.unsqueeze(0).expand(batch, -1, -1)
    if ids.ndim == 3 and ids.shape == (batch, supports, pixels):
        return ids
    raise ValueError("Candidate IDs must be [S,N] or [B,S,N]")


def route_from_q(
    q: torch.Tensor,
    ids: torch.Tensor,
    count: int,
    shape: tuple[int, int],
    building: torch.Tensor,
    mode: str,
    *,
    threshold: float = 0.5,
    soft_floor: float = 0.0,
) -> dict[str, Any]:
    """Turn the original Q into 9- or 18-support routes.

    ``building`` is the frozen A1 probability map.  It is bilinearly sampled
    at Q's feature resolution, just as the inherited front samples it for its
    structural adapter.  A soft floor leaves some mass on the other stream
    when the building detector is overconfident; zero is the pure QB/Q(1-B)
    proposal.  The value is an experimental parameter, not a learned gate.
    ``capacity_control`` uses a fixed 4x4 feature-grid checkerboard; its route
    weights and IDs are independent of ``building``.  The returned
    ``building_grid`` is diagnostic only.  The frozen source Q and adapted
    features may themselves contain A1 building information.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown route mode: {mode}")
    if not isinstance(count, int) or count < 1:
        raise ValueError("Seed count must be a positive integer")
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Invalid feature-grid shape")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be inside (0,1)")
    if not math.isfinite(soft_floor) or not 0 <= soft_floor < 0.5:
        raise ValueError("Soft floor must be in [0,0.5)")
    if q.ndim != 3 or q.shape[1] != 9 or q.shape[2] != shape[0] * shape[1]:
        raise ValueError("Expected original Q with shape [B,9,H*W]")
    batch, supports, pixels = q.shape
    ids = _batched_ids(ids, batch, supports, pixels)
    if ids.dtype != torch.long or bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError("Candidate IDs must be int64 and inside the seed table")
    q = q.detach().float()
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("Original Q has invalid weights")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Original Q must sum to one at each location")
    if building.ndim != 4 or building.shape[0:2] != (batch, 1):
        raise ValueError("Predicted building probability must be [B,1,H,W]")
    building_grid = F.interpolate(
        building.detach().float(), size=shape, mode="bilinear", align_corners=False
    )
    if not bool(torch.isfinite(building_grid).all()) or bool(((building_grid < 0) | (building_grid > 1)).any()):
        raise ValueError("Predicted building probability must be finite in [0,1]")
    if mode == "unsplit":
        return dict(q=q, ids=ids, count=count, shape=tuple(shape),
                    building_grid=building_grid, mode=mode)
    if mode == "capacity_control":
        yy, xx = torch.meshgrid(
            torch.arange(shape[0], device=q.device),
            torch.arange(shape[1], device=q.device),
            indexing="ij",
        )
        probability = (((yy // CONTROL_TILE + xx // CONTROL_TILE) % 2)
                       .reshape(1, 1, pixels).to(q.dtype))
    else:
        probability = building_grid.flatten(2)
        if mode == "hard":
            probability = (probability >= threshold).float()
        else:
            probability = soft_floor + (1.0 - 2.0 * soft_floor) * probability
    background = q * (1.0 - probability)
    foreground = q * probability
    route_q = torch.cat((background, foreground), dim=1)
    route_ids = torch.cat((ids * 2, ids * 2 + 1), dim=1)
    return dict(q=route_q, ids=route_ids, count=count * 2, shape=tuple(shape),
                building_grid=building_grid, mode=mode)


def pool_routes(
    values: torch.Tensor,
    q: torch.Tensor,
    ids: torch.Tensor,
    count: int,
) -> dict[str, torch.Tensor]:
    """Weighted means for arbitrary sparse candidate count (9 or 18 here).

    Unlike the inherited ``soft_pool_unpool``, this supports batched 18-way
    IDs while keeping every intermediate at BxN or BxK rather than a dense
    BxKxN tensor.  A zero-mass region has a zero token and is marked invalid
    by the caller.
    """
    if values.ndim != 4 or q.ndim != 3:
        raise ValueError("Expected feature [B,C,H,W] and route [B,S,N]")
    batch, channels, height, width = values.shape
    if q.shape[0] != batch or q.shape[2] != height * width:
        raise ValueError("Feature and route grid disagree")
    if not isinstance(count, int) or count < 1:
        raise ValueError("Invalid token count")
    supports, pixels = q.shape[1:]
    ids = _batched_ids(ids, batch, supports, pixels)
    if ids.dtype != torch.long or bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError("Invalid candidate IDs")
    values = values.float().flatten(2).transpose(1, 2)
    q = q.float()
    numerator = values.new_zeros(batch, count, channels)
    mass = values.new_zeros(batch, count, 1)
    for candidate in range(supports):
        candidate_ids = ids[:, candidate].unsqueeze(-1)
        weight = q[:, candidate].unsqueeze(-1)
        numerator.scatter_add_(1, candidate_ids.expand(-1, -1, channels), values * weight)
        mass.scatter_add_(1, candidate_ids, weight)
    tokens = numerator / mass.clamp_min(1e-6)
    return dict(tokens=tokens, mass=mass.squeeze(-1))


def route_geometry(route: dict[str, Any]) -> torch.Tensor:
    """Match the original mass/centroid/spread geometry on routed slots."""
    height, width = route["shape"]
    q = route["q"]
    batch = q.shape[0]
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height, device=q.device),
        torch.linspace(0, 1, width, device=q.device),
        indexing="ij",
    )
    values = torch.stack((yy, xx, yy.square(), xx.square()))[None].expand(batch, -1, -1, -1)
    pooled = pool_routes(values, q, route["ids"], route["count"])
    mean = pooled["tokens"][..., :2]
    spread = (pooled["tokens"][..., 2:] - mean.square()).clamp_min(0).sqrt()
    return torch.cat((pooled["mass"][..., None] / float(height * width), mean, spread), dim=-1)


def read_regions(
    logits: torch.Tensor,
    route: dict[str, Any],
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Read region class probabilities with the *same* association weights."""
    q, ids, count = route["q"], route["ids"], route["count"]
    batch, supports, pixels = q.shape
    if logits.ndim != 3 or logits.shape[:2] != (batch, count) or logits.shape[-1] != 4:
        raise ValueError("Expected region logits [B,K,4]")
    if len(output_size) != 2 or min(output_size) < 1:
        raise ValueError("Invalid output size")
    ids = _batched_ids(ids, batch, supports, pixels)
    probability = logits.float().softmax(-1)
    reconstruction = probability.new_zeros(batch, pixels, 4)
    for candidate in range(supports):
        selected = probability.gather(1, ids[:, candidate, :, None].expand(-1, -1, 4))
        reconstruction = reconstruction + selected * q[:, candidate, :, None]
    reconstruction = reconstruction.transpose(1, 2).reshape(batch, 4, *route["shape"])
    reconstruction = F.interpolate(reconstruction, size=output_size, mode="bilinear", align_corners=False)
    return reconstruction / reconstruction.sum(1, keepdim=True).clamp_min(1e-8)
