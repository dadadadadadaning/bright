"""Fixed-Q two-slot pooling. Slot 0/1 means a routing role, never a GT class.

Both modes use identical shapes. ``soft`` assigns Q*(1-B), Q*B; the spatial
control assigns a fixed 4-cell checkerboard. Routes do not use query labels.
No epsilon floors alter the mean of a positive-mass slot.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

CONTROL_TILE = 4


def finite_fp32(name: str, value: Tensor) -> None:
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite float32")


def validate_routes(q: Tensor, ids: Tensor, count: int, batch: int, pixels: int) -> Tensor:
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    finite_fp32("q", q)
    if q.ndim != 3 or q.shape[0] != batch or q.shape[2] != pixels or q.shape[1] < 1:
        raise ValueError("q must have shape [B,S,N]")
    if ids.dtype != torch.long or ids.device != q.device:
        raise ValueError("ids must be int64 on the q device")
    if ids.shape == q.shape[1:]:
        ids = ids.unsqueeze(0).expand(batch, -1, -1)
    elif ids.shape != q.shape:
        raise ValueError("ids must have shape [S,N] or [B,S,N]")
    if bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError("out-of-range route ids")
    if bool((q < 0).any()) or not torch.allclose(
        q.sum(1), torch.ones_like(q[:, 0]), atol=2e-5, rtol=2e-5
    ):
        raise ValueError("route weights must be nonnegative and sum to one")
    return ids


def route_from_q(q: Tensor, ids: Tensor, count: int, shape: tuple[int, int],
                 building: Tensor, mode: str) -> tuple[Tensor, Tensor, int]:
    if mode not in ("soft", "capacity_control"):
        raise ValueError("mode must be soft or capacity_control")
    if len(shape) != 2 or min(shape) < 1 or q.ndim != 3 or q.shape[1] != 9:
        raise ValueError("expected original nine-candidate Q and valid grid shape")
    batch = q.shape[0]
    ids = validate_routes(q, ids, count, batch, shape[0] * shape[1])
    if building.ndim != 4 or building.shape[:2] != (batch, 1) or min(building.shape[2:]) < 1:
        raise ValueError("building must have shape [B,1,H,W]")
    if building.device != q.device:
        raise ValueError("building and Q must share a device")
    finite_fp32("building", building)
    if bool(((building < 0) | (building > 1)).any()):
        raise ValueError("building probabilities must be in [0,1]")
    q = q.detach()
    if mode == "soft":
        split = F.interpolate(building.detach(), size=shape, mode="bilinear", align_corners=False).flatten(2)
    else:
        yy, xx = torch.meshgrid(torch.arange(shape[0], device=q.device),
                                torch.arange(shape[1], device=q.device), indexing="ij")
        split = ((yy // CONTROL_TILE + xx // CONTROL_TILE) % 2).reshape(1, 1, -1).to(q.dtype)
    return (torch.cat((q * (1.0 - split), q * split), 1),
            torch.cat((2 * ids, 2 * ids + 1), 1), 2 * count)


def _sums(values: Tensor, q: Tensor, ids: Tensor, count: int) -> tuple[Tensor, Tensor]:
    if values.ndim != 4 or min(values.shape) < 1:
        raise ValueError("values must be nonempty [B,C,h,w]")
    finite_fp32("values", values)
    batch, channels, height, width = values.shape
    ids = validate_routes(q, ids, count, batch, height * width)
    if values.device != q.device:
        raise ValueError("values and Q must share a device")
    flat = values.flatten(2).transpose(1, 2)
    total = values.new_zeros(batch, count, channels)
    mass = values.new_zeros(batch, count)
    for candidate in range(q.shape[1]):
        index = ids[:, candidate]
        weight = q[:, candidate]
        total = total.scatter_add(1, index[..., None].expand(-1, -1, channels), flat * weight[..., None])
        mass = mass.scatter_add(1, index, weight)
    return total, mass


def slot_pool(values: Tensor, route_q: Tensor, route_ids: Tensor,
              count: int) -> tuple[Tensor, Tensor]:
    if count < 2 or count % 2:
        raise ValueError("slot count must be positive and even")
    numerator, mass = _sums(values, route_q, route_ids, count)
    divisor = torch.where(mass > 0, mass, torch.ones_like(mass))
    means = numerator / divisor[..., None]
    return means.reshape(values.shape[0], count // 2, 2, values.shape[1]), mass.reshape(values.shape[0], count // 2, 2)


def slot_labels(target: Tensor, route_q: Tensor, route_ids: Tensor,
                shape: tuple[int, int], count: int) -> tuple[Tensor, Tensor]:
    """Area-sample valid one-hot mass, then pool with the feature route weights.

    ``valid_label_mass`` is measured in feature-grid units, not raw image
    pixels. A partly ignored grid cell contributes proportionally; its labels
    are never normalized before pooling. Only classes 0..3 and ignore 255 are
    accepted. This is support Train-label processing, not a query input.
    """
    if target.ndim != 3 or min(target.shape) < 1 or len(shape) != 2 or min(shape) < 1:
        raise ValueError("target must be [B,H,W], with valid feature shape")
    if target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("target labels must be integer")
    if target.device != route_q.device or target.shape[0] != route_q.shape[0]:
        raise ValueError("target batch/device mismatch")
    if count < 2 or count % 2:
        raise ValueError("slot count must be positive and even")
    valid = target != 255
    if bool((valid & ((target < 0) | (target > 3))).any()):
        raise ValueError("unknown target class (expected 0..3 or 255)")
    safe = torch.where(valid, target, 0).long()
    onehot = F.one_hot(safe, 4).permute(0, 3, 1, 2).float() * valid[:, None]
    grid = F.interpolate(onehot, size=shape, mode="area")
    numerator, _ = _sums(grid, route_q, route_ids, count)
    mass = numerator.sum(-1)
    labels = numerator / torch.where(mass > 0, mass, torch.ones_like(mass))[..., None]
    return labels.reshape(target.shape[0], count // 2, 2, 4), mass.reshape(target.shape[0], count // 2, 2)


def prepare_slots(feature: Tensor, q: Tensor, ids: Tensor, count: int,
                  shape: tuple[int, int], building: Tensor, rule: str) -> dict[str, Tensor | int]:
    if feature.shape[-2:] != tuple(shape):
        raise ValueError("feature shape and supplied grid shape disagree")
    route_q, route_ids, slot_count = route_from_q(q, ids, count, shape, building, rule)
    means, mass = slot_pool(feature, route_q, route_ids, slot_count)
    parent_mass = mass.sum(-1, keepdim=True)
    fraction = mass / torch.where(parent_mass > 0, parent_mass, torch.ones_like(parent_mass))
    return dict(slot_f=means, slot_mass=mass, slot_fraction=fraction, slot_valid=mass > 0,
                route_q=route_q, route_ids=route_ids, slot_count=slot_count)


def sparse_unpool(values: Tensor, q: Tensor, ids: Tensor) -> Tensor:
    """Validated same-weight readback [B,R,C] -> [B,C,N]."""
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("values must be [B,R,C]")
    finite_fp32("region values", values)
    ids = validate_routes(q, ids, values.shape[1], values.shape[0], q.shape[-1])
    if values.device != q.device:
        raise ValueError("values and Q must share a device")
    result = values.new_zeros(values.shape[0], q.shape[-1], values.shape[-1])
    for candidate in range(q.shape[1]):
        chosen = values.gather(1, ids[:, candidate, :, None].expand(-1, -1, values.shape[-1]))
        result = result + chosen * q[:, candidate, :, None]
    return result.transpose(1, 2)
