"""Boundary controls computed locally from a fixed sparse association Q.

The old channel measures disagreement between independent draws. The change
channel measures aligned distribution change, eliminating uncertainty alone.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def boundary_maps(q: torch.Tensor, ids: torch.Tensor, shape: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Return ``old``, ``zero``, ``mean``, ``change`` maps of shape [B,1,H,W].

    ``old`` exactly matches the existing max-over-neighbor ``1 - aligned dot``.
    ``change`` uses ``0.5 * ||global_Q_i - global_Q_j||^2`` and the same neighbor
    aggregation. Within-pixel aligned dot products aggregate repeated IDs, so
    repeated slots for a single global region do not distort squared norms.
    No dense global-region probability table is constructed.
    """
    if q.ndim != 3 or ids.ndim != 2 or not q.is_floating_point():
        raise ValueError("Expected floating Q[B,S,N] and ids[S,N]")
    batch, supports, pixels = q.shape
    height, width = shape
    if min(batch, supports, height, width) < 1 or pixels != height * width or ids.shape != (supports, pixels):
        raise ValueError("Sparse association shape mismatch")
    if ids.dtype != torch.long or ids.device != q.device or bool((ids < 0).any()):
        raise ValueError("IDs must be nonnegative torch.long on Q's device")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("Q must be finite and nonnegative")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Q must be normalized over candidate supports")
    probabilities = q.reshape(batch, supports, height, width)
    seed_ids = ids.reshape(supports, height, width)
    self_match = (seed_ids[:, None] == seed_ids[None, :]).to(q.dtype)
    squared_norm = torch.einsum("bshw,bthw,sthw->bhw", probabilities, probabilities, self_match)
    old_contributions = [q.new_zeros(batch, 1, height, width)]
    change_contributions = [q.new_zeros(batch, 1, height, width)]
    if width > 1:
        match = (seed_ids[:, None, :, :-1] == seed_ids[None, :, :, 1:]).to(q.dtype)
        same = torch.einsum("bshw,bthw,sthw->bhw", probabilities[:, :, :, :-1], probabilities[:, :, :, 1:], match)
        old = (1 - same).clamp(0, 1).unsqueeze(1)
        change = (0.5 * (squared_norm[:, :, :-1] + squared_norm[:, :, 1:] - 2 * same)).clamp(0, 1).unsqueeze(1)
        old_contributions.extend((F.pad(old, (0, 1, 0, 0)), F.pad(old, (1, 0, 0, 0))))
        change_contributions.extend((F.pad(change, (0, 1, 0, 0)), F.pad(change, (1, 0, 0, 0))))
    if height > 1:
        match = (seed_ids[:, None, :-1, :] == seed_ids[None, :, 1:, :]).to(q.dtype)
        same = torch.einsum("bshw,bthw,sthw->bhw", probabilities[:, :, :-1, :], probabilities[:, :, 1:, :], match)
        old = (1 - same).clamp(0, 1).unsqueeze(1)
        change = (0.5 * (squared_norm[:, :-1, :] + squared_norm[:, 1:, :] - 2 * same)).clamp(0, 1).unsqueeze(1)
        old_contributions.extend((F.pad(old, (0, 0, 0, 1)), F.pad(old, (0, 0, 1, 0))))
        change_contributions.extend((F.pad(change, (0, 0, 0, 1)), F.pad(change, (0, 0, 1, 0))))
    old = torch.stack(old_contributions, dim=0).amax(dim=0)
    change = torch.stack(change_contributions, dim=0).amax(dim=0)
    return {"old": old, "zero": torch.zeros_like(old), "mean": old.mean(dim=(2, 3), keepdim=True).expand_as(old), "change": change}
