# -*- coding: utf-8 -*-
"""Hard-pixel objective for the isolated P3 Stage2 Occurrence branch."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _hardest(values: torch.Tensor, count: int, *, largest: bool) -> torch.Tensor:
    if values.numel() <= count:
        return values
    order = torch.topk(values.detach(), k=count, largest=largest, sorted=False).indices
    return values[order]


def hard_stage2_occurrence_loss(
    new_occurrence: torch.Tensor,
    base_occurrence: torch.Tensor,
    occurrence_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    hard_fraction: float = 0.25,
    minimum_hard_pixels: int = 64,
    required_margin: float = 0.0,
    protection_weight: float = 0.10,
    ignore_index: int = 255,
) -> tuple[torch.Tensor | None, Dict[str, float | int | bool]]:
    """Mine weak Damaged and risky Intact pixels using frozen-P3 margins.

    An optimiser step is permitted only when both classes are present.  This
    avoids the empty/easy batches that diluted P3-O1.  Destroyed and unselected
    Intact pixels constrain the learned residual toward zero but are not used
    as hard negatives for Damaged severity.
    """

    if new_occurrence.ndim == 4 and new_occurrence.shape[1] == 1:
        new_occurrence = new_occurrence[:, 0]
    if base_occurrence.ndim == 4 and base_occurrence.shape[1] == 1:
        base_occurrence = base_occurrence[:, 0]
    if occurrence_delta.ndim == 4 and occurrence_delta.shape[1] == 1:
        occurrence_delta = occurrence_delta[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if not (
        new_occurrence.shape
        == base_occurrence.shape
        == occurrence_delta.shape
        == target.shape
    ):
        raise ValueError("occurrence tensors and target must share [B,H,W]")
    if not 0.0 < float(hard_fraction) <= 1.0:
        raise ValueError("hard_fraction must be in (0,1]")
    if int(minimum_hard_pixels) <= 0:
        raise ValueError("minimum_hard_pixels must be positive")

    valid = target != int(ignore_index)
    damaged_mask = valid & (target == 2)
    intact_mask = valid & (target == 1)
    destroyed_mask = valid & (target == 3)
    damaged_count = int(damaged_mask.sum().item())
    intact_count = int(intact_mask.sum().item())
    if damaged_count == 0 or intact_count == 0:
        return None, {
            "usable": False,
            "damaged_pixels": damaged_count,
            "intact_pixels": intact_count,
            "selected_per_class": 0,
        }

    desired = max(
        int(minimum_hard_pixels),
        int(round(float(hard_fraction) * damaged_count)),
    )
    selected = min(desired, damaged_count, intact_count)
    base_damaged = base_occurrence[damaged_mask]
    base_intact = base_occurrence[intact_mask]
    new_damaged = new_occurrence[damaged_mask]
    new_intact = new_occurrence[intact_mask]

    # Frozen P3 defines difficulty: lowest D margins and highest I margins.
    d_indices = torch.topk(
        base_damaged.detach(), selected, largest=False, sorted=False
    ).indices
    i_indices = torch.topk(
        base_intact.detach(), selected, largest=True, sorted=False
    ).indices
    hard_damaged = new_damaged[d_indices]
    hard_intact = new_intact[i_indices]
    margin = float(required_margin)
    damaged_loss = F.softplus(margin - hard_damaged).mean()
    intact_loss = F.softplus(margin + hard_intact).mean()
    boundary_loss = 0.5 * (damaged_loss + intact_loss)

    protect_mask = destroyed_mask.clone()
    # Easy Intact locations should not drift merely because they share a head.
    if intact_count > selected:
        easy_i = torch.ones_like(base_intact, dtype=torch.bool)
        easy_i[i_indices] = False
        intact_delta = occurrence_delta[intact_mask]
        protect_values = [intact_delta[easy_i]]
    else:
        protect_values = []
    if bool(destroyed_mask.any()):
        protect_values.append(occurrence_delta[destroyed_mask])
    if protect_values:
        protection_loss = torch.cat(protect_values).float().pow(2).mean()
    else:
        protection_loss = boundary_loss * 0.0
    loss = boundary_loss + float(protection_weight) * protection_loss

    diagnostics: Dict[str, float | int | bool] = {
        "usable": True,
        "damaged_pixels": damaged_count,
        "intact_pixels": intact_count,
        "destroyed_pixels": int(protect_mask.sum().item()),
        "selected_per_class": int(selected),
        "base_hard_damaged_margin": float(base_damaged[d_indices].mean()),
        "base_hard_intact_margin": float(base_intact[i_indices].mean()),
        "new_hard_damaged_margin": float(hard_damaged.detach().mean()),
        "new_hard_intact_margin": float(hard_intact.detach().mean()),
        "damaged_loss": float(damaged_loss.detach()),
        "intact_loss": float(intact_loss.detach()),
        "protection_loss": float(protection_loss.detach()),
    }
    return loss, diagnostics


__all__ = ["hard_stage2_occurrence_loss"]
