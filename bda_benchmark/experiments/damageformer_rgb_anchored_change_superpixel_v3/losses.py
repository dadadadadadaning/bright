"""Damage-aware supervision for a shared soft superpixel assignment."""
from __future__ import annotations

import os
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.association import (
    soft_pool_mean_std,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.losses import (
    stage_losses as v2_stage_losses,
    weak_building_bag_loss,
)


def _zero(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return aux["association_probabilities"].float().sum() * 0.0


def damage_semantic_reconstruction_loss(
    aux: Mapping[str, torch.Tensor],
    damage_target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Reconstruct I/D/X labels through Q, with extra weight on rare damage."""
    probabilities = aux["association_probabilities"]
    candidate_ids = aux["candidate_ids"]
    b, _, n = probabilities.shape
    h, w = aux["building_ids"].shape[-2:]
    if damage_target.ndim == 4:
        damage_target = damage_target[:, 0]
    target = F.interpolate(
        damage_target[:, None].float(), size=(h, w), mode="nearest"
    )[:, 0].long()
    building = aux["building_ids"] > 0
    valid = building & (target != ignore_index) & (target >= 1) & (target <= 3)
    if not bool(valid.any().item()):
        return _zero(aux)

    safe = (target - 1).clamp(0, 2)
    one_hot = F.one_hot(safe, num_classes=3).permute(0, 3, 1, 2).float()
    token_count = int(aux["token_count"].item())
    reconstruction = soft_pool_mean_std(
        one_hot, probabilities, candidate_ids, token_count
    )["reconstruction"].float().clamp_min(1.0e-6)
    nll = -torch.log(torch.gather(reconstruction, 1, safe[:, None])[:, 0])
    # Intact=1, Damaged=2, Destroyed=3.  Preserve rare affected pixels instead
    # of allowing the regional majority to dominate the partition objective.
    class_weight = torch.where(target == 1, 1.0, torch.where(target == 2, 2.0, 2.5))
    weight = valid.float() * class_weight
    return (nll * weight).sum() / weight.sum().clamp_min(1.0)


def _shared_probability(
    p_left: torch.Tensor,
    id_left: torch.Tensor,
    p_right: torch.Tensor,
    id_right: torch.Tensor,
) -> torch.Tensor:
    """Probability that adjacent pixels choose the same global region token."""
    same_id = id_left[:, :, None] == id_right[:, None, :]
    return (
        p_left[:, :, None] * p_right[:, None, :] * same_id.to(p_left.dtype)
    ).sum(dim=(1, 2)).clamp(1.0e-6, 1.0 - 1.0e-6)


def occurrence_boundary_affinity_loss(
    aux: Mapping[str, torch.Tensor],
    damage_target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Keep same occurrence together and split Intact/Affected neighbours."""
    p = aux["association_probabilities"].float()
    ids = aux["candidate_ids"]
    b, _, _ = p.shape
    h, w = aux["building_ids"].shape[-2:]
    p = p.reshape(b, 9, h, w)
    ids = ids.reshape(b, 9, h, w)
    if damage_target.ndim == 4:
        damage_target = damage_target[:, 0]
    target = F.interpolate(
        damage_target[:, None].float(), size=(h, w), mode="nearest"
    )[:, 0].long()
    building_ids = aux["building_ids"]
    valid = (building_ids > 0) & (target != ignore_index) & (target >= 1) & (target <= 3)
    affected = target >= 2
    losses = []
    for horizontal in (True, False):
        if horizontal:
            pl, pr = p[..., :-1], p[..., 1:]
            il, ir = ids[..., :-1], ids[..., 1:]
            vl, vr = valid[..., :-1], valid[..., 1:]
            yl, yr = affected[..., :-1], affected[..., 1:]
            bl, br = building_ids[..., :-1], building_ids[..., 1:]
        else:
            pl, pr = p[..., :-1, :], p[..., 1:, :]
            il, ir = ids[..., :-1, :], ids[..., 1:, :]
            vl, vr = valid[..., :-1, :], valid[..., 1:, :]
            yl, yr = affected[..., :-1, :], affected[..., 1:, :]
            bl, br = building_ids[..., :-1, :], building_ids[..., 1:, :]
        # Never ask two neighbouring but distinct buildings to share a token.
        pair_valid = vl & vr & (bl == br)
        if not bool(pair_valid.any().item()):
            continue
        shared = _shared_probability(
            pl.flatten(2), il.flatten(2), pr.flatten(2), ir.flatten(2)
        ).reshape_as(pair_valid)
        same = yl == yr
        pair_loss = torch.where(same, -torch.log(shared), -torch.log1p(-shared))
        # True Intact/Affected boundaries are sparse and receive more weight.
        weight = pair_valid.float() * torch.where(same, 1.0, 3.0)
        losses.append((pair_loss * weight).sum() / weight.sum().clamp_min(1.0))
    return torch.stack(losses).mean() if losses else _zero(aux)


def stage_losses(
    aux: Mapping[str, torch.Tensor],
    damage_target: torch.Tensor,
    **kwargs,
) -> Dict[str, torch.Tensor | float]:
    result = dict(v2_stage_losses(aux, damage_target, **kwargs))
    zero = _zero(aux)
    result["damage_reconstruction"] = zero
    result["occurrence_boundary"] = zero
    enabled = bool(float(aux.get("v3_use_damage_boundary", zero).detach().item()) > 0.5)
    if kwargs.get("stage") == "change" and enabled:
        result["damage_reconstruction"] = damage_semantic_reconstruction_loss(
            aux, damage_target, ignore_index=int(kwargs.get("ignore_index", 255))
        )
        result["occurrence_boundary"] = occurrence_boundary_affinity_loss(
            aux, damage_target, ignore_index=int(kwargs.get("ignore_index", 255))
        )
        reconstruction_weight = float(os.environ.get("V3_DAMAGE_RECON_WEIGHT", "0.05"))
        boundary_weight = float(os.environ.get("V3_DAMAGE_BOUNDARY_WEIGHT", "0.02"))
        result["total"] = (
            result["total"]
            + reconstruction_weight * result["damage_reconstruction"]
            + boundary_weight * result["occurrence_boundary"]
        )
    return result


__all__ = [
    "damage_semantic_reconstruction_loss",
    "occurrence_boundary_affinity_loss",
    "stage_losses",
    "weak_building_bag_loss",
]
