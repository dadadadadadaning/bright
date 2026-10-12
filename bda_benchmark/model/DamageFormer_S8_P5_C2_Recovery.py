# -*- coding: utf-8 -*-
"""P3-anchored recovery utilities for the accepted S8-P5 C2 branch.

The recovery branch follows a controlled nesting:

R0: continue the original C2 output.
R1: frozen P3 output + zero-gated C2 residual.
R2: R1 + damage-to-intact error-flow margin.
R3: R2 + building-instance supervised contrastive learning.

The gate follows the zero-initialized residual principle used by gated
cross-attention adapters.  The already-trained C2 prediction is non-zero, so a
zero gate does not create the double-zero dead-gradient failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DamageFormer_S8_MICB_DSMIL import BuildingBagSpec


@dataclass(frozen=True)
class C2RecoverySpec:
    name: str
    p3_anchor: bool
    error_flow_margin: bool
    instance_supcon: bool


C2_RECOVERY_VARIANTS: Dict[str, C2RecoverySpec] = {
    "r0": C2RecoverySpec(
        name="S8-P5-C2-R0-LongScheduleControl",
        p3_anchor=False,
        error_flow_margin=False,
        instance_supcon=False,
    ),
    "r1": C2RecoverySpec(
        name="S8-P5-C2-R1-P3ZeroGatedAnchor",
        p3_anchor=True,
        error_flow_margin=False,
        instance_supcon=False,
    ),
    "r2": C2RecoverySpec(
        name="S8-P5-C2-R2-ErrorFlowMargin",
        p3_anchor=True,
        error_flow_margin=True,
        instance_supcon=False,
    ),
    "r3": C2RecoverySpec(
        name="S8-P5-C2-R3-InstanceSupCon",
        p3_anchor=True,
        error_flow_margin=True,
        instance_supcon=True,
    ),
}


def get_c2_recovery_spec(variant: str) -> C2RecoverySpec:
    key = str(variant).lower()
    if key not in C2_RECOVERY_VARIANTS:
        raise ValueError(
            f"unknown C2 recovery variant {variant!r}; "
            f"choose {sorted(C2_RECOVERY_VARIANTS)}"
        )
    return C2_RECOVERY_VARIANTS[key]


class C2RecoveryHeads(nn.Module):
    """Small trainable recovery state kept outside the frozen P3 teacher."""

    def __init__(self, feature_channels: int = 256, projection_dim: int = 128) -> None:
        super().__init__()
        self.damage_residual_gate = nn.Parameter(torch.zeros(()))
        self.instance_projector = nn.Sequential(
            nn.Linear(feature_channels, projection_dim, bias=False),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
        )

    @property
    def gate_value(self) -> torch.Tensor:
        return torch.tanh(self.damage_residual_gate)

    def anchor_outputs(
        self,
        c2_location: torch.Tensor,
        c2_damage: torch.Tensor,
        p3_location: torch.Tensor,
        p3_damage: torch.Tensor,
        enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not enabled:
            return c2_location, c2_damage
        # P3 stays an exact, detached anchor.  C2 supplies only the residual.
        p3_location = p3_location.detach()
        p3_damage = p3_damage.detach()
        damage = p3_damage + self.gate_value * (c2_damage - p3_damage)
        return p3_location, damage


def error_flow_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """Penalize damaged/destroyed logits that fall below intact logits.

    BRIGHT label order is background=0, intact=1, damaged=2, destroyed=3.
    Only ground-truth damaged and destroyed pixels participate.
    """

    if logits.ndim != 4 or int(logits.shape[1]) != 4:
        raise ValueError("damage logits must be [B,4,H,W]")
    if labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("damage labels must match logits spatial dimensions")
    valid = labels.ne(int(ignore_index))
    damaged = valid & labels.eq(2)
    destroyed = valid & labels.eq(3)
    intact_logit = logits[:, 1]
    damaged_term = F.relu(float(margin) - logits[:, 2] + intact_logit)
    destroyed_term = F.relu(float(margin) - logits[:, 3] + intact_logit)
    zero = logits.float().sum() * 0.0
    loss_damaged = damaged_term[damaged].mean() if damaged.any() else zero
    loss_destroyed = destroyed_term[destroyed].mean() if destroyed.any() else zero
    active = int(damaged.any()) + int(destroyed.any())
    combined = (
        (loss_damaged + loss_destroyed) / float(active) if active > 0 else zero
    )
    return {
        "loss": combined,
        "loss_damaged_to_intact": loss_damaged,
        "loss_destroyed_to_intact": loss_destroyed,
        "damaged_pixel_count": damaged.sum().detach(),
        "destroyed_pixel_count": destroyed.sum().detach(),
    }


def pool_building_embeddings(
    fused_feature: torch.Tensor,
    bag_specs: Sequence[BuildingBagSpec],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average-pool one differentiable feature vector per labelled building."""

    if fused_feature.ndim != 4:
        raise ValueError("fused_feature must be [B,C,H,W]")
    embeddings = []
    labels = []
    height, width = fused_feature.shape[-2:]
    for spec in bag_specs:
        label = int(spec.label)
        batch_index = int(spec.batch_index)
        if label not in (1, 2, 3):
            continue
        if batch_index < 0 or batch_index >= int(fused_feature.shape[0]):
            continue
        mask = spec.mask.to(
            device=fused_feature.device, dtype=fused_feature.dtype
        )
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[None]
        else:
            raise ValueError("building mask must be 2-D or 3-D")
        mask = F.interpolate(mask, size=(height, width), mode="nearest")[0, 0]
        denominator = mask.sum()
        if float(denominator.detach().item()) <= 0.0:
            continue
        feature = fused_feature[batch_index]
        embeddings.append((feature * mask[None]).sum(dim=(1, 2)) / denominator)
        labels.append(label - 1)
    if not embeddings:
        return (
            fused_feature.new_empty((0, int(fused_feature.shape[1]))),
            torch.empty((0,), device=fused_feature.device, dtype=torch.long),
        )
    return torch.stack(embeddings), torch.tensor(
        labels, device=fused_feature.device, dtype=torch.long
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Single-view supervised contrastive loss with safe missing-class handling."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N,D]")
    if labels.ndim != 1 or int(labels.shape[0]) != int(embeddings.shape[0]):
        raise ValueError("labels must be [N]")
    zero = embeddings.float().sum() * 0.0
    count = int(embeddings.shape[0])
    if count < 2:
        return {"loss": zero, "anchor_count": 0, "embedding_count": count}

    features = F.normalize(embeddings.float(), dim=1)
    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(count, device=embeddings.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid_anchor = positive.any(dim=1)
    if not valid_anchor.any():
        return {"loss": zero, "anchor_count": 0, "embedding_count": count}
    exp_logits = torch.exp(logits) * (~identity).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (
        (positive.to(log_prob.dtype) * log_prob).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )
    loss = -mean_positive[valid_anchor].mean()
    return {
        "loss": loss,
        "anchor_count": int(valid_anchor.sum().item()),
        "embedding_count": count,
    }


def c2_recovery_losses(
    heads: C2RecoveryHeads,
    spec: C2RecoverySpec,
    output_damage: torch.Tensor,
    damage_labels: torch.Tensor,
    fused_feature: torch.Tensor,
    bag_specs: Sequence[BuildingBagSpec],
    margin: float,
    temperature: float,
    ignore_index: int,
) -> Mapping[str, object]:
    zero = output_damage.float().sum() * 0.0
    if spec.error_flow_margin:
        error_flow = error_flow_margin_loss(
            output_damage, damage_labels, margin, ignore_index
        )
    else:
        error_flow = {
            "loss": zero,
            "loss_damaged_to_intact": zero,
            "loss_destroyed_to_intact": zero,
            "damaged_pixel_count": torch.zeros(
                (), device=output_damage.device, dtype=torch.long
            ),
            "destroyed_pixel_count": torch.zeros(
                (), device=output_damage.device, dtype=torch.long
            ),
        }
    if spec.instance_supcon:
        embeddings, labels = pool_building_embeddings(fused_feature, bag_specs)
        projected = heads.instance_projector(embeddings)
        supcon = supervised_contrastive_loss(projected, labels, temperature)
    else:
        supcon = {"loss": zero, "anchor_count": 0, "embedding_count": 0}
    return {"error_flow": error_flow, "supcon": supcon}


__all__ = [
    "C2_RECOVERY_VARIANTS",
    "C2RecoveryHeads",
    "C2RecoverySpec",
    "c2_recovery_losses",
    "error_flow_margin_loss",
    "get_c2_recovery_spec",
    "pool_building_embeddings",
    "supervised_contrastive_loss",
]
