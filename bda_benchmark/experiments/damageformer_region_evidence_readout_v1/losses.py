from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def rescue_gate_loss(
    gate_logits: torch.Tensor,
    parent_probability: torch.Tensor,
    building_ids: torch.Tensor,
    damage_target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    size = gate_logits.shape[-2:]
    if damage_target.ndim == 4:
        damage_target = damage_target[:, 0]
    target = F.interpolate(damage_target[:, None].float(), size=size, mode="nearest")[:, 0].long()
    base_pred = parent_probability.argmax(dim=1)
    candidate = (
        (building_ids > 0)
        & (base_pred == 1)
        & (target != ignore_index)
        & ((target == 1) | (target == 2))
    )
    if not bool(candidate.any().item()):
        return gate_logits.float().sum() * 0.0
    labels = (target[candidate] == 2).float()
    predictions = gate_logits[:, 0][candidate].float()
    positive = labels.sum()
    negative = labels.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(predictions, labels, pos_weight=pos_weight)


def dense_preservation_kl(
    new_logits: torch.Tensor,
    base_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-pixel KD; unlike batchmean this does not scale with H*W."""
    t = float(temperature)
    teacher = torch.softmax(base_logits.detach().float() / t, dim=1)
    student = torch.log_softmax(new_logits.float() / t, dim=1)
    return F.kl_div(student, teacher, reduction="none").sum(dim=1).mean() * (t * t)


def readout_losses(
    variant: str,
    aux: Mapping[str, torch.Tensor],
    damage_target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    reference = aux["damage_logits"]
    zero = reference.float().sum() * 0.0
    gate = zero
    preserve = zero
    residual = zero
    if variant in ("w2_id_logit", "w3_id_mass"):
        gate = rescue_gate_loss(
            aux["id_gate_logits"],
            aux["parent_damage_probability"],
            aux["building_ids"],
            damage_target,
            ignore_index,
        )
    if variant == "w2_id_logit":
        residual = aux["id_logit_delta"].float().abs().mean()
    elif variant == "w3_id_mass":
        residual = aux["id_expert_residual"].float().abs().mean()
    elif variant in ("w0_full_residual", "w5_logit4_kd"):
        residual = aux["delta4"].float().abs().mean()
        if variant == "w5_logit4_kd":
            preserve = dense_preservation_kl(
                aux["damage_logits"], aux["parent_damage_logits_feature"]
            )
    return {
        "gate": gate,
        "preserve": preserve,
        "residual": residual,
        "auxiliary": 0.2 * gate + 0.2 * preserve + 0.001 * residual,
    }


__all__ = ["dense_preservation_kl", "readout_losses", "rescue_gate_loss"]
