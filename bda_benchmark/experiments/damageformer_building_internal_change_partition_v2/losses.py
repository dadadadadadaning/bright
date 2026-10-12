"""Stage-wise losses for Building-Internal Change-Aware Partition V2.

Important: no pixel-level local damage mask is assumed.  GT damage labels are
used only to derive weak labels for predicted building bags during Stage-1.
"""
from __future__ import annotations

from math import log
from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from .association import normalized_coordinate_grid, soft_pool_mean_std, soft_unpool_tokens


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _building_mask(aux: Mapping[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    return (aux["building_ids"] > 0).unsqueeze(1).to(dtype=dtype)


def _reconstruct_mean(value: torch.Tensor, aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    token_count = int(aux["token_count"].item())
    pooled = soft_pool_mean_std(
        value,
        aux["association_probabilities"],
        aux["candidate_ids"],
        token_count,
    )
    return pooled["reconstruction"]


def compactness_loss(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    reference = aux["rgb_structure"]
    b, _, h, w = reference.shape
    coords = normalized_coordinate_grid(b, h, w, reference)
    reconstructed = _reconstruct_mean(coords, aux)
    mask = _building_mask(aux, reference.dtype)
    return ((reconstructed - coords).square() * mask).sum() / (
        mask.sum().clamp_min(1.0) * coords.shape[1]
    )


def structure_reconstruction_loss(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    structure = aux["rgb_structure"].detach()
    reconstructed = _reconstruct_mean(structure, aux)
    mask = _building_mask(aux, structure.dtype)
    return ((reconstructed - structure).square() * mask).sum() / (
        mask.sum().clamp_min(1.0) * structure.shape[1]
    )


def building_local_balance_loss(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mass = aux["region_mass"].float()
    target = aux["token_target_mass"].float()
    owners = aux["token_building_ids"]
    valid = (owners > 0) & (target > 0)
    if not bool(valid.any().item()):
        return _zero(mass)
    ratio = mass[valid] / target[valid].clamp_min(1.0e-6)
    return (ratio - 1.0).square().mean()


def evidence_preservation_loss(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Weakly preserve the learned local evidence magnitude inside regions.

    This is intentionally weak and does NOT equate RGB-SAR difference with GT
    damage. It only discourages completely washing out local evidence.
    """
    score = torch.sigmoid(aux["evidence_logit"].detach())
    reconstructed = _reconstruct_mean(score, aux)
    mask = _building_mask(aux, score.dtype)
    return ((reconstructed - score).square() * mask).sum() / mask.sum().clamp_min(1.0)


def change_residual_regularizer(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    alpha = aux["change_alpha"].float()
    delta = aux["change_delta_logits"].float()
    mask = _building_mask(aux, alpha.dtype)
    value = (alpha * delta).square().mean(dim=1, keepdim=True)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _bag_logits_logmeanexp(region_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    k = max(1, int(region_logits.shape[0]))
    tau = max(float(temperature), 1.0e-3)
    return tau * (torch.logsumexp(region_logits / tau, dim=0) - log(float(k)))


def _bag_logits_topk(region_logits: torch.Tensor, topk: int) -> torch.Tensor:
    k = max(1, min(int(topk), int(region_logits.shape[0])))
    return region_logits.topk(k=k, dim=0).values.mean(dim=0)


def weak_building_bag_loss(
    aux: Mapping[str, torch.Tensor],
    damage_target: torch.Tensor,
    *,
    ignore_index: int = 255,
    label_purity_threshold: float = 0.70,
    aggregator: str = "logmeanexp",
    bag_temperature: float = 0.50,
    bag_topk: int = 3,
    min_region_mass: float = 0.50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weak supervision from GT building class to predicted-building region bags.

    Forward partition uses predicted buildings only.  GT enters here only to
    assign an Intact/Damaged/Destroyed label to a predicted building bag by
    majority vote over valid non-background semantic damage pixels.  Low-purity
    bags are skipped rather than forcing noisy supervision.
    """
    region_logits = aux["region_logits"]
    owners = aux["token_building_ids"]
    mass = aux["region_mass"]
    building_ids = aux["building_ids"]
    h, w = building_ids.shape[-2:]
    if damage_target.ndim == 4 and damage_target.shape[1] == 1:
        damage_target = damage_target[:, 0]
    resized = F.interpolate(
        damage_target.unsqueeze(1).float(), size=(h, w), mode="nearest"
    )[:, 0].long()

    losses = []
    correct = 0
    used = 0
    skipped_purity = 0
    skipped_regions = 0
    class_counts = [0, 0, 0]

    for bi in range(building_ids.shape[0]):
        ids = torch.unique(building_ids[bi])
        ids = ids[ids > 0]
        for building_id in ids.tolist():
            pixel_mask = building_ids[bi] == int(building_id)
            labels = resized[bi][pixel_mask]
            valid_labels = labels[(labels != ignore_index) & (labels >= 1) & (labels <= 3)]
            if valid_labels.numel() == 0:
                skipped_purity += 1
                continue
            counts = torch.stack([(valid_labels == cls).sum() for cls in (1, 2, 3)]).float()
            dominant = int(counts.argmax().item())
            purity = float(counts[dominant] / counts.sum().clamp_min(1.0))
            if purity < float(label_purity_threshold):
                skipped_purity += 1
                continue

            token_mask = (owners[bi] == int(building_id)) & (mass[bi] >= float(min_region_mass))
            token_idx = torch.nonzero(token_mask, as_tuple=False).flatten()
            if token_idx.numel() == 0:
                skipped_regions += 1
                continue
            logits = region_logits[bi, token_idx]
            if aggregator == "logmeanexp":
                bag_logits = _bag_logits_logmeanexp(logits, bag_temperature)
            elif aggregator == "topk":
                bag_logits = _bag_logits_topk(logits, bag_topk)
            else:
                raise ValueError(f"unknown bag aggregator: {aggregator}")
            target = torch.tensor([dominant], device=bag_logits.device, dtype=torch.long)
            losses.append(F.cross_entropy(bag_logits.unsqueeze(0).float(), target))
            correct += int(bag_logits.argmax().item() == dominant)
            used += 1
            class_counts[dominant] += 1

    if not losses:
        loss = _zero(region_logits)
    else:
        loss = torch.stack(losses).mean()
    stats = {
        "bag_count": float(used),
        "bag_accuracy": float(correct / max(1, used)),
        "bag_skipped_purity": float(skipped_purity),
        "bag_skipped_regions": float(skipped_regions),
        "bag_intact": float(class_counts[0]),
        "bag_damaged": float(class_counts[1]),
        "bag_destroyed": float(class_counts[2]),
    }
    return loss, stats


def feedback_regularizer(aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    gate = aux["region_residual_gate"].float()
    update = aux["region_update"].float()
    # Conservative: prefer small spatial coverage and small residual energy
    # unless pixel damage loss demonstrates utility.
    return gate.mean() + 0.1 * (gate * update.square().mean(dim=1, keepdim=True)).mean()


def stage_losses(
    aux: Mapping[str, torch.Tensor],
    damage_target: torch.Tensor,
    *,
    stage: str,
    use_bag_supervision: bool,
    ignore_index: int = 255,
    bag_aggregator: str = "logmeanexp",
    bag_temperature: float = 0.50,
    bag_topk: int = 3,
    bag_label_purity: float = 0.70,
) -> Dict[str, torch.Tensor | float]:
    reference = aux["feature_roi"]
    zero = _zero(reference)
    result: Dict[str, torch.Tensor | float] = {
        "compact": zero,
        "structure": zero,
        "balance": zero,
        "evidence": zero,
        "bag": zero,
        "change_reg": zero,
        "feedback_reg": zero,
        "bag_count": 0.0,
        "bag_accuracy": 0.0,
    }

    if stage in ("structure", "change"):
        result["compact"] = compactness_loss(aux)
        result["structure"] = structure_reconstruction_loss(aux)
        result["balance"] = building_local_balance_loss(aux)

    if stage == "change":
        result["evidence"] = evidence_preservation_loss(aux)
        result["change_reg"] = change_residual_regularizer(aux)
        if use_bag_supervision:
            bag, bag_stats = weak_building_bag_loss(
                aux,
                damage_target,
                ignore_index=ignore_index,
                label_purity_threshold=bag_label_purity,
                aggregator=bag_aggregator,
                bag_temperature=bag_temperature,
                bag_topk=bag_topk,
            )
            result["bag"] = bag
            result.update(bag_stats)

    if stage == "feedback":
        result["feedback_reg"] = feedback_regularizer(aux)

    if stage == "structure":
        total = (
            0.05 * result["compact"]
            + 0.10 * result["structure"]
            + 0.01 * result["balance"]
        )
    elif stage == "change":
        total = (
            0.05 * result["compact"]
            + 0.05 * result["structure"]
            + 0.005 * result["balance"]
            + 0.02 * result["evidence"]
            + 0.01 * result["change_reg"]
            + (1.0 * result["bag"] if use_bag_supervision else zero)
        )
    elif stage == "feedback":
        total = 0.001 * result["feedback_reg"]
    else:
        raise ValueError(f"unknown stage: {stage}")
    result["total"] = total
    return result


__all__ = [
    "compactness_loss",
    "structure_reconstruction_loss",
    "building_local_balance_loss",
    "evidence_preservation_loss",
    "change_residual_regularizer",
    "weak_building_bag_loss",
    "feedback_regularizer",
    "stage_losses",
]
