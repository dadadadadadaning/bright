"""Auxiliary losses for region formation; official pixel labels remain unchanged."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .association import (
    hard_pool_unpool,
    normalized_coordinate_grid,
    reconstruct_scalar_soft,
    soft_pool_unpool,
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _reconstruct(value: torch.Tensor, aux: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "hard_region_ids" in aux:
        return hard_pool_unpool(value, aux["hard_region_ids"])["reconstruction"]
    probabilities = aux["association_probabilities"]
    ids = aux["candidate_ids"]
    token_count = int(aux["token_count"].item())
    return soft_pool_unpool(value, probabilities, ids, token_count)["reconstruction"]


def boundary_affinity_loss(
    probabilities: torch.Tensor,
    building_target: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Keep neighbouring assignments similar except across building boundaries."""
    b, _, n = probabilities.shape
    h, w = building_target.shape[-2:]
    probs = probabilities.reshape(b, 9, h, w)
    target = building_target.float()
    losses = []
    for dim in (-1, -2):
        if dim == -1:
            left, right = probs[..., :-1], probs[..., 1:]
            t_left, t_right = target[..., :-1], target[..., 1:]
        else:
            left, right = probs[..., :-1, :], probs[..., 1:, :]
            t_left, t_right = target[..., :-1, :], target[..., 1:, :]
        similarity = F.cosine_similarity(left, right, dim=1).unsqueeze(1)
        same = (t_left == t_right).float()
        losses.append((same * (1.0 - similarity) + (1.0 - same) * F.relu(similarity - margin)).mean())
    return torch.stack(losses).mean()


def region_losses(
    aux: Mapping[str, torch.Tensor],
    location_target: torch.Tensor,
    use_boundary_loss: bool,
) -> Dict[str, torch.Tensor]:
    reference = aux["feature_roi"]
    zero = _zero(reference)
    result = {
        "spatial": zero,
        "building": zero,
        "evidence": zero,
        "balance": zero,
        "boundary": zero,
    }
    if "region_content" not in aux:
        result["total"] = zero
        return result

    size = aux["region_content"].shape[-2:]
    if location_target.ndim == 3:
        location_target = location_target.unsqueeze(1)
    valid = (location_target != 255).float()
    building = ((location_target > 0) & (location_target != 255)).float()
    building = F.interpolate(building, size=size, mode="nearest")
    valid = F.interpolate(valid, size=size, mode="nearest")

    coordinates = normalized_coordinate_grid(reference.shape[0], *size, reference)
    reconstructed_coordinates = _reconstruct(coordinates, aux)
    result["spatial"] = ((reconstructed_coordinates - coordinates).square() * valid).sum() / valid.sum().clamp_min(1.0)

    # ``binary_cross_entropy`` intentionally raises inside CUDA autocast.
    # This branch already receives probabilities (not logits), so evaluate the
    # equivalent Bernoulli negative log-likelihood explicitly in FP32 rather
    # than converting pooled probabilities back to artificial logits.
    reconstructed_building = _reconstruct(building, aux).float().clamp(1.0e-5, 1.0 - 1.0e-5)
    building_fp32 = building.float()
    valid_fp32 = valid.float()
    building_nll = -(
        building_fp32 * reconstructed_building.log()
        + (1.0 - building_fp32) * torch.log1p(-reconstructed_building)
    )
    result["building"] = (building_nll * valid_fp32).sum() / valid_fp32.sum().clamp_min(1.0)

    evidence = aux["region_relation"].detach().square().mean(dim=1, keepdim=True).sqrt()
    reconstructed_evidence = _reconstruct(evidence, aux)
    evidence_weight = building * aux["region_confidence"].detach()
    result["evidence"] = ((reconstructed_evidence - evidence).square() * evidence_weight).sum() / evidence_weight.sum().clamp_min(1.0)

    mass = aux["region_mass"]
    positive = mass > 1.0e-5
    if bool(positive.any().item()):
        selected = mass[positive]
        result["balance"] = ((selected / selected.mean().clamp_min(1.0e-5) - 1.0).square()).mean()

    if use_boundary_loss and "association_probabilities" in aux:
        result["boundary"] = boundary_affinity_loss(
            aux["association_probabilities"], building
        )
    result["total"] = (
        0.05 * result["spatial"]
        + 0.10 * result["building"]
        + 0.05 * result["evidence"]
        + 0.002 * result["balance"]
        + 0.02 * result["boundary"]
    )
    return result
