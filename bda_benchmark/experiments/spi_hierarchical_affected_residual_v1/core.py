from __future__ import annotations

import torch
from scipy import ndimage


AO_OUTCOME_NEUTRAL = 0
AO_OUTCOME_CORRECTED = 1
AO_OUTCOME_HARMED = 2


def severity_choice(probabilities: torch.Tensor) -> torch.Tensor:
    """Return the unchanged P(Damaged) vs P(Destroyed) preference."""

    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("probabilities must be [4,H,W]")
    return probabilities[2:4].argmax(0) + 2


def affected_reachable_mask(
    fixed_probabilities: torch.Tensor,
    building_mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """GT-only AO-E0 mask reachable by an equal affected-logit residual.

    A pixel is reachable only when it is inside the frozen P3 building mask,
    fixed S2 currently predicts Background/Intact, and the frozen severity
    preference already agrees with the true Damaged/Destroyed class.
    """

    prediction = fixed_probabilities.argmax(0)
    severity = severity_choice(fixed_probabilities)
    valid = (target >= 0) & (target <= 3)
    return (
        building_mask.bool() & valid & (prediction < 2)
        & (target >= 2) & (severity == target)
    )


def apply_affected_oracle(
    fixed_probabilities: torch.Tensor,
    building_mask: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-S2 labels with only reachable correct promotions applied."""

    prediction = fixed_probabilities.argmax(0)
    reachable = affected_reachable_mask(fixed_probabilities, building_mask, target)
    result = prediction.clone()
    severity = severity_choice(fixed_probabilities)
    result[reachable] = severity[reachable]
    return result, reachable


def equal_affected_residual(
    logits: torch.Tensor,
    residual: torch.Tensor,
    building_mask: torch.Tensor,
) -> torch.Tensor:
    """Add exactly the same residual to Damaged and Destroyed logits."""

    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("logits must be [N,4,H,W]")
    if residual.ndim == 3:
        residual = residual[:, None]
    if building_mask.ndim == 3:
        building_mask = building_mask[:, None]
    if residual.shape[0] != logits.shape[0] or residual.shape[-2:] != logits.shape[-2:]:
        raise ValueError("residual/logit shape mismatch")
    update = residual * building_mask.to(residual.dtype)
    result = logits.clone()
    result[:, 2:4] = result[:, 2:4] + update
    return result


def minimum_affected_residual(probabilities: torch.Tensor) -> torch.Tensor:
    """Smallest equal logit increment that makes an affected class win.

    Probabilities are sufficient because log-softmax differs from logits by a
    pixel-wise constant, which cancels in the difference below.
    """

    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("probabilities must be [4,H,W]")
    logp = probabilities.float().clamp_min(1e-8).log()
    non_affected = logp[:2].amax(0)
    affected = logp[2:4].amax(0)
    return (non_affected - affected).clamp_min(0.0)


def affected_candidate_outcomes(
    fixed_probabilities: torch.Tensor,
    building_mask: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label every deployable AO candidate without using GT as an input.

    Candidate generation uses predictions only.  GT is used solely to assign
    the audit target: corrected, harmed, or neutral.
    """

    prediction = fixed_probabilities.argmax(0)
    severity = severity_choice(fixed_probabilities)
    valid = (target >= 0) & (target <= 3)
    # Candidate generation is deployable: it depends on predictions and the
    # predicted building mask only.  Invalid GT pixels remain candidates but
    # receive the neutral audit label and are excluded from supervised scores.
    candidate = building_mask.bool() & (prediction < 2)
    outcome = torch.full_like(target, AO_OUTCOME_NEUTRAL, dtype=torch.uint8)
    corrected = candidate & (severity == target) & (prediction != target)
    harmed = candidate & (prediction == target) & (severity != target)
    outcome[corrected] = AO_OUTCOME_CORRECTED
    outcome[harmed] = AO_OUTCOME_HARMED
    return candidate, outcome


def building_core_and_ring(region_labels: torch.Tensor, radius: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic erosion-based core/ring diagnostic."""

    labels = region_labels.detach().cpu().numpy()
    core = torch.zeros_like(region_labels, dtype=torch.bool)
    structure = ndimage.generate_binary_structure(2, 1)
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        if not mask.any():
            continue
        eroded = ndimage.binary_erosion(mask, structure=structure, iterations=radius)
        core |= torch.from_numpy(eroded)
    building = region_labels > 0
    return core, building & ~core


def deterministic_subsample(mask: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    """Prediction-only sampling mask used to bound AO-E1 cache size."""

    flat = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    if maximum <= 0 or len(flat) <= maximum:
        return mask.bool()
    generator = torch.Generator().manual_seed(int(seed))
    chosen = flat[torch.randperm(len(flat), generator=generator)[:maximum]]
    result = torch.zeros(mask.numel(), dtype=torch.bool)
    result[chosen] = True
    return result.reshape(mask.shape)


__all__ = [
    "AO_OUTCOME_CORRECTED", "AO_OUTCOME_HARMED", "AO_OUTCOME_NEUTRAL",
    "affected_candidate_outcomes", "affected_reachable_mask",
    "apply_affected_oracle", "building_core_and_ring",
    "deterministic_subsample", "equal_affected_residual",
    "minimum_affected_residual", "severity_choice",
]
