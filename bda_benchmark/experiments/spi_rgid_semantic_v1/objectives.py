from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class JointLossWeights:
    semantic: float = 1.0
    geometry: float = 0.05
    region: float = 0.05
    center_equivariance: float = 0.02
    boundary_equivariance: float = 0.01
    anchor: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("joint-loss weights must be nonnegative")


@dataclass(frozen=True)
class GeometryLossWeights:
    center: float = 1.0
    offset: float = 0.0
    boundary: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("geometry-loss weights must be nonnegative")


def compose_region_quality(
    *, shape: float, center: float, boundary: float, semantic: float
) -> float:
    """q_j = q_shape q_center q_boundary q_semantic, all prediction-only."""

    factors = (float(shape), float(center), float(boundary), float(semantic))
    if any(value < 0.0 or value > 1.0 for value in factors):
        raise ValueError("all region-quality factors must be in [0,1]")
    quality = 1.0
    for value in factors:
        quality *= value
    return quality


def region_quality_tier(
    quality: float, *, reliable_threshold: float = 0.50, ambiguous_threshold: float = 0.20
) -> str:
    if not 0.0 <= ambiguous_threshold <= reliable_threshold <= 1.0:
        raise ValueError("quality thresholds must satisfy 0 <= ambiguous <= reliable <= 1")
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must be in [0,1]")
    if quality >= reliable_threshold:
        return "reliable"
    if quality >= ambiguous_threshold:
        return "ambiguous"
    return "ignored"


def quality_weighted_geometry_loss(
    *,
    center_losses: Sequence[torch.Tensor],
    offset_losses: Sequence[torch.Tensor],
    boundary_losses: Sequence[torch.Tensor],
    qualities: Sequence[float],
    weights: GeometryLossWeights = GeometryLossWeights(),
) -> torch.Tensor:
    """Instance-equal geometry loss; each input loss must already be a scalar."""

    count = len(qualities)
    if not (
        len(center_losses) == len(offset_losses) == len(boundary_losses) == count
    ):
        raise ValueError("geometry losses and qualities must align")
    if count == 0:
        raise ValueError("at least one geometry loss is required")
    reference = center_losses[0]
    numerator = reference.sum() * 0.0
    denominator = 0.0
    for center, offset, boundary, quality in zip(
        center_losses, offset_losses, boundary_losses, qualities
    ):
        quality = float(quality)
        if not 0.0 <= quality <= 1.0:
            raise ValueError("geometry quality must be in [0,1]")
        for value in (center, offset, boundary):
            if value.numel() != 1:
                raise ValueError("per-region geometry losses must be scalar")
        if quality == 0.0:
            continue
        combined = (
            weights.center * center
            + weights.offset * offset
            + weights.boundary * boundary
        )
        numerator = numerator + quality * combined
        denominator += quality
    if denominator == 0.0:
        return reference.sum() * 0.0
    return numerator / denominator


def aligned_equivariance_loss(
    reference_logits: torch.Tensor,
    augmented_logits_aligned_to_reference: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consistency after callers invert the geometric augmentation.

    Keeping inverse warping outside this function prevents silent mistakes in
    offset-vector signs. Center/boundary logits may use this function directly;
    vector offsets require a vector-aware inverse transform before the call.
    """

    if reference_logits.shape != augmented_logits_aligned_to_reference.shape:
        raise ValueError("equivariance tensors must have identical shapes")
    difference = (
        torch.sigmoid(reference_logits.float())
        - torch.sigmoid(augmented_logits_aligned_to_reference.float())
    ).abs()
    if valid_mask is None:
        return difference.mean()
    mask = valid_mask.bool()
    while mask.ndim < difference.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(difference)
    if not mask.any():
        return difference.sum() * 0.0
    return difference[mask].mean()


def quality_weighted_region_loss(
    semantic_logits: torch.Tensor,
    region_masks: Sequence[torch.Tensor],
    region_labels: Sequence[int],
    qualities: Sequence[float],
) -> torch.Tensor:
    """Soft region-level supervision without broadcasting a hard pixel label."""

    if semantic_logits.ndim != 3:
        raise ValueError("semantic_logits must be [C,H,W]")
    if semantic_logits.shape[0] != 4:
        raise ValueError("four semantic classes are required")
    if not (len(region_masks) == len(region_labels) == len(qualities)):
        raise ValueError("region masks, labels and qualities must align")
    probabilities = semantic_logits.float().softmax(dim=0)
    numerator = semantic_logits.sum() * 0.0
    denominator = 0.0
    for mask, label, quality in zip(region_masks, region_labels, qualities):
        label = int(label)
        quality = float(quality)
        if label not in (1, 2, 3):
            raise ValueError("region labels must be Intact/Damaged/Destroyed")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("region quality must be in [0,1]")
        selected = torch.as_tensor(mask, device=semantic_logits.device).bool()
        if selected.shape != semantic_logits.shape[-2:]:
            raise ValueError("region mask and logits have different shapes")
        if quality == 0.0 or not selected.any():
            continue
        mean_probability = probabilities[label, selected].mean().clamp_min(1e-8)
        numerator = numerator - quality * mean_probability.log()
        denominator += quality
    if denominator == 0.0:
        return semantic_logits.sum() * 0.0
    return numerator / denominator


def anchor_kl_loss(
    student_logits: torch.Tensor,
    frozen_teacher_logits: torch.Tensor,
    unsafe_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve P3 decisions where pseudo geometry is not trusted."""

    if student_logits.shape != frozen_teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 4:
        raise ValueError("student logits must be [B,C,H,W]")
    mask = unsafe_mask.bool()
    if mask.shape != (student_logits.shape[0], *student_logits.shape[-2:]):
        raise ValueError("unsafe_mask must be [B,H,W]")
    if not mask.any():
        return student_logits.sum() * 0.0
    teacher = frozen_teacher_logits.detach().float().softmax(dim=1)
    log_student = student_logits.float().log_softmax(dim=1)
    per_pixel = F.kl_div(log_student, teacher, reduction="none").sum(dim=1)
    return per_pixel[mask].mean()


class PseudoInstanceJointObjective(nn.Module):
    """Minimal safe objective for the post-C3 raw-semantic experiment."""

    def __init__(
        self,
        weights: JointLossWeights = JointLossWeights(),
        geometry_weights: GeometryLossWeights = GeometryLossWeights(),
    ) -> None:
        super().__init__()
        self.weights = weights
        self.geometry_weights = geometry_weights

    def forward(
        self,
        *,
        semantic_logits: torch.Tensor,
        semantic_target: torch.Tensor,
        teacher_logits: torch.Tensor,
        unsafe_mask: torch.Tensor,
        region_masks: Sequence[Sequence[torch.Tensor]],
        region_labels: Sequence[Sequence[int]],
        region_qualities: Sequence[Sequence[float]],
        center_logits_view1: torch.Tensor | None = None,
        center_logits_view2_aligned: torch.Tensor | None = None,
        boundary_logits_view1: torch.Tensor | None = None,
        boundary_logits_view2_aligned: torch.Tensor | None = None,
        center_region_losses: Sequence[torch.Tensor] | None = None,
        offset_region_losses: Sequence[torch.Tensor] | None = None,
        boundary_region_losses: Sequence[torch.Tensor] | None = None,
        geometry_qualities: Sequence[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        if semantic_logits.ndim != 4 or semantic_logits.shape[1] != 4:
            raise ValueError("semantic_logits must be [B,4,H,W]")
        if semantic_target.shape != (
            semantic_logits.shape[0],
            *semantic_logits.shape[-2:],
        ):
            raise ValueError("semantic_target must be [B,H,W]")
        batch = semantic_logits.shape[0]
        if not (
            len(region_masks)
            == len(region_labels)
            == len(region_qualities)
            == batch
        ):
            raise ValueError("region supervision must contain one entry per batch row")

        semantic = F.cross_entropy(
            semantic_logits.float(), semantic_target.long(), ignore_index=255
        )
        rows = [
            quality_weighted_region_loss(
                semantic_logits[index],
                region_masks[index],
                region_labels[index],
                region_qualities[index],
            )
            for index in range(batch)
        ]
        region = torch.stack(rows).mean()
        anchor = anchor_kl_loss(semantic_logits, teacher_logits, unsafe_mask)

        zero = semantic_logits.sum() * 0.0
        center_eq = zero
        if center_logits_view1 is not None or center_logits_view2_aligned is not None:
            if center_logits_view1 is None or center_logits_view2_aligned is None:
                raise ValueError("both center views are required")
            center_eq = aligned_equivariance_loss(
                center_logits_view1, center_logits_view2_aligned
            )
        boundary_eq = zero
        if boundary_logits_view1 is not None or boundary_logits_view2_aligned is not None:
            if boundary_logits_view1 is None or boundary_logits_view2_aligned is None:
                raise ValueError("both boundary views are required")
            boundary_eq = aligned_equivariance_loss(
                boundary_logits_view1, boundary_logits_view2_aligned
            )

        geometry_inputs = (
            center_region_losses,
            offset_region_losses,
            boundary_region_losses,
            geometry_qualities,
        )
        geometry = zero
        if any(value is not None for value in geometry_inputs):
            if any(value is None for value in geometry_inputs):
                raise ValueError("all per-region geometry inputs are required together")
            geometry = quality_weighted_geometry_loss(
                center_losses=center_region_losses,
                offset_losses=offset_region_losses,
                boundary_losses=boundary_region_losses,
                qualities=geometry_qualities,
                weights=self.geometry_weights,
            )

        total = (
            self.weights.semantic * semantic
            + self.weights.geometry * geometry
            + self.weights.region * region
            + self.weights.center_equivariance * center_eq
            + self.weights.boundary_equivariance * boundary_eq
            + self.weights.anchor * anchor
        )
        return {
            "loss": total,
            "semantic_loss": semantic,
            "geometry_loss": geometry,
            "region_loss": region,
            "center_equivariance_loss": center_eq,
            "boundary_equivariance_loss": boundary_eq,
            "anchor_loss": anchor,
        }


__all__ = [
    "GeometryLossWeights",
    "JointLossWeights",
    "PseudoInstanceJointObjective",
    "aligned_equivariance_loss",
    "anchor_kl_loss",
    "compose_region_quality",
    "quality_weighted_geometry_loss",
    "quality_weighted_region_loss",
    "region_quality_tier",
]
