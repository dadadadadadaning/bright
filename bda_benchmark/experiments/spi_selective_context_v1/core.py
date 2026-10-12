from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    gate_region,
    region_evidence,
)


VARIANTS = (
    "s0_raw_p3",
    "s1_loccc_hard_a6",
    "s2_loccc_soft_context",
    "s3_loccc_uncertain_interior",
    "s4_local_uncertain_control",
)


@dataclass(frozen=True)
class SelectiveContextConfig:
    minimum_purity: float = 0.70
    minimum_margin: float = 0.10
    blend_strength: float = 0.75
    uncertainty_threshold: float = 0.50
    uncertainty_temperature: float = 0.10
    interior_radius: int = 2
    boundary_floor: float = 0.20
    local_kernel_size: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_purity <= 1.0:
            raise ValueError("minimum_purity must be in [0,1]")
        if self.minimum_margin < 0.0:
            raise ValueError("minimum_margin must be nonnegative")
        if not 0.0 <= self.blend_strength <= 1.0:
            raise ValueError("blend_strength must be in [0,1]")
        if not 0.0 < self.uncertainty_temperature:
            raise ValueError("uncertainty_temperature must be positive")
        if self.interior_radius < 0:
            raise ValueError("interior_radius must be nonnegative")
        if not 0.0 <= self.boundary_floor <= 1.0:
            raise ValueError("boundary_floor must be in [0,1]")
        if self.local_kernel_size < 1 or self.local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd number")


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Return per-pixel entropy in [0,1] for a [C,H,W] probability tensor."""

    if probabilities.ndim != 3 or probabilities.shape[0] < 2:
        raise ValueError("probabilities must be [C,H,W] with C >= 2")
    probabilities = probabilities.float().clamp_min(1e-8)
    denominator = torch.log(
        torch.tensor(float(probabilities.shape[0]), device=probabilities.device)
    )
    return -(probabilities * probabilities.log()).sum(dim=0) / denominator


def foreground_region_distribution(
    probabilities: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean foreground distribution; background cannot erase predicted support."""

    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("probabilities must be [4,H,W]")
    if mask.shape != probabilities.shape[-2:] or not bool(mask.any().item()):
        raise ValueError("mask must be nonempty and match probabilities")
    scores = probabilities[1:4, mask].mean(dim=1)
    scores = scores / scores.sum().clamp_min(1e-8)
    output = torch.zeros(4, dtype=probabilities.dtype, device=probabilities.device)
    output[1:4] = scores
    return output


def interior_weight(
    mask: torch.Tensor, *, radius: int, boundary_floor: float
) -> torch.Tensor:
    """Cheap erosion-depth proxy: boundary gets floor, stable interior gets one."""

    if mask.ndim != 2:
        raise ValueError("mask must be [H,W]")
    if radius < 0 or not 0.0 <= boundary_floor <= 1.0:
        raise ValueError("invalid interior-weight settings")
    mask = mask.bool()
    output = torch.zeros_like(mask, dtype=torch.float32)
    output[mask] = float(boundary_floor)
    if radius == 0:
        output[mask] = 1.0
        return output
    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        return output
    y0, x0 = coordinates.amin(dim=0).tolist()
    y1, x1 = (coordinates.amax(dim=0) + 1).tolist()
    crop = mask[y0:y1, x0:x1]
    crop_output = torch.zeros_like(crop, dtype=torch.float32)
    crop_output[crop] = float(boundary_floor)
    current = crop.float()[None, None]
    for depth in range(1, radius + 1):
        current = (F.avg_pool2d(current, 3, stride=1, padding=1) == 1.0).float()
        value = boundary_floor + (1.0 - boundary_floor) * depth / radius
        crop_output[current[0, 0].bool()] = float(value)
    output[y0:y1, x0:x1] = crop_output
    return output


def local_foreground_context(
    probabilities: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    kernel_size: int,
) -> torch.Tensor:
    """Matched local context control with the same foreground-only semantics."""

    if probabilities.shape != (4, *support_mask.shape):
        raise ValueError("probabilities and support mask do not align")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    pad = kernel_size // 2
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    support = support_mask.float()[None, None]
    denominator = F.conv2d(support, kernel, padding=pad)[0, 0].clamp_min(1.0)
    output = torch.zeros_like(probabilities)
    for class_id in (1, 2, 3):
        numerator = F.conv2d(
            (probabilities[class_id] * support_mask)[None, None],
            kernel,
            padding=pad,
        )[0, 0]
        output[class_id] = numerator / denominator
    foreground_sum = output[1:4].sum(dim=0, keepdim=True).clamp_min(1e-8)
    output[1:4] /= foreground_sum
    return output


def _blend(
    probabilities: torch.Tensor,
    context: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if context.ndim == 1:
        context = context[:, None, None]
    if context.shape not in (probabilities.shape, (4, 1, 1)):
        raise ValueError("context must be [4] or [4,H,W]")
    alpha = alpha.clamp(0.0, 1.0)[None]
    mixed = (1.0 - alpha) * probabilities + alpha * context
    return mixed / mixed.sum(dim=0, keepdim=True).clamp_min(1e-8)


def selective_context_refinement(
    base_prediction: torch.Tensor,
    probabilities: torch.Tensor,
    regions: Sequence[torch.Tensor],
    *,
    config: SelectiveContextConfig,
    mode: str,
) -> tuple[torch.Tensor, list[dict[str, float | int | bool]]]:
    """Refine with soft region context or a matched local-context control.

    No target is accepted by this function. All decisions are prediction-only.
    """

    if mode not in {"soft_region", "uncertain_interior", "local_control"}:
        raise ValueError(f"unknown refinement mode: {mode}")
    if probabilities.shape != (4, *base_prediction.shape):
        raise ValueError("probabilities must be [4,H,W]")
    output_probabilities = probabilities.clone().float()
    entropy = normalized_entropy(probabilities)
    support = torch.zeros_like(base_prediction, dtype=torch.bool)
    for mask in regions:
        support |= mask.bool()
    local_context = None
    if mode == "local_control":
        local_context = local_foreground_context(
            probabilities,
            support,
            kernel_size=config.local_kernel_size,
        )

    records: list[dict[str, float | int | bool]] = []
    for index, mask in enumerate(regions):
        mask = mask.bool()
        evidence = region_evidence(base_prediction, probabilities, mask)
        accepted = gate_region(
            evidence,
            minimum_purity=config.minimum_purity,
            minimum_margin=config.minimum_margin,
            maximum_area=0,
        )
        alpha_values = torch.zeros(
            int(mask.sum().item()), dtype=entropy.dtype, device=entropy.device
        )
        if accepted:
            alpha_values.fill_(config.blend_strength)
            if mode in {"uncertain_interior", "local_control"}:
                uncertainty = torch.sigmoid(
                    (entropy - config.uncertainty_threshold)
                    / config.uncertainty_temperature
                )
                depth = interior_weight(
                    mask,
                    radius=config.interior_radius,
                    boundary_floor=config.boundary_floor,
                ).to(entropy.device)
                alpha_values *= uncertainty[mask] * depth[mask]
            context = (
                local_context
                if mode == "local_control"
                else foreground_region_distribution(probabilities, mask)
            )
            region_probabilities = output_probabilities[:, mask]
            region_context = (
                context[:, mask] if context.ndim == 3 else context[:, None]
            )
            region_alpha = alpha_values[None].clamp(0.0, 1.0)
            mixed = (
                (1.0 - region_alpha) * region_probabilities
                + region_alpha * region_context
            )
            mixed /= mixed.sum(dim=0, keepdim=True).clamp_min(1e-8)
            output_probabilities[:, mask] = mixed
        changed_pixels = int(
            (
                output_probabilities[:, mask].argmax(dim=0)
                != base_prediction[mask]
            ).sum().item()
        )
        records.append(
            {
                "region_index": index,
                "accepted": accepted,
                "area": evidence.area,
                "purity": evidence.purity,
                "margin": evidence.margin,
                "mean_alpha": float(alpha_values.mean().item()),
                "changed_pixels": changed_pixels,
            }
        )
    return output_probabilities.argmax(dim=0), records


__all__ = [
    "VARIANTS",
    "SelectiveContextConfig",
    "foreground_region_distribution",
    "interior_weight",
    "local_foreground_context",
    "normalized_entropy",
    "selective_context_refinement",
]
