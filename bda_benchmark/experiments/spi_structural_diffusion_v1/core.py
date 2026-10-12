from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VARIANTS = (
    "n0_raw_p3",
    "n1_selective_context_s2",
    "n2_loccc_uniform_diffusion",
    "n3_feature_edge_diffusion",
    "n4_modality_safe_diffusion",
    "n5_semantic_barrier_diffusion",
)


@dataclass(frozen=True)
class StructuralDiffusionConfig:
    iterations: int = 4
    maximum_alpha: float = 0.75
    feature_temperature: float = 0.25
    modality_temperature: float = 0.50
    semantic_temperature: float = 0.10

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 <= self.maximum_alpha <= 1.0:
            raise ValueError("maximum_alpha must be in [0,1]")
        for name in (
            "feature_temperature",
            "modality_temperature",
            "semantic_temperature",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


def _normalized_foreground(probabilities: torch.Tensor) -> torch.Tensor:
    foreground = probabilities[:, 1:4].float().clamp_min(1e-8)
    return foreground / foreground.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _normalized_entropy(foreground: torch.Tensor) -> torch.Tensor:
    denominator = foreground.new_tensor(3.0).log()
    return -(foreground * foreground.clamp_min(1e-8).log()).sum(1, keepdim=True) / denominator


def _cosine_dissimilarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=1, eps=1e-6)
    right = F.normalize(right.float(), dim=1, eps=1e-6)
    return (1.0 - (left * right).sum(1, keepdim=True)).clamp(0.0, 2.0)


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float().clamp_min(1e-8)
    right = right.float().clamp_min(1e-8)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(1, keepdim=True)
        + (right * (right.log() - middle.log())).sum(1, keepdim=True)
    )


def _shift_slices(direction: str):
    if direction == "up":
        return (slice(1, None), slice(None)), (slice(None, -1), slice(None))
    if direction == "down":
        return (slice(None, -1), slice(None)), (slice(1, None), slice(None))
    if direction == "left":
        return (slice(None), slice(1, None)), (slice(None), slice(None, -1))
    if direction == "right":
        return (slice(None), slice(None, -1)), (slice(None), slice(1, None))
    raise ValueError(direction)


def _edge_weights(
    labels: torch.Tensor,
    foreground: torch.Tensor,
    fused_feature: torch.Tensor | None,
    modality_reliability: torch.Tensor | None,
    *,
    mode: str,
    config: StructuralDiffusionConfig,
) -> list[tuple[tuple[slice, slice], tuple[slice, slice], torch.Tensor]]:
    edges = []
    normalized_feature = None
    if mode in {"feature", "modality", "semantic"}:
        if fused_feature is None:
            raise ValueError(f"{mode} mode requires fused_feature")
        normalized_feature = F.normalize(fused_feature.float(), dim=1, eps=1e-6)
    for direction in ("up", "down", "left", "right"):
        destination, source = _shift_slices(direction)
        dst_label = labels[:, destination[0], destination[1]]
        src_label = labels[:, source[0], source[1]]
        valid = (dst_label > 0) & (dst_label == src_label)
        weight = valid[:, None].float()
        if normalized_feature is not None:
            dst = normalized_feature[:, :, destination[0], destination[1]]
            src = normalized_feature[:, :, source[0], source[1]]
            distance = (1.0 - (dst * src).sum(1, keepdim=True)).clamp(0.0, 2.0)
            weight *= torch.exp(-distance / config.feature_temperature)
        if modality_reliability is not None and mode in {"modality", "semantic"}:
            reliability = modality_reliability
            dst = reliability[:, :, destination[0], destination[1]]
            src = reliability[:, :, source[0], source[1]]
            weight *= torch.sqrt((dst * src).clamp_min(0.0))
        if mode == "semantic":
            dst = foreground[:, :, destination[0], destination[1]]
            src = foreground[:, :, source[0], source[1]]
            weight *= torch.exp(
                -_js_divergence(dst, src) / config.semantic_temperature
            )
        edges.append((destination, source, weight))
    return edges


def structural_diffusion(
    probabilities: torch.Tensor,
    region_labels: torch.Tensor,
    *,
    fused_feature: torch.Tensor | None = None,
    rgb_feature: torch.Tensor | None = None,
    sar_feature: torch.Tensor | None = None,
    mode: str,
    config: StructuralDiffusionConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Diffuse foreground evidence only inside the same positive LocCC label.

    LocCC labels are barriers, not class labels: different labels never exchange
    evidence, while pixels inside one label retain separate probability vectors.
    """

    if mode not in {"uniform", "feature", "modality", "semantic"}:
        raise ValueError(f"unknown diffusion mode: {mode}")
    if probabilities.ndim != 4 or probabilities.shape[1] != 4:
        raise ValueError("probabilities must be [B,4,H,W]")
    if region_labels.shape != probabilities.shape[:1] + probabilities.shape[-2:]:
        raise ValueError("region_labels must be [B,H,W]")
    support = (region_labels > 0)[:, None]
    base = probabilities.float()
    foreground = _normalized_foreground(base)
    entropy = _normalized_entropy(foreground)
    modality_reliability = None
    if mode in {"modality", "semantic"}:
        if rgb_feature is None or sar_feature is None:
            raise ValueError(f"{mode} mode requires rgb_feature and sar_feature")
        disagreement = _cosine_dissimilarity(rgb_feature, sar_feature)
        modality_reliability = torch.exp(
            -disagreement / config.modality_temperature
        ).clamp(0.0, 1.0)
    alpha = config.maximum_alpha * entropy
    if modality_reliability is not None:
        alpha = alpha * modality_reliability
    alpha = alpha * support.float()
    edges = _edge_weights(
        region_labels,
        foreground,
        fused_feature,
        modality_reliability,
        mode=mode,
        config=config,
    )
    current = foreground
    for _ in range(config.iterations):
        numerator = torch.zeros_like(current)
        denominator = torch.zeros_like(current[:, :1])
        for destination, source, weight in edges:
            numerator[:, :, destination[0], destination[1]] += (
                weight * current[:, :, source[0], source[1]]
            )
            denominator[:, :, destination[0], destination[1]] += weight
        neighbor = numerator / denominator.clamp_min(1e-8)
        has_neighbor = denominator > 0
        proposal = (1.0 - alpha) * foreground + alpha * neighbor
        current = torch.where(has_neighbor, proposal, foreground)
        current = current / current.sum(1, keepdim=True).clamp_min(1e-8)
    output = base.clone()
    foreground_mass = base[:, 1:4].sum(1, keepdim=True)
    refined_foreground = foreground_mass * current
    output[:, 1:4] = torch.where(support, refined_foreground, base[:, 1:4])
    output = output / output.sum(1, keepdim=True).clamp_min(1e-8)
    diagnostics = {
        "support_fraction": float(support.float().mean()),
        "mean_alpha_on_support": float(alpha[support].mean()) if bool(support.any()) else 0.0,
        "mean_modality_reliability": (
            float(modality_reliability[support].mean())
            if modality_reliability is not None and bool(support.any())
            else 1.0
        ),
    }
    return output, diagnostics


def regions_to_label_map(
    regions: list[torch.Tensor], output_size: tuple[int, int]
) -> torch.Tensor:
    if not regions:
        return torch.zeros((1, *output_size), dtype=torch.long)
    full = torch.zeros_like(regions[0], dtype=torch.long)
    for label, mask in enumerate(regions, start=1):
        full[mask.bool()] = label
    low = F.interpolate(
        full[None, None].float(), size=output_size, mode="nearest"
    )[0, 0].long()
    return low[None]


def residual_to_full_logits(
    base_low_probabilities: torch.Tensor,
    refined_low_probabilities: torch.Tensor,
    base_full_logits: torch.Tensor,
    full_support: torch.Tensor,
) -> torch.Tensor:
    if full_support.ndim == 3:
        full_support = full_support[:, None]
    delta = refined_low_probabilities.clamp_min(1e-8).log() - base_low_probabilities.clamp_min(1e-8).log()
    delta = F.interpolate(delta, size=base_full_logits.shape[-2:], mode="bilinear", align_corners=False)
    return base_full_logits.float() + delta * full_support.float()


__all__ = [
    "VARIANTS",
    "StructuralDiffusionConfig",
    "regions_to_label_map",
    "residual_to_full_logits",
    "structural_diffusion",
]
