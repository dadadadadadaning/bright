from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VARIANTS = (
    "m0_raw_p3",
    "m1_single_prototype_s2",
    "m2_probability_multiproto",
    "m3_feature_multiproto",
    "m4_hybrid_single_multi",
    "m5_hybrid_selective_multi",
)


@dataclass(frozen=True)
class MultiPrototypeConfig:
    seed_confidence: float = 0.70
    minimum_seed_pixels: int = 2
    feature_temperature: float = 0.25
    maximum_alpha: float = 0.75
    uncertainty_threshold: float = 0.50
    assignment_margin: float = 0.10

    def __post_init__(self) -> None:
        for name in ("seed_confidence", "maximum_alpha", "uncertainty_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.minimum_seed_pixels < 1:
            raise ValueError("minimum_seed_pixels must be positive")
        if self.feature_temperature <= 0.0:
            raise ValueError("feature_temperature must be positive")
        if self.assignment_margin < 0.0:
            raise ValueError("assignment_margin must be nonnegative")


def _foreground(probabilities: torch.Tensor) -> torch.Tensor:
    values = probabilities[:, 1:4].float().clamp_min(1e-8)
    return values / values.sum(1, keepdim=True).clamp_min(1e-8)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1) / probabilities.new_tensor(3.0).log()


def multi_prototype_context(
    probabilities: torch.Tensor,
    feature: torch.Tensor,
    region_labels: torch.Tensor,
    *,
    config: MultiPrototypeConfig,
    use_feature_assignment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Return refined probabilities and a prediction-only selective mask.

    Each (LocCC, predicted foreground class) pair owns one prototype.  A LocCC
    is eligible only when at least two class prototypes have enough confident
    seed pixels, so single-state regions are left to the locked S2 baseline.
    """

    if probabilities.ndim != 4 or probabilities.shape[0] != 1 or probabilities.shape[1] != 4:
        raise ValueError("probabilities must be [1,4,H,W]")
    if feature.ndim != 4 or feature.shape[0] != 1 or feature.shape[-2:] != probabilities.shape[-2:]:
        raise ValueError("feature must be [1,C,H,W] on the probability grid")
    if region_labels.shape != (1, *probabilities.shape[-2:]):
        raise ValueError("region_labels must be [1,H,W]")

    foreground = _foreground(probabilities)
    confidence, prediction = foreground.max(1)
    full_prediction = probabilities.argmax(1)
    foreground_mass_map = probabilities[:, 1:4].sum(1)
    labels = region_labels.long()
    maximum_label = int(labels.max().item())
    if maximum_label == 0:
        empty = torch.zeros_like(labels, dtype=torch.bool)
        return probabilities.clone(), empty, empty, {
            "regions": 0, "eligible_regions": 0, "available_prototypes": 0,
            "selective_fraction": 0.0,
        }

    height, width = labels.shape[-2:]
    pixels = height * width
    flat_labels = labels.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    flat_full_prediction = full_prediction.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    flat_foreground_mass = foreground_mass_map.reshape(-1)
    flat_foreground = foreground[0].permute(1, 2, 0).reshape(pixels, 3)
    flat_feature = feature[0].float().permute(1, 2, 0).reshape(pixels, feature.shape[1])
    seed = (
        (flat_labels > 0)
        & (flat_full_prediction > 0)
        & (flat_confidence >= config.seed_confidence)
    )
    group_count = (maximum_label + 1) * 3
    group_index = flat_labels * 3 + flat_prediction
    seed_group = group_index[seed]
    seed_weight = flat_confidence[seed] * flat_foreground_mass[seed]

    counts = torch.zeros(group_count, device=probabilities.device)
    counts.index_add_(0, seed_group, torch.ones_like(seed_weight))
    weight_sums = torch.zeros_like(counts)
    weight_sums.index_add_(0, seed_group, seed_weight)
    probability_sums = torch.zeros((group_count, 3), device=probabilities.device)
    probability_sums.index_add_(0, seed_group, flat_foreground[seed] * seed_weight[:, None])
    feature_sums = torch.zeros((group_count, feature.shape[1]), device=probabilities.device)
    feature_sums.index_add_(0, seed_group, flat_feature[seed] * seed_weight[:, None])
    prototype_probability = probability_sums / weight_sums[:, None].clamp_min(1e-8)
    prototype_probability = prototype_probability / prototype_probability.sum(1, keepdim=True).clamp_min(1e-8)
    prototype_feature = F.normalize(feature_sums / weight_sums[:, None].clamp_min(1e-8), dim=1, eps=1e-6)
    available = counts >= float(config.minimum_seed_pixels)

    classes = torch.arange(3, device=probabilities.device)
    per_pixel_groups = flat_labels[:, None] * 3 + classes[None]
    pixel_available = available[per_pixel_groups]
    available_count = pixel_available.sum(1)
    eligible = (flat_labels > 0) & (available_count >= 2)
    gathered_probability = prototype_probability[per_pixel_groups]
    diagonal_confidence = gathered_probability.gather(
        2, classes[None, :, None].expand(pixels, -1, 1)
    ).squeeze(2).clamp_min(1e-8)
    scores = flat_foreground.clamp_min(1e-8).log() + diagonal_confidence.log()
    if use_feature_assignment:
        normalized_pixel = F.normalize(flat_feature, dim=1, eps=1e-6)
        gathered_feature = prototype_feature[per_pixel_groups]
        similarity = (gathered_feature * normalized_pixel[:, None]).sum(2)
        scores = scores + similarity / config.feature_temperature
    scores = scores.masked_fill(~pixel_available, -1e4)
    assignment = scores.softmax(1)
    context = (assignment[:, :, None] * gathered_probability).sum(1)
    context = context / context.sum(1, keepdim=True).clamp_min(1e-8)
    entropy = _entropy(foreground).reshape(-1)
    alpha = config.maximum_alpha * entropy
    mixed_foreground = (1.0 - alpha[:, None]) * flat_foreground + alpha[:, None] * context
    mixed_foreground = mixed_foreground / mixed_foreground.sum(1, keepdim=True).clamp_min(1e-8)
    mixed_foreground = torch.where(eligible[:, None], mixed_foreground, flat_foreground)

    candidate = probabilities.float().clone()
    foreground_mass = probabilities[:, 1:4].sum(1, keepdim=True)[0].permute(1, 2, 0).reshape(pixels, 1)
    output_foreground = foreground_mass * mixed_foreground
    candidate[:, 1:4] = output_foreground.reshape(height, width, 3).permute(2, 0, 1)[None]
    candidate = candidate / candidate.sum(1, keepdim=True).clamp_min(1e-8)
    eligible_map = eligible.reshape(1, 1, height, width)
    output = torch.where(eligible_map, candidate, probabilities.float())
    top = scores.topk(2, dim=1).values
    margin = top[:, 0] - top[:, 1]
    selective = eligible & (entropy >= config.uncertainty_threshold) & (margin >= config.assignment_margin)
    region_available = available.reshape(maximum_label + 1, 3)
    eligible_regions = (region_available.sum(1) >= 2)
    return output, eligible.reshape(1, height, width), selective.reshape(1, height, width), {
        "regions": maximum_label,
        "eligible_regions": int(eligible_regions[1:].sum().item()),
        "available_prototypes": int(available.reshape(maximum_label + 1, 3)[1:].sum().item()),
        "eligible_fraction": float(eligible.float().mean()),
        "selective_fraction": float(selective.float().mean()),
        "mean_assignment_margin_on_eligible": float(margin[eligible].mean()) if bool(eligible.any()) else 0.0,
    }


def regions_to_label_map(regions: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
    if not regions:
        return torch.zeros((1, *output_size), dtype=torch.long)
    full = torch.zeros_like(regions[0], dtype=torch.long)
    for label, mask in enumerate(regions, start=1):
        full[mask.bool()] = label
    return F.interpolate(full[None, None].float(), size=output_size, mode="nearest")[:, 0].long()


def probability_residual_to_prediction(
    base_low: torch.Tensor,
    refined_low: torch.Tensor,
    base_full_logits: torch.Tensor,
    full_support: torch.Tensor,
) -> torch.Tensor:
    delta = refined_low.clamp_min(1e-8).log() - base_low.clamp_min(1e-8).log()
    delta = F.interpolate(delta, size=base_full_logits.shape[-2:], mode="bilinear", align_corners=False)
    output = base_full_logits.float() + delta * full_support.float()
    return output.argmax(1)


__all__ = [
    "VARIANTS", "MultiPrototypeConfig", "multi_prototype_context",
    "probability_residual_to_prediction", "regions_to_label_map",
]
