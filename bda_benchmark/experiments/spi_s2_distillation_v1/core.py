from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import (
    RegionMaps,
    broadcast_table,
    scatter_region_mean,
)


@dataclass(frozen=True)
class DistillationConfig:
    minimum_purity: float = 0.70
    minimum_margin: float = 0.10
    blend_strength: float = 0.75

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_purity <= 1:
            raise ValueError("minimum_purity must be in [0,1]")
        if self.minimum_margin < 0:
            raise ValueError("minimum_margin must be nonnegative")
        if not 0 <= self.blend_strength <= 1:
            raise ValueError("blend_strength must be in [0,1]")


@dataclass
class StructuralBatch:
    teacher_probability: torch.Tensor
    distillation_weight: torch.Tensor
    accepted_support: torch.Tensor
    region_context: torch.Tensor
    region_variance: torch.Tensor
    region_distribution: torch.Tensor
    accepted_regions: int
    total_regions: int


def _region_sums(values: torch.Tensor, labels: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate BxCxHxW values into (count+1)xC FP32 tables."""
    if values.ndim != 4 or labels.shape != values.shape[:1] + values.shape[2:]:
        raise ValueError("values and labels are not aligned")
    flat_values = values.permute(0, 2, 3, 1).reshape(-1, values.shape[1]).float()
    flat_labels = labels.reshape(-1)
    sums = torch.zeros((count + 1, values.shape[1]), device=values.device, dtype=torch.float32)
    counts = torch.zeros((count + 1, 1), device=values.device, dtype=torch.float32)
    sums.index_add_(0, flat_labels, flat_values)
    counts.index_add_(0, flat_labels, torch.ones((flat_labels.numel(), 1), device=values.device))
    return sums, counts


def build_structural_batch(
    base_logits: torch.Tensor,
    feature: torch.Tensor,
    regions: RegionMaps,
    config: DistillationConfig,
) -> StructuralBatch:
    """Create a prediction-only, low-grid S2 soft teacher and region statistics.

    No semantic target is accepted by this function. Region zero is background
    and is never distilled or modified.
    """
    if base_logits.shape[0] != feature.shape[0] or base_logits.shape[-2:] != feature.shape[-2:]:
        raise ValueError("base logits and feature must share batch/spatial dimensions")
    if base_logits.shape[1] != 4:
        raise ValueError("BRIGHT damage logits must have four classes")
    labels = regions.labels
    if labels.shape != base_logits.shape[:1] + base_logits.shape[2:]:
        raise ValueError("region labels and logits are not aligned")

    probability = F.softmax(base_logits.float(), dim=1)
    foreground = probability[:, 1:4]
    sums, counts = _region_sums(foreground, labels, regions.region_count)
    means = sums / counts.clamp_min(1.0)
    distribution = means / means.sum(dim=1, keepdim=True).clamp_min(1e-8)
    distribution[0] = 0

    prediction = probability.argmax(dim=1)
    foreground_votes = torch.stack([(prediction == c).float() for c in (1, 2, 3)], dim=1)
    vote_sums, _ = _region_sums(foreground_votes, labels, regions.region_count)
    winning_index = vote_sums.argmax(dim=1)
    winning_votes = vote_sums.gather(1, winning_index[:, None])[:, 0]
    purity = winning_votes / counts[:, 0].clamp_min(1.0)
    chosen_score = distribution.gather(1, winning_index[:, None])[:, 0]
    alternatives = distribution.clone()
    alternatives.scatter_(1, winning_index[:, None], -1.0)
    margin = chosen_score - alternatives.max(dim=1).values
    accepted = (purity >= config.minimum_purity) & (margin >= config.minimum_margin)
    accepted[0] = False

    per_pixel_distribution = distribution[labels].permute(0, 3, 1, 2)
    support = accepted[labels].unsqueeze(1)
    alpha = support.float() * float(config.blend_strength)
    region_probability = torch.zeros_like(probability)
    region_probability[:, 1:4] = per_pixel_distribution
    teacher = (1.0 - alpha) * probability + alpha * region_probability
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Weight uses deployable prediction-only evidence, never GT correctness.
    confidence = (purity * margin.clamp_min(0.0))[labels].unsqueeze(1)
    distillation_weight = support.float() * confidence

    region_mean, _ = scatter_region_mean(feature, labels, regions.region_count)
    context = broadcast_table(region_mean, labels)
    squared_mean_sums, squared_counts = _region_sums(feature.float().square(), labels, regions.region_count)
    squared_mean = squared_mean_sums / squared_counts.clamp_min(1.0)
    variance_table = (squared_mean - region_mean.float().square()).clamp_min(0.0)
    variance_table[0] = 0
    variance = broadcast_table(variance_table.to(feature.dtype), labels)
    region_distribution = per_pixel_distribution.to(feature.dtype)
    return StructuralBatch(
        teacher_probability=teacher,
        distillation_weight=distillation_weight,
        accepted_support=support.to(feature.dtype),
        region_context=context,
        region_variance=variance,
        region_distribution=region_distribution,
        accepted_regions=int(accepted.sum().item()),
        total_regions=int(regions.region_count),
    )


class LocalResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CorrectedGBC(nn.Module):
    """Minimal 2-D GBC-style residual block; no faulty InstanceNorm3d path."""
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1

        def bottleneck(kernel: int) -> nn.Sequential:
            middle = max(channels // 4, 8)
            return nn.Sequential(
                nn.Conv2d(channels, middle, 1, bias=False),
                nn.Conv2d(middle, middle, kernel, padding=kernel // 2, groups=middle, bias=False),
                nn.Conv2d(middle, channels, 1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
            )

        self.content = nn.Sequential(bottleneck(3), bottleneck(3))
        self.gate = nn.Sequential(bottleneck(1), nn.Sigmoid())
        self.out = bottleneck(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.out(self.content(x) * self.gate(x))


class StructuralResidualStudent(nn.Module):
    """Small zero-initialized residual student conditioned by LocCC statistics."""
    def __init__(self, feature_channels: int = 256, hidden: int = 64, block: str = "local", use_variance: bool = False):
        super().__init__()
        self.use_variance = bool(use_variance)
        self.feature = nn.Conv2d(feature_channels, hidden, 1, bias=False)
        self.context = nn.Conv2d(feature_channels, hidden, 1, bias=False)
        self.difference = nn.Conv2d(feature_channels, hidden, 1, bias=False)
        self.variance = nn.Conv2d(feature_channels, hidden, 1, bias=False) if use_variance else None
        self.base_probability = nn.Conv2d(4, hidden, 1, bias=False)
        self.region_probability = nn.Conv2d(3, hidden, 1, bias=False)
        self.mix = LocalResidualBlock(hidden) if block == "local" else CorrectedGBC(hidden)
        self.output = nn.Conv2d(hidden, 4, 1)
        self.residual_scale_raw = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_scale_raw)

    def forward(self, feature: torch.Tensor, base_probability: torch.Tensor, structural: StructuralBatch) -> torch.Tensor:
        hidden = (
            self.feature(feature)
            + self.context(structural.region_context)
            + self.difference(feature - structural.region_context)
            + self.base_probability(base_probability)
            + self.region_probability(structural.region_distribution)
        )
        if self.variance is not None:
            hidden = hidden + self.variance(structural.region_variance)
        residual = self.output(self.mix(hidden))
        return self.residual_scale() * torch.tanh(residual) * structural.accepted_support


def weighted_forward_kl(student_logits: torch.Tensor, teacher_probability: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per_class = F.kl_div(
        F.log_softmax(student_logits.float(), dim=1),
        teacher_probability.float(),
        reduction="none",
    )
    per_pixel = per_class.sum(dim=1, keepdim=True)
    return (per_pixel * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def outside_anchor_kl(student_logits: torch.Tensor, base_logits: torch.Tensor, accepted_support: torch.Tensor) -> torch.Tensor:
    base_probability = F.softmax(base_logits.float(), dim=1)
    per_pixel = F.kl_div(
        F.log_softmax(student_logits.float(), dim=1), base_probability, reduction="none"
    ).sum(dim=1, keepdim=True)
    weight = 1.0 - accepted_support.float()
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)


__all__ = [
    "CorrectedGBC",
    "DistillationConfig",
    "StructuralBatch",
    "StructuralResidualStudent",
    "build_structural_batch",
    "outside_anchor_kl",
    "weighted_forward_kl",
]
