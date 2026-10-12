# -*- coding: utf-8 -*-
"""Intact-anchored RGB--SAR alignment for controlled AL0--AL3 experiments.

This module is intentionally independent from P3, RM, SRAC and OA-HSDNet.
Frozen modality-specific encoders provide Stage2/Stage3 features.  AL1 trains
only the two local aligners on GT-Intact building pixels.  AL2 consumes the
frozen residual maps without a classifier.  AL3 adds a separate occurrence
head only after the AL2 evidence gate passes.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


AL_STAGE_CHANNELS: Mapping[str, int] = {"stage2": 192, "stage3": 384}


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class DescriptorProjection(nn.Module):
    """Modality-specific projection with no cross-modal weight sharing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def local_correlation_volume(
    query: torch.Tensor,
    key: torch.Tensor,
    radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine correlation and validity for a local SAR search window."""

    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query/key must share [B,C,H,W]")
    if radius < 1:
        raise ValueError("radius must be positive")
    batch, channels, height, width = query.shape
    window = 2 * radius + 1
    neighbours = window * window
    query = F.normalize(query.float(), dim=1, eps=1e-6)
    key = F.normalize(key.float(), dim=1, eps=1e-6)
    patches = F.unfold(key, window, padding=radius).reshape(
        batch, channels, neighbours, height * width
    )
    correlation = torch.einsum(
        "bcl,bckl->bkl", query.reshape(batch, channels, -1), patches
    ).reshape(batch, neighbours, height, width)
    valid = F.unfold(
        torch.ones((batch, 1, height, width), device=query.device),
        window,
        padding=radius,
    ).reshape(batch, neighbours, height, width)
    return correlation.masked_fill(valid < 0.5, -1e4), valid


def _offset_table(radius: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            (float(dx), float(dy))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ],
        device=device,
        dtype=torch.float32,
    )


def warp_with_offsets(value: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Sample ``value`` at pixel coordinates displaced by ``offsets``."""

    if offsets.ndim != 4 or offsets.shape[1] != 2:
        raise ValueError("offsets must be [B,2,H,W]")
    batch, _, height, width = value.shape
    if offsets.shape[0] != batch or offsets.shape[-2:] != (height, width):
        raise ValueError("offset/value shape mismatch")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=torch.float32),
        torch.arange(width, device=value.device, dtype=torch.float32),
        indexing="ij",
    )
    x = xx[None] + offsets[:, 0]
    y = yy[None] + offsets[:, 1]
    x = 2.0 * x / max(width - 1, 1) - 1.0
    y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1)
    return F.grid_sample(
        value, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


class LocalDeformableAligner(nn.Module):
    """Learn cross-modal descriptors, local correlation and bounded offsets."""

    def __init__(
        self,
        in_channels: int,
        descriptor_channels: int = 64,
        radius: int = 2,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.rgb_projection = DescriptorProjection(in_channels, descriptor_channels)
        self.sar_projection = DescriptorProjection(in_channels, descriptor_channels)
        self.radius = int(radius)
        self.temperature = float(temperature)
        neighbours = (2 * radius + 1) ** 2
        self.offset_refiner = nn.Sequential(
            nn.Conv2d(neighbours + 2, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.offset_refiner[-1].weight)
        nn.init.zeros_(self.offset_refiner[-1].bias)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if sar_feature.shape[-2:] != rgb_feature.shape[-2:]:
            sar_feature = F.interpolate(
                sar_feature,
                rgb_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        rgb = self.rgb_projection(rgb_feature)
        sar = self.sar_projection(sar_feature)
        correlation, valid = local_correlation_volume(rgb, sar, self.radius)
        probability = (correlation / self.temperature).softmax(dim=1)
        offsets = _offset_table(self.radius, rgb.device)
        base_x = (probability * offsets[:, 0].view(1, -1, 1, 1)).sum(1)
        base_y = (probability * offsets[:, 1].view(1, -1, 1, 1)).sum(1)
        base = torch.stack((base_x, base_y), dim=1)
        entropy = -(
            probability.clamp_min(1e-8).log() * probability
        ).sum(1, keepdim=True) / math.log(float(probability.shape[1]))
        top = correlation.topk(2, dim=1).values
        margin = (top[:, 0:1] - top[:, 1:2]).clamp_min(0.0)
        correction = self.radius * torch.tanh(
            self.offset_refiner(torch.cat((probability, entropy, margin), dim=1))
        )
        final_offset = (base + correction).clamp(-self.radius, self.radius)
        aligned_sar = warp_with_offsets(sar, final_offset)
        rgb_unit = F.normalize(rgb.float(), dim=1, eps=1e-6)
        sar_unit = F.normalize(sar.float(), dim=1, eps=1e-6)
        aligned_unit = F.normalize(aligned_sar.float(), dim=1, eps=1e-6)
        unaligned_residual = (1.0 - (rgb_unit * sar_unit).sum(1, keepdim=True)).clamp(0, 2)
        aligned_residual = (1.0 - (rgb_unit * aligned_unit).sum(1, keepdim=True)).clamp(0, 2)
        return {
            "rgb_descriptor": rgb,
            "sar_descriptor": sar,
            "aligned_sar_descriptor": aligned_sar,
            "correlation": correlation,
            "valid": valid,
            "probability": probability,
            "entropy": entropy,
            "margin": margin,
            "base_offset": base,
            "offset_correction": correction,
            "offset": final_offset,
            "unaligned_residual": unaligned_residual,
            "aligned_residual": aligned_residual,
        }


def alignment_training_loss(
    outputs: Mapping[str, torch.Tensor],
    intact_mask: torch.Tensor,
    *,
    variance_floor: float = 0.25,
    smoothness_weight: float = 0.05,
    magnitude_weight: float = 0.01,
    variance_weight: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """AL1 loss. Only intact building pixels contribute correspondence loss."""

    residual = outputs["aligned_residual"]
    # Preserve the fractional coverage of small buildings on coarse Stage2/3
    # grids. Nearest-neighbour downsampling can erase a valid Intact component
    # completely depending on its sub-pixel position, making otherwise valid
    # batches fail nondeterministically.
    mask = F.interpolate(intact_mask.float(), residual.shape[-2:], mode="area")
    valid_count = mask.sum()
    if float(valid_count.detach()) <= 1e-8:
        raise ValueError("alignment batch contains no intact support at source resolution")
    robust = torch.sqrt(residual.square() + 1e-4)
    data = (robust * mask).sum() / valid_count.clamp_min(1e-8)
    offset = outputs["offset"]
    smoothness = (
        (offset[:, :, 1:] - offset[:, :, :-1]).abs().mean()
        + (offset[:, :, :, 1:] - offset[:, :, :, :-1]).abs().mean()
    )
    magnitude = (offset.square().sum(1, keepdim=True).sqrt() * mask).sum() / valid_count
    variance_terms = []
    for name in ("rgb_descriptor", "sar_descriptor"):
        descriptor = outputs[name].float()
        flat = descriptor.permute(1, 0, 2, 3).reshape(descriptor.shape[1], -1)
        std = flat.std(dim=1, unbiased=False)
        variance_terms.append(F.relu(float(variance_floor) - std).mean())
    variance = sum(variance_terms)
    total = (
        data
        + float(smoothness_weight) * smoothness
        + float(magnitude_weight) * magnitude
        + float(variance_weight) * variance
    )
    return {
        "total": total,
        "data": data.detach(),
        "smoothness": smoothness.detach(),
        "magnitude": magnitude.detach(),
        "variance": variance.detach(),
        "intact_pixels": valid_count.detach(),
    }


class IntactAnchoredMultiScaleAlignment(nn.Module):
    """Independent Stage2/Stage3 aligners and an optional AL3 occurrence head."""

    stages: Sequence[str] = ("stage2", "stage3")

    def __init__(
        self,
        descriptor_channels: int = 64,
        radius: int = 2,
        temperature: float = 0.10,
        occurrence_head: bool = False,
    ) -> None:
        super().__init__()
        self.aligners = nn.ModuleDict(
            {
                stage: LocalDeformableAligner(
                    channels, descriptor_channels, radius, temperature
                )
                for stage, channels in AL_STAGE_CHANNELS.items()
            }
        )
        self.occurrence_head = (
            nn.Sequential(
                nn.Conv2d(6, 32, 3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, 2, 1),
            )
            if occurrence_head
            else None
        )

    def freeze_alignment(self) -> None:
        self.aligners.eval()
        for parameter in self.aligners.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        rgb_stages: Mapping[str, torch.Tensor],
        sar_stages: Mapping[str, torch.Tensor],
    ) -> Dict[str, object]:
        aligned: Dict[str, Dict[str, torch.Tensor]] = {}
        for stage in self.stages:
            aligned[stage] = self.aligners[stage](rgb_stages[stage], sar_stages[stage])
        target = aligned["stage2"]["aligned_residual"].shape[-2:]
        evidence = []
        for stage in self.stages:
            values = aligned[stage]
            for name in ("aligned_residual", "entropy", "margin"):
                value = values[name]
                if value.shape[-2:] != target:
                    value = F.interpolate(value, target, mode="bilinear", align_corners=False)
                evidence.append(value)
        occurrence_input = torch.cat(evidence, dim=1)
        result: Dict[str, object] = {
            "stages": aligned,
            "occurrence_input": occurrence_input,
            "aligned_residual": occurrence_input[:, [0, 3]].mean(1, keepdim=True),
            "unaligned_residual": torch.stack(
                [
                    F.interpolate(
                        aligned[stage]["unaligned_residual"], target,
                        mode="bilinear", align_corners=False,
                    )
                    for stage in self.stages
                ],
                dim=0,
            ).mean(0),
        }
        if self.occurrence_head is not None:
            result["occurrence_logits"] = self.occurrence_head(occurrence_input)
        return result


__all__ = [
    "AL_STAGE_CHANNELS",
    "DescriptorProjection",
    "IntactAnchoredMultiScaleAlignment",
    "LocalDeformableAligner",
    "alignment_training_loss",
    "local_correlation_volume",
    "warp_with_offsets",
]
