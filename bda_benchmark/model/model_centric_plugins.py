from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _fixed_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    weight = kernel.to(device=x.device, dtype=x.dtype).expand(channels, 1, -1, -1)
    return F.conv2d(x, weight, padding=kernel.shape[-1] // 2, groups=channels)


class RGBStructureAdapter(nn.Module):
    """Scharr/Gaussian/strip structure branch with an identity initial state."""

    def __init__(self, channels: int, hidden: int = 32, strip_kernel: int = 15) -> None:
        super().__init__()
        if strip_kernel < 3 or strip_kernel % 2 == 0:
            raise ValueError("strip_kernel must be an odd integer >= 3")
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        pad = strip_kernel // 2
        self.horizontal = nn.Conv2d(
            hidden, hidden, (1, strip_kernel), padding=(0, pad), groups=hidden, bias=False
        )
        self.vertical = nn.Conv2d(
            hidden, hidden, (strip_kernel, 1), padding=(pad, 0), groups=hidden, bias=False
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.gain = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "scharr_x",
            torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
            .div(32.0)[None, None],
        )
        self.register_buffer("scharr_y", self.scharr_x.transpose(-1, -2).contiguous())
        self.register_buffer(
            "gaussian",
            torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
            .div(16.0)[None, None],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.reduce(x)
        edge_x = _fixed_depthwise(latent, self.scharr_x)
        edge_y = _fixed_depthwise(latent, self.scharr_y)
        smooth = _fixed_depthwise(latent, self.gaussian)
        strip = self.vertical(self.horizontal(latent))
        update = self.fuse(torch.cat((edge_x, edge_y, smooth, strip), dim=1))
        return x + torch.tanh(self.gain) * update


class SARScatteringFrequencyExperts(nn.Module):
    """Fixed Haar/MAD experts followed by a small content-dependent router."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.hidden = hidden
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.expert_projection = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 1, bias=False) for _ in range(3)]
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, max(hidden // 4, 4), 1),
            nn.GELU(),
            nn.Conv2d(max(hidden // 4, 4), 3, 1),
        )
        self.expand = nn.Sequential(
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.gain = nn.Parameter(torch.zeros(()))
        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).div(2.0)
        self.register_buffer("haar", haar[:, None])

    def _haar(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.haar.to(device=x.device, dtype=x.dtype).repeat(self.hidden, 1, 1, 1)
        values = F.conv2d(x, weight, stride=2, groups=self.hidden)
        batch, _, height, width = values.shape
        return values.view(batch, self.hidden, 4, height, width)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        latent = self.reduce(x)
        bands = self._haar(latent)
        target = latent.shape[-2:]
        low = F.interpolate(bands[:, :, 0], target, mode="bilinear", align_corners=False)
        directional = F.interpolate(
            torch.sqrt(bands[:, :, 1].square() + bands[:, :, 2].square() + 1e-6),
            target,
            mode="bilinear",
            align_corners=False,
        )
        high = F.interpolate(bands[:, :, 3].abs(), target, mode="bilinear", align_corners=False)
        mad = (latent - F.avg_pool2d(latent, 3, stride=1, padding=1)).abs()
        experts = (low, directional, high + mad)
        route = torch.softmax(self.router(latent), dim=1)
        mixed = sum(
            route[:, index : index + 1] * projection(value)
            for index, (projection, value) in enumerate(zip(self.expert_projection, experts))
        )
        output = x + torch.tanh(self.gain) * self.expand(mixed)
        if not return_diagnostics:
            return output
        entropy = -(route.clamp_min(1e-8) * route.clamp_min(1e-8).log()).sum(1) / math.log(3.0)
        return output, {"sar_expert_weights": route, "sar_router_entropy": entropy}


@dataclass
class InteractionOutput:
    feature: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    offset: torch.Tensor


class RegistrationTolerantInteraction(nn.Module):
    """RGB-anchored local SAR lookup added on top of the original fusion."""

    def __init__(
        self,
        feature_channels: int = 256,
        match_channels: int = 32,
        value_channels: int = 64,
        window_size: int = 3,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = window_size
        self.temperature = float(temperature)
        self.query = nn.Conv2d(feature_channels, match_channels, 1, bias=False)
        self.key = nn.Conv2d(feature_channels, match_channels, 1, bias=False)
        self.value = nn.Conv2d(feature_channels, value_channels, 1, bias=False)
        self.update = nn.Sequential(
            nn.Conv2d(feature_channels + value_channels, feature_channels, 1, bias=False),
            nn.GroupNorm(_groups(feature_channels), feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
        )
        self.gain = nn.Parameter(torch.zeros(()))
        radius = window_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.register_buffer(
            "offset_table", torch.stack((xx.flatten(), yy.flatten()), dim=1).float(), persistent=False
        )

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor, parent_fused: torch.Tensor
    ) -> InteractionOutput:
        if rgb.shape != sar.shape:
            raise ValueError("RGB and SAR fused features must have identical shapes")
        batch, _, height, width = rgb.shape
        locations = height * width
        neighbours = self.window_size**2
        query = F.normalize(self.query(rgb).float(), dim=1).flatten(2)
        key = F.normalize(self.key(sar).float(), dim=1)
        key_patches = F.unfold(
            key, self.window_size, padding=self.window_size // 2
        ).view(batch, key.shape[1], neighbours, locations)
        scores = torch.einsum("bcl,bckl->bkl", query, key_patches) / self.temperature
        valid = F.unfold(
            torch.ones((batch, 1, height, width), device=rgb.device, dtype=scores.dtype),
            self.window_size,
            padding=self.window_size // 2,
        ).view(batch, neighbours, locations)
        weights = torch.softmax(scores.masked_fill(valid < 0.5, -1e4), dim=1)
        raw_value = self.value(sar).float()
        value_patches = F.unfold(
            raw_value, self.window_size, padding=self.window_size // 2
        ).view(batch, raw_value.shape[1], neighbours, locations)
        matched = torch.einsum("bkl,bckl->bcl", weights, value_patches)
        matched = matched.view(batch, raw_value.shape[1], height, width).to(rgb.dtype)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)
        entropy = entropy / valid.sum(1).clamp_min(2.0).log()
        confidence = (1.0 - entropy).clamp(0.0, 1.0).view(batch, 1, height, width)
        update = self.update(torch.cat((rgb, matched), dim=1))
        feature = parent_fused + torch.tanh(self.gain) * confidence.to(update.dtype) * update
        offset = torch.einsum(
            "bkl,kd->bdl", weights, self.offset_table.to(weights)
        ).view(batch, 2, height, width)
        return InteractionOutput(
            feature=feature,
            confidence=confidence.to(rgb.dtype),
            entropy=entropy.view(batch, 1, height, width).to(rgb.dtype),
            offset=offset.to(rgb.dtype),
        )


class HierarchicalOccurrenceSeverityDecoder(nn.Module):
    """Conditional B/A/X decoder blended safely with the parent's flat logits."""

    def __init__(self, channels: int = 256, hidden: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.building = nn.Conv2d(hidden, 1, 1)
        self.affected = nn.Conv2d(hidden, 1, 1)
        self.destroyed = nn.Conv2d(hidden, 1, 1)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor, parent_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(feature)
        building = self.building(shared)
        affected = self.affected(shared)
        destroyed = self.destroyed(shared)
        log_b = F.logsigmoid(building)
        log_not_b = F.logsigmoid(-building)
        log_a = F.logsigmoid(affected)
        log_not_a = F.logsigmoid(-affected)
        log_x = F.logsigmoid(destroyed)
        log_not_x = F.logsigmoid(-destroyed)
        hierarchical = torch.cat(
            (log_not_b, log_b + log_not_a, log_b + log_a + log_not_x, log_b + log_a + log_x),
            dim=1,
        )
        output = parent_logits + torch.tanh(self.gain) * (hierarchical - parent_logits.detach())
        return {
            "logits": output,
            "hierarchical_logits": hierarchical,
            "building_logits": building,
            "affected_logits": affected,
            "destroyed_logits": destroyed,
        }


__all__ = [
    "HierarchicalOccurrenceSeverityDecoder",
    "InteractionOutput",
    "RGBStructureAdapter",
    "RegistrationTolerantInteraction",
    "SARScatteringFrequencyExperts",
]
