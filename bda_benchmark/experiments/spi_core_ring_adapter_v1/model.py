from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .regions import RegionMaps, broadcast_table, scatter_region_mean


class ParameterMatchedAdapter(nn.Module):
    """One identical residual adapter for M2/M3/M4; only its context changes."""

    def __init__(self, channels: int = 256, hidden: int = 64, classes: int = 4):
        super().__init__()
        self.pixel = nn.Conv2d(channels, hidden, 1)
        self.context = nn.Conv2d(channels, hidden, 1)
        self.difference = nn.Conv2d(channels, hidden, 1)
        self.product = nn.Conv2d(channels, hidden, 1)
        self.contrast = nn.Conv2d(channels, hidden, 1)
        self.mix = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, classes, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pixel, context, contrast, support):
        fused = (
            self.pixel(pixel)
            + self.context(context)
            + self.difference(pixel - context)
            + self.product(pixel * context)
            + self.contrast(contrast)
        )
        return self.out(self.mix(fused)) * support


class DamageReadoutControl(nn.Module):
    def __init__(self, source: nn.Conv2d):
        super().__init__()
        self.readout = nn.Conv2d(
            source.in_channels, source.out_channels, source.kernel_size,
            stride=source.stride, padding=source.padding, bias=source.bias is not None,
        )
        self.readout.load_state_dict(source.state_dict())

    def forward(self, feature):
        return self.readout(feature)


def adapter_inputs(variant: str, feature: torch.Tensor, regions: RegionMaps | None):
    if variant == "m2_local_adapter":
        if regions is None:
            raise ValueError("M2 requires the same LocCC support used by M3/M4")
        context = F.avg_pool2d(feature, 5, stride=1, padding=2)
        contrast = F.avg_pool2d(feature, 3, stride=1, padding=1) - F.avg_pool2d(
            feature, 7, stride=1, padding=3
        )
        support = (regions.labels > 0).unsqueeze(1).to(feature.dtype)
        return context, contrast, support
    if regions is None:
        raise ValueError(f"{variant} requires LocCC region maps")
    region_table, _ = scatter_region_mean(feature, regions.labels, regions.region_count)
    region_context = broadcast_table(region_table, regions.labels)
    support = (regions.labels > 0).unsqueeze(1).to(feature.dtype)
    if variant == "m3_loccc_region":
        return region_context, torch.zeros_like(feature), support
    if variant != "m4_core_ring_region":
        raise ValueError(f"unsupported adapter variant: {variant}")

    core_table, core_counts = scatter_region_mean(
        feature, regions.core_labels, regions.region_count
    )
    ring_table, ring_counts = scatter_region_mean(
        feature, regions.ring_labels, regions.region_count
    )
    no_core = core_counts == 0
    core_table[no_core] = region_table[no_core]
    contrast_table = core_table - ring_table
    contrast_table[ring_counts == 0] = 0
    return (
        broadcast_table(core_table, regions.labels),
        broadcast_table(contrast_table, regions.labels),
        support,
    )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
