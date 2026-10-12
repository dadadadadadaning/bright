"""Fallback blocks used when VMamba / selective_scan is unavailable.

The goal is not to faithfully reproduce VMamba.  Instead, this file provides a
small, dependency-free spatial mixing block with the same input/output shape:

    [B, C, H, W] -> [B, C, H, W]

It is intentionally conservative and safe for CPU smoke tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def safe_group_count(channels: int, max_groups: int = 32) -> int:
    """Return a GroupNorm group count that divides ``channels``."""

    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


def make_norm2d(channels: int, norm_type: str = "bn") -> nn.Module:
    """Create a safe 2D normalization layer.

    ``norm_type="gn"`` automatically chooses a group count that divides
    ``channels``.  This avoids GroupNorm construction errors for arbitrary C.
    """

    norm_type = norm_type.lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        return nn.GroupNorm(safe_group_count(channels), channels)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBNGELU(nn.Sequential):
    """A tiny Conv2d + BatchNorm2d + GELU helper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        norm_type: str = "bn",
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            make_norm2d(out_channels, norm_type=norm_type),
            nn.GELU(),
        )


class FallbackMambaLikeBlock(nn.Module):
    """Lightweight spatial mixing block used as a VMamba/VSS fallback.

    Structure:

        1x1 Conv projection
        depthwise spatial Conv
        1x1 Conv projection
        residual connection with learnable residual scale

    This gives a cheap local spatial mixer that is robust on CPU and does not
    require custom CUDA kernels.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        kernel_size: int = 3,
        residual_scale_init: float = 0.1,
        norm_type: str = "bn",
    ) -> None:
        super().__init__()
        hidden_channels = max(channels, int(channels * expansion))
        self.pre = ConvBNGELU(channels, hidden_channels, kernel_size=1, norm_type=norm_type)
        self.spatial = ConvBNGELU(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            groups=hidden_channels,
            norm_type=norm_type,
        )
        self.post = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            make_norm2d(channels, norm_type=norm_type),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.post(self.spatial(self.pre(x)))
        return self.act(x + self.residual_scale * mixed)
