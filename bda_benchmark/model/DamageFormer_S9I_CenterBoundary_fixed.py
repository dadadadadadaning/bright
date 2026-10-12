# -*- coding: utf-8 -*-
"""S9-I lightweight center/boundary geometry head.

The accepted S8/P3 network remains external and frozen. Only this head is
trained. A deterministic feature selector reads one spatial tensor from S8's
`return_aux=True` output.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


# The user's real S8/P3 aux output contains:
#   p_building : [B, 1, H, W]
#   roi_weight : [B, 1, H, W]
#   rgb_feature: [B, 256, H, W]
#   sar_feature: [B, 256, H, W]
#   fused_roi  : [B, 256, H, W]
#
# For instance geometry, fused_roi is preferred because it is the fused
# RGB-SAR representation already modulated by the building ROI prior.
FEATURE_PRIORITY = (
    "fused_roi",
    "rgb_feature",
    "sar_feature",
    # Compatibility with possible alternate S8 branches.
    "final_feat_fused_roi",
    "damage_feature",
    "final_feat_fused",
    "final_feat_2_roi",
    "metric_embedding",
)


def _valid_spatial_feature(value: Any) -> bool:
    """Return True only for multi-channel [B,C,H,W] feature tensors."""
    return (
        torch.is_tensor(value)
        and value.ndim == 4
        and int(value.shape[1]) > 1
    )


def select_geometry_feature(
    aux: Mapping[str, Any],
    requested_key: Optional[str] = None,
) -> Tuple[str, Tensor]:
    """Select one deterministic multi-channel spatial feature.

    One-channel maps such as ``p_building`` and ``roi_weight`` are deliberately
    excluded from automatic selection. They are useful masks/probabilities, but
    do not provide enough semantic capacity for the learnable Center/Boundary
    head.
    """
    if requested_key is not None:
        value = aux.get(requested_key)
        if not _valid_spatial_feature(value):
            shape = (
                tuple(value.shape)
                if torch.is_tensor(value)
                else None
            )
            raise RuntimeError(
                f"Requested feature {requested_key!r} is unavailable or is "
                f"not a multi-channel [B,C,H,W] tensor. shape={shape}; "
                f"aux keys={sorted(aux.keys())}"
            )
        return requested_key, value

    for key in FEATURE_PRIORITY:
        value = aux.get(key)
        if _valid_spatial_feature(value):
            return key, value

    # Deterministic fallback: among all valid multi-channel spatial tensors,
    # prefer the largest channel count, then the lexicographically first key.
    candidates = [
        (str(key), value)
        for key, value in aux.items()
        if _valid_spatial_feature(value)
    ]
    if candidates:
        candidates.sort(
            key=lambda item: (
                -int(item[1].shape[1]),
                item[0],
            )
        )
        return candidates[0]

    all_spatial = {
        key: tuple(value.shape)
        for key, value in aux.items()
        if torch.is_tensor(value) and value.ndim == 4
    }
    raise RuntimeError(
        "No usable multi-channel four-dimensional S8 feature found. "
        f"Four-dimensional aux tensors={all_spatial}"
    )


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class CenterBoundaryGeometryHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)

        self.shared = nn.Sequential(
            ConvGNAct(in_channels, hidden_channels, 3),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            ConvGNAct(hidden_channels, hidden_channels, 1),
        )
        self.center_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self._reset()

    def _reset(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, -2.0)

    def forward(self, feature: Tensor):
        if feature.ndim != 4 or feature.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Expected [B,{self.in_channels},H,W], "
                f"got {tuple(feature.shape)}"
            )
        shared = self.shared(feature)
        return {
            "center_logits": self.center_head(shared),
            "boundary_logits": self.boundary_head(shared),
        }
