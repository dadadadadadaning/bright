# -*- coding: utf-8 -*-
"""S9-Ic reliable-geometry head.

The architecture intentionally matches S9-I so that the controlled change is
the center target/loss/decoder, not model capacity. S8 remains external and
frozen. The Boundary Head is trained now and retained for later G2/G3
decoding, but G1 does not use a boundary permission gate or rollback.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


FEATURE_PRIORITY = (
    "fused_roi",
    "rgb_feature",
    "sar_feature",
    "final_feat_fused_roi",
    "damage_feature",
    "final_feat_fused",
    "final_feat_2_roi",
    "metric_embedding",
)


def select_geometry_feature(
    aux: Mapping[str, Any],
    requested_key: Optional[str] = None,
) -> Tuple[str, Tensor]:
    if requested_key is not None:
        value = aux.get(requested_key)
        if not torch.is_tensor(value) or value.ndim != 4:
            candidates = {
                key: tuple(item.shape)
                for key, item in aux.items()
                if torch.is_tensor(item)
            }
            raise RuntimeError(
                f"Requested feature {requested_key!r} is unavailable. "
                f"Candidates={candidates}"
            )
        return requested_key, value

    for key in FEATURE_PRIORITY:
        value = aux.get(key)
        if torch.is_tensor(value) and value.ndim == 4 and value.shape[1] > 1:
            return key, value

    candidates = {
        key: tuple(value.shape)
        for key, value in aux.items()
        if torch.is_tensor(value)
    }
    raise RuntimeError(
        "No supported multi-channel four-dimensional S8 feature found. "
        f"Candidates={candidates}"
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


class ReliableGeometryHead(nn.Module):
    """Center/boundary head; S9-Id optionally adds a two-channel offset branch."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        predict_offset: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.predict_offset = bool(predict_offset)

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
        self.offset_head = (
            nn.Sequential(
                ConvGNAct(hidden_channels, hidden_channels, 3),
                nn.Conv2d(hidden_channels, 2, 1),
            )
            if self.predict_offset
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, -2.0)
        if self.offset_head is not None:
            final_offset = self.offset_head[-1]
            if isinstance(final_offset, nn.Conv2d) and final_offset.bias is not None:
                nn.init.zeros_(final_offset.bias)

    def forward(self, feature: Tensor):
        if feature.ndim != 4 or feature.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Expected [B,{self.in_channels},H,W], "
                f"got {tuple(feature.shape)}"
            )
        shared = self.shared(feature)
        output = {
            "center_logits": self.center_head(shared),
            "boundary_logits": self.boundary_head(shared),
        }
        if self.offset_head is not None:
            output["offset_logits"] = self.offset_head(shared)
        return output


# Explicit alias for later G2/G3 code.
CenterBoundaryGeometryHead = ReliableGeometryHead
