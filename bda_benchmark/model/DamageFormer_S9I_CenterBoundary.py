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
            raise RuntimeError(
                f"Requested feature {requested_key!r} is unavailable. "
                f"Aux keys={sorted(aux.keys())}"
            )
        return requested_key, value

    for key in FEATURE_PRIORITY:
        value = aux.get(key)
        if torch.is_tensor(value) and value.ndim == 4:
            return key, value

    tensors = {
        key: tuple(value.shape)
        for key, value in aux.items()
        if torch.is_tensor(value) and value.ndim == 4
    }
    raise RuntimeError(
        "No supported four-dimensional S8 feature found. "
        f"Candidates={tensors}"
    )


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
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
    def __init__(self, in_channels: int, hidden_channels: int = 64):
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
                f"Expected [B,{self.in_channels},H,W], got {tuple(feature.shape)}"
            )
        shared = self.shared(feature)
        return {
            "center_logits": self.center_head(shared),
            "boundary_logits": self.boundary_head(shared),
        }
