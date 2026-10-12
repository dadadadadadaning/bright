from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_diff import _group_count, _initial_transition


class _Expert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ModalitySpecificMoELite(nn.Module):
    """C2: one RGB, one SAR and one common expert; no top-k routing."""

    def __init__(self, rgb_channels: int, sar_channels: int, width: int = 128, out_channels: int = 256) -> None:
        super().__init__()
        self.rgb_project = nn.Conv2d(rgb_channels, width, 1)
        self.sar_project = nn.Conv2d(sar_channels, width, 1)
        self.rgb_expert = _Expert(width)
        self.sar_expert = _Expert(width)
        self.common_expert = _Expert(width)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * width, 3))
        self.output = nn.Conv2d(width, out_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.transition = nn.Parameter(_initial_transition())

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sar = F.interpolate(sar, rgb.shape[-2:], mode="bilinear", align_corners=False)
        r = self.rgb_project(rgb)
        s = self.sar_project(sar)
        gate_input = torch.cat((r, s, r - s, (r - s).abs()), dim=1)
        weights = self.gate(gate_input).softmax(-1)
        experts = torch.stack((self.rgb_expert(r), self.sar_expert(s), self.common_expert(0.5 * (r + s))), dim=1)
        mixed = (experts * weights[:, :, None, None, None]).sum(1)
        candidate = self.output(mixed)
        transition = torch.sigmoid(self.transition)
        residual = transition * candidate
        return residual, {
            "transition": transition.reshape(1),
            "expert_weights": weights.detach().mean(0),
            "residual_abs": residual.detach().abs().mean().reshape(1),
        }
