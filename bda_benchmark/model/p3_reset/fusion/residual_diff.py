from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _initial_transition(probability: float = 0.01) -> torch.Tensor:
    value = torch.tensor(float(probability)).clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(value)


class SafeDifferenceResidual(nn.Module):
    """C1: modality-specific projections plus an exactly safe residual.

    The comparison uses [R,S,R-S,|R-S|].  It intentionally avoids FCA,
    optical flow, Conv3D and correlation matching: earlier controlled audits
    already rejected those assumptions for this dataset.
    """

    def __init__(self, rgb_channels: int, sar_channels: int, width: int = 128, out_channels: int = 256) -> None:
        super().__init__()
        self.rgb_project = nn.Sequential(
            nn.Conv2d(rgb_channels, width, 1, bias=False),
            nn.GroupNorm(_group_count(width), width),
        )
        self.sar_project = nn.Sequential(
            nn.Conv2d(sar_channels, width, 1, bias=False),
            nn.GroupNorm(_group_count(width), width),
        )
        self.body = nn.Sequential(
            nn.Conv2d(4 * width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        # The output layer is exactly zero, while a small positive gate lets
        # that layer receive a gradient on the first optimiser step.
        self.transition = nn.Parameter(_initial_transition())

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sar = F.interpolate(sar, rgb.shape[-2:], mode="bilinear", align_corners=False)
        r = self.rgb_project(rgb)
        s = self.sar_project(sar)
        relation = torch.cat((r, s, r - s, (r - s).abs()), dim=1)
        candidate = self.body(relation)
        gate = torch.sigmoid(self.transition)
        residual = gate * candidate
        return residual, {
            "transition": gate.reshape(1),
            "candidate_abs": candidate.detach().abs().mean().reshape(1),
            "residual_abs": residual.detach().abs().mean().reshape(1),
        }
