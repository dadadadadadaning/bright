from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class FrozenFeatureSource(nn.Module, ABC):
    """Common contract for read-only SAR feature sources."""

    output_channels: Dict[str, int]

    def train(self, mode: bool = True):
        # A feature audit must never silently update BN/dropout statistics.
        super().train(False)
        return self

    @abstractmethod
    def forward(self, post_sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


def raw_display_channel(post_sar: torch.Tensor) -> torch.Tensor:
    """Return an unstandardised SAR display channel in [0,255].

    BRIGHT loaders used in this repository retain the raw display image as the
    second channel.  A one-channel tensor is accepted only when it is already
    non-negative display data; a standardised one-channel tensor is rejected.
    """

    if post_sar.ndim != 4:
        raise ValueError(f"post_sar must be BCHW, got {tuple(post_sar.shape)}")
    if post_sar.shape[1] >= 2:
        raw = post_sar[:, 1:2]
    elif float(post_sar.detach().amin()) >= 0.0:
        raw = post_sar[:, :1]
    else:
        raise ValueError(
            "SAR-DINO needs the raw display channel; received only a standardised channel"
        )
    return raw.float().clamp_(0.0, 255.0)
