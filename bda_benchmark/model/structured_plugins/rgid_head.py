from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from bda_benchmark.experiments.rgid_s9ic_clean.rgid.config import RGIDConfig
from bda_benchmark.experiments.rgid_s9ic_clean.rgid.model import (
    ConvGNAct,
    RGIDHead,
)

from .blocks import StripSpatialBlock


class StripRGIDHead(nn.Module):
    """RGID center/boundary head with axial building-shape context."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        num_shared_blocks: int = 2,
        strip_kernel: int = 19,
    ) -> None:
        super().__init__()
        if num_shared_blocks < 1:
            raise ValueError("num_shared_blocks must be at least one")
        shared = [ConvGNAct(in_channels, hidden_channels)]
        shared.extend(
            StripSpatialBlock(hidden_channels, strip_kernel)
            for _ in range(num_shared_blocks)
        )
        self.shared = nn.Sequential(*shared)
        self.center_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.center_head[-1].bias, -2.19)
        nn.init.constant_(self.boundary_head[-1].bias, 0.0)

    def forward(self, fused_roi: torch.Tensor) -> Dict[str, torch.Tensor]:
        if fused_roi.ndim != 4:
            raise ValueError("fused_roi must be [B,C,H,W]")
        feature = self.shared(fused_roi)
        return {
            "center_logits": self.center_head(feature),
            "boundary_logits": self.boundary_head(feature),
        }


def build_rgid_head(
    head_type: str,
    config: RGIDConfig,
    strip_kernel: int = 19,
) -> nn.Module:
    common = dict(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_shared_blocks=config.num_shared_blocks,
    )
    if head_type == "baseline":
        return RGIDHead(**common)
    if head_type == "strip":
        return StripRGIDHead(**common, strip_kernel=strip_kernel)
    raise ValueError(f"unknown RGID head_type: {head_type}")


def load_rgid_head(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> Tuple[nn.Module, RGIDConfig, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = RGIDConfig.from_dict(payload["config"])
    head_type = str(payload.get("head_type", "baseline"))
    strip_kernel = int(payload.get("strip_kernel", 19))
    head = build_rgid_head(head_type, config, strip_kernel=strip_kernel)
    state = payload.get("head_state_dict", payload.get("model_state_dict"))
    if state is None:
        raise KeyError("checkpoint has no RGID head state_dict")
    head.load_state_dict(state, strict=True)
    head.to(device).eval()
    return head, config, payload
