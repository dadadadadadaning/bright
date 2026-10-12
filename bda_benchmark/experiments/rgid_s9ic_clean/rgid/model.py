from __future__ import annotations

import torch
import torch.nn as nn


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = 8 if out_channels % 8 == 0 else 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class RGIDHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        num_shared_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_shared_blocks < 1:
            raise ValueError("num_shared_blocks must be at least one")
        blocks = [ConvGNAct(in_channels, hidden_channels)]
        blocks.extend(
            ConvGNAct(hidden_channels, hidden_channels)
            for _ in range(num_shared_blocks - 1)
        )
        self.shared = nn.Sequential(*blocks)
        self.center_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.center_head[-1].bias, -2.19)
        nn.init.constant_(self.boundary_head[-1].bias, 0.0)

    def forward(self, fused_roi: torch.Tensor):
        if fused_roi.ndim != 4:
            raise ValueError(
                f"fused_roi must be [B,C,H,W], got {tuple(fused_roi.shape)}"
            )
        feature = self.shared(fused_roi)
        return {
            "center_logits": self.center_head(feature),
            "boundary_logits": self.boundary_head(feature),
        }
