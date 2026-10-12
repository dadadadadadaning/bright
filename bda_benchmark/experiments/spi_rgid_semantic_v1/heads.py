from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _center_branch(in_channels: int, hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
        nn.GroupNorm(1, hidden_channels),
        nn.GELU(),
        nn.Conv2d(hidden_channels, 1, 1),
    )


class MultiScaleCenterHead(nn.Module):
    """R0/R1/R2 center-head ablation with an explicit shared output grid.

    `low` is R0, `high` is R1, and `dual` is R2. In dual mode alpha is
    learned but initialized to `alpha`; both logits are fused at high-feature
    resolution before decoding.
    """

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        *,
        hidden_channels: int = 64,
        mode: str = "dual",
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in {"low", "high", "dual"}:
            raise ValueError("mode must be low|high|dual")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        self.mode = mode
        self.low_branch = (
            _center_branch(low_channels, hidden_channels) if mode != "high" else None
        )
        self.high_branch = (
            _center_branch(high_channels, hidden_channels) if mode != "low" else None
        )
        self.alpha_logit = (
            nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))
            if mode == "dual"
            else None
        )

    def forward(
        self, low_feature: torch.Tensor, high_feature: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if low_feature.ndim != 4 or high_feature.ndim != 4:
            raise ValueError("center features must be [B,C,H,W]")
        if low_feature.shape[0] != high_feature.shape[0]:
            raise ValueError("low/high features must have the same batch size")
        target_size = high_feature.shape[-2:]
        low_logits = self.low_branch(low_feature) if self.low_branch is not None else None
        high_logits = (
            self.high_branch(high_feature) if self.high_branch is not None else None
        )
        if self.mode == "low":
            fused = F.interpolate(
                low_logits, size=target_size, mode="bilinear", align_corners=False
            )
            alpha = fused.new_tensor(1.0)
        elif self.mode == "high":
            fused = high_logits
            alpha = fused.new_tensor(0.0)
        else:
            low_aligned = F.interpolate(
                low_logits, size=target_size, mode="bilinear", align_corners=False
            )
            alpha = self.alpha_logit.sigmoid()
            fused = alpha * low_aligned + (1.0 - alpha) * high_logits
        return {
            "center_logits": fused,
            "center_logits_low": low_logits,
            "center_logits_high": high_logits,
            "center_alpha": alpha,
        }


__all__ = ["MultiScaleCenterHead"]
