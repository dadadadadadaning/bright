"""Base RGB/SAR fusion baseline.

This module is intentionally simple:

    F_base = Conv1x1(concat(F_rgb, F_sar))

It does not use difference features, multiplicative features, spatial mixing,
damage-existence gates, or ROI masks.  It is useful as a parameter-count
control baseline: if BaseFusion improves performance, some gain may come from
extra projection capacity rather than the proposed difference/gating logic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BaseFusion(nn.Module):
    """Simple RGB/SAR base fusion with a unified forward interface."""

    def __init__(self, channels: int, base_mode: str = "conv", enable: bool = True) -> None:
        super().__init__()
        if base_mode not in {"conv", "rgb", "sar", "avg"}:
            raise ValueError(f"Unsupported base_mode: {base_mode}")
        self.channels = channels
        self.base_mode = base_mode
        self.enable = enable
        self.base_proj: Optional[nn.Module]
        if base_mode == "conv":
            self.base_proj = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
        else:
            # Avoid unused trainable parameters for rgb/sar/avg no-op modes.
            self.base_proj = None

    @staticmethod
    def check_inputs(f_rgb: torch.Tensor, f_sar: torch.Tensor) -> None:
        if f_rgb.ndim != 4 or f_sar.ndim != 4:
            raise ValueError("F_rgb and F_sar must be 4D tensors: [B, C, H, W].")
        if f_rgb.shape != f_sar.shape:
            raise ValueError(f"F_rgb and F_sar must have the same shape, got {f_rgb.shape} and {f_sar.shape}.")

    def base_feature(self, f_rgb: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        self.check_inputs(f_rgb, f_sar)
        if self.base_mode == "rgb":
            return f_rgb
        if self.base_mode == "sar":
            return f_sar
        if self.base_mode == "avg":
            return 0.5 * (f_rgb + f_sar)
        if self.base_proj is None:
            raise RuntimeError("base_proj is only available when base_mode='conv'.")
        return self.base_proj(torch.cat([f_rgb, f_sar], dim=1))

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_sar: torch.Tensor,
        g_exist: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        cur_iter: Optional[int] = None,
    ) -> torch.Tensor:
        del g_exist, roi_mask, cur_iter
        return self.base_feature(f_rgb, f_sar)
