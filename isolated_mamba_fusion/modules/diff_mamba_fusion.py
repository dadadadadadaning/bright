"""Difference-aware Mamba-style fusion for RGB/SAR feature maps."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_fusion import BaseFusion
from .lightweight_vss_adapter import LightweightVSSAdapter


def _init_raw_gamma(channels: int, gamma_init: float, gamma_scale: float) -> torch.Tensor:
    """Initialize raw gamma so effective scale ~= gamma_init.

    Effective scale is:

        scale = gamma_scale * tanh(raw_gamma)

    Therefore, when gamma_init is non-zero we initialize raw_gamma with
    atanh(gamma_init / gamma_scale).  Values are clamped to avoid infinities.
    """

    if gamma_scale <= 0:
        raise ValueError("gamma_scale must be positive.")
    ratio = max(min(float(gamma_init) / float(gamma_scale), 0.999), -0.999)
    raw = math.atanh(ratio)
    return torch.full((1, channels, 1, 1), raw, dtype=torch.float32)


def resize_roi_mask(
    roi_mask: Optional[torch.Tensor],
    target_hw: tuple[int, int],
    like: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Resize and clamp ROI mask to [0, 1]."""

    if roi_mask is None:
        return None
    if roi_mask.ndim != 4 or roi_mask.shape[1] != 1:
        raise ValueError("roi_mask must have shape [B, 1, H, W].")
    if like is not None:
        roi_mask = roi_mask.to(device=like.device, dtype=like.dtype)
    elif not torch.is_floating_point(roi_mask):
        roi_mask = roi_mask.float()
    if roi_mask.shape[-2:] != target_hw:
        roi_mask = F.interpolate(roi_mask, size=target_hw, mode="bilinear", align_corners=False)
    return roi_mask.clamp(0.0, 1.0)


class DiffMambaFusion(nn.Module):
    """Fuse RGB and SAR features using explicit difference cues.

    Inputs:
        F_rgb: [B, C, H, W]
        F_sar: [B, C, H, W]

    Core cues:
        F_diff = abs(F_rgb - F_sar)
        F_mul  = F_rgb * F_sar
        F_cat  = concat(F_rgb, F_sar, F_diff, F_mul)

    Output:
        F_out = F_base + scale * G_roi * F_mamba

    ``gamma`` is per-channel and bounded:

        scale = gamma_scale * tanh(gamma)

    The bound keeps the RGB-SAR difference branch from overwhelming original
    features during early training.
    """

    def __init__(
        self,
        channels: int,
        gamma_init: float = 0.0,
        gamma_scale: float = 0.1,
        backend: str = "auto",
        base_mode: str = "rgb",
        enable: bool = True,
        bg_alpha: float = 0.05,
        fallback_kernel_size: int = 3,
        fallback_expansion: float = 1.0,
        fallback_norm_type: str = "bn",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.enable = enable
        self.gamma_scale = float(gamma_scale)
        self.bg_alpha = float(bg_alpha)
        self.base_fusion = BaseFusion(channels=channels, base_mode=base_mode)
        self.proj_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.proj_sar = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.cross_proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.mamba_block = LightweightVSSAdapter(
            channels=channels,
            backend=backend,
            fallback_kernel_size=fallback_kernel_size,
            fallback_expansion=fallback_expansion,
            fallback_norm_type=fallback_norm_type,
        )
        self.gamma = nn.Parameter(_init_raw_gamma(channels, gamma_init, gamma_scale))
        if not self.enable:
            self._freeze_enhancement_branch()

    def _freeze_enhancement_branch(self) -> None:
        """Freeze parameters unused when enable=False."""

        for module in (self.proj_rgb, self.proj_sar, self.cross_proj, self.mamba_block):
            for param in module.parameters():
                param.requires_grad_(False)
        self.gamma.requires_grad_(False)

    @staticmethod
    def _check_inputs(f_rgb: torch.Tensor, f_sar: torch.Tensor) -> None:
        BaseFusion.check_inputs(f_rgb, f_sar)

    def scale(self) -> torch.Tensor:
        """Return bounded per-channel residual scale, shape [1, C, 1, 1]."""

        return self.gamma_scale * torch.tanh(self.gamma)

    def base_feature(self, f_rgb: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        return self.base_fusion.base_feature(f_rgb, f_sar)

    def roi_gate(self, roi_mask: Optional[torch.Tensor], target_hw: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
        mask = resize_roi_mask(roi_mask, target_hw, like=like)
        if mask is None:
            return torch.ones((like.shape[0], 1, target_hw[0], target_hw[1]), device=like.device, dtype=like.dtype)
        return mask + self.bg_alpha * (1.0 - mask)

    def forward_mamba(self, f_rgb: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        """Return the difference-enhanced feature before residual scaling."""

        self._check_inputs(f_rgb, f_sar)
        z_rgb = self.proj_rgb(f_rgb)
        z_sar = self.proj_sar(f_sar)
        f_diff = torch.abs(z_rgb - z_sar)
        f_mul = z_rgb * z_sar
        f_cross = self.cross_proj(torch.cat([z_rgb, z_sar, f_diff, f_mul], dim=1))
        return self.mamba_block(f_cross)

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_sar: torch.Tensor,
        g_exist: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        cur_iter: Optional[int] = None,
    ) -> torch.Tensor:
        del g_exist, cur_iter
        self._check_inputs(f_rgb, f_sar)
        f_base = self.base_feature(f_rgb, f_sar)
        if not self.enable:
            return f_base
        f_mamba = self.forward_mamba(f_rgb, f_sar)
        gate = self.roi_gate(roi_mask, target_hw=f_rgb.shape[-2:], like=f_rgb)
        scale = self.scale().to(device=f_mamba.device, dtype=f_mamba.dtype)
        return f_base + scale * gate * f_mamba
