"""Damage-existence guided RGB/SAR Mamba-style fusion."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_fusion import BaseFusion
from .diff_mamba_fusion import DiffMambaFusion, resize_roi_mask, _init_raw_gamma


class ExistGuidedMambaFusion(nn.Module):
    """Fuse RGB/SAR features under damage-existence guidance.

    Inputs:
        F_rgb:   [B, C, H, W]
        F_sar:   [B, C, H, W]
        G_exist: None, [B, 1, h, w], or [B, 2, h, w]

    Gate interpretation:
        - 1 channel: sigmoid(G_exist)
        - 2 channels: softmax(G_exist, dim=1)[:, 1:2]

    Default output:
        F_base = Conv1x1(concat(F_rgb, F_sar))
        F_out  = F_base + scale * G_final * F_mamba
    """

    def __init__(
        self,
        channels: int,
        gamma_init: float = 0.0,
        gamma_scale: float = 0.1,
        backend: str = "auto",
        base_mode: str = "conv",
        enable: bool = True,
        gate_min: float = 0.1,
        bg_alpha: float = 0.05,
        detach_gate: bool = False,
        gate_warmup: int = 0,
        fallback_norm_type: str = "bn",
    ) -> None:
        super().__init__()
        if base_mode not in {"conv", "rgb", "sar", "avg"}:
            raise ValueError(f"Unsupported base_mode: {base_mode}")

        self.channels = channels
        self.enable = enable
        self.gamma_scale = float(gamma_scale)
        self.gate_min = float(gate_min)
        self.bg_alpha = float(bg_alpha)
        self.detach_gate = detach_gate
        self.gate_warmup = int(gate_warmup)
        self.diff_fusion = DiffMambaFusion(
            channels=channels,
            gamma_init=0.0,
            gamma_scale=gamma_scale,
            backend=backend,
            base_mode=base_mode,
            enable=True,
            bg_alpha=bg_alpha,
            fallback_norm_type=fallback_norm_type,
        )
        self.diff_fusion.gamma.requires_grad_(False)
        for param in self.diff_fusion.base_fusion.parameters():
            param.requires_grad_(False)
        self.base_fusion = BaseFusion(channels=channels, base_mode=base_mode)
        self.gamma = nn.Parameter(_init_raw_gamma(channels, gamma_init, gamma_scale))
        if not self.enable:
            self._freeze_enhancement_branch()

    def _freeze_enhancement_branch(self) -> None:
        """Freeze parameters unused when enable=False."""

        for param in self.diff_fusion.parameters():
            param.requires_grad_(False)
        self.gamma.requires_grad_(False)

    def scale(self) -> torch.Tensor:
        """Return bounded per-channel residual scale, shape [1, C, 1, 1]."""

        return self.gamma_scale * torch.tanh(self.gamma)

    def _exist_prob(
        self,
        g_exist: Optional[torch.Tensor],
        target_hw: tuple[int, int],
        like: torch.Tensor,
        cur_iter: Optional[int] = None,
    ) -> torch.Tensor:
        if self.gate_warmup > 0 and cur_iter is not None and cur_iter < self.gate_warmup:
            return torch.ones((like.shape[0], 1, target_hw[0], target_hw[1]), device=like.device, dtype=like.dtype)
        if g_exist is None:
            return torch.ones((like.shape[0], 1, target_hw[0], target_hw[1]), device=like.device, dtype=like.dtype)
        if g_exist.ndim != 4:
            raise ValueError("G_exist must be a 4D tensor: [B, 1/2, H, W].")
        g_exist = g_exist.to(device=like.device, dtype=like.dtype)
        if g_exist.shape[1] == 1:
            gate = torch.sigmoid(g_exist)
        elif g_exist.shape[1] == 2:
            gate = torch.softmax(g_exist, dim=1)[:, 1:2]
        else:
            raise ValueError(f"G_exist must have 1 or 2 channels, got {g_exist.shape[1]}.")

        if gate.shape[-2:] != target_hw:
            gate = F.interpolate(gate, size=target_hw, mode="bilinear", align_corners=False)
        gate = self.gate_min + (1.0 - self.gate_min) * gate
        if self.detach_gate:
            gate = gate.detach()
        return gate

    def _final_gate(
        self,
        g_exist: Optional[torch.Tensor],
        roi_mask: Optional[torch.Tensor],
        target_hw: tuple[int, int],
        like: torch.Tensor,
        cur_iter: Optional[int],
    ) -> torch.Tensor:
        gate = self._exist_prob(g_exist, target_hw=target_hw, like=like, cur_iter=cur_iter)
        roi = resize_roi_mask(roi_mask, target_hw, like=like)
        if roi is None:
            return gate
        return gate * roi + self.bg_alpha * (1.0 - roi)

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_sar: torch.Tensor,
        g_exist: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        cur_iter: Optional[int] = None,
    ) -> torch.Tensor:
        BaseFusion.check_inputs(f_rgb, f_sar)
        f_base = self.base_fusion.base_feature(f_rgb, f_sar)
        if not self.enable:
            return f_base
        f_mamba = self.diff_fusion.forward_mamba(f_rgb, f_sar)
        gate = self._final_gate(g_exist, roi_mask, target_hw=f_rgb.shape[-2:], like=f_rgb, cur_iter=cur_iter)
        scale = self.scale().to(device=f_mamba.device, dtype=f_mamba.dtype)
        return f_base + scale * gate * f_mamba
