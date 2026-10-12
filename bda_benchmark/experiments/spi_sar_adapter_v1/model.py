from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SafeFrequencyStripAttention(nn.Module):
    """FSA transfer with exact identity initialization and bounded mixing.

    The local low/high and horizontal/vertical decomposition follows the local
    2024 FSA implementation.  The bounded coefficients are an experiment
    safety modification: at initialization the result is bitwise ``x`` while
    both coefficient tensors receive gradients on the first update.
    """

    def __init__(self, channels: int = 256, kernel_size: int = 7, max_scale: float = 0.25):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.max_scale = float(max_scale)

        shape = (channels, 1, 1)
        self.horizontal_low = nn.Parameter(torch.zeros(shape))
        self.horizontal_high = nn.Parameter(torch.zeros(shape))
        self.vertical_low = nn.Parameter(torch.zeros(shape))
        self.vertical_high = nn.Parameter(torch.zeros(shape))
        self.output_gain = nn.Parameter(torch.zeros(shape))

    def _coefficient(self, value: torch.Tensor) -> torch.Tensor:
        return self.max_scale * torch.tanh(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(x.shape)}")
        pad = self.kernel_size // 2
        horizontal_low = F.avg_pool2d(
            F.pad(x, (pad, pad, 0, 0), mode="reflect"),
            kernel_size=(1, self.kernel_size),
            stride=1,
        )
        horizontal_high = x - horizontal_low
        horizontal = (
            horizontal_high
            + self._coefficient(self.horizontal_low) * horizontal_low
            + self._coefficient(self.horizontal_high) * horizontal_high
        )

        vertical_low = F.avg_pool2d(
            F.pad(horizontal, (0, 0, pad, pad), mode="reflect"),
            kernel_size=(self.kernel_size, 1),
            stride=1,
        )
        vertical_high = horizontal - vertical_low
        strip_delta = (
            vertical_high
            + self._coefficient(self.vertical_low) * vertical_low
            + self._coefficient(self.vertical_high) * vertical_high
        )
        return x + self._coefficient(self.output_gain) * strip_delta


class SafeRectangularContextAdapter(nn.Module):
    """RCM-style rectangular context branch with exact identity initialization.

    GroupNorm is used instead of BatchNorm so the adapter does not introduce
    running-statistic drift under the small per-GPU batches used by BRIGHT.
    """

    def __init__(
        self,
        channels: int = 256,
        band_kernel_size: int = 11,
        mlp_ratio: int = 2,
        groups: int = 32,
        max_scale: float = 0.25,
    ):
        super().__init__()
        if band_kernel_size < 3 or band_kernel_size % 2 == 0:
            raise ValueError("band_kernel_size must be an odd integer >= 3")
        if channels % groups != 0:
            raise ValueError("channels must be divisible by GroupNorm groups")
        hidden = channels * mlp_ratio
        self.channels = int(channels)
        self.max_scale = float(max_scale)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.context_h = nn.Conv2d(
            channels,
            channels,
            (1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=channels,
        )
        self.context_v = nn.Conv2d(
            channels,
            channels,
            (band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=channels,
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.residual_gain = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(x.shape)}")
        local = self.local(x)
        context = F.adaptive_avg_pool2d(x, (x.shape[-2], 1))
        context = context + F.adaptive_avg_pool2d(x, (1, x.shape[-1]))
        attention = torch.sigmoid(self.context_v(self.context_h(context)))
        delta = self.mlp(self.norm(local * attention))
        gain = self.max_scale * torch.tanh(self.residual_gain)
        return x + gain * delta


class SARFeatureAdapter(nn.Module):
    """One switchable adapter; use FSA and RCM as separate pilot variants."""

    MODES = ("none", "fsa", "rcm", "fsa_rcm")

    def __init__(self, mode: str, channels: int = 256):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.fsa = SafeFrequencyStripAttention(channels) if mode in {"fsa", "fsa_rcm"} else None
        self.rcm = SafeRectangularContextAdapter(channels) if mode in {"rcm", "fsa_rcm"} else None

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        output = feature
        if self.fsa is not None:
            output = self.fsa(output)
        if self.rcm is not None:
            output = self.rcm(output)
        return output

    def diagnostics(self) -> Dict[str, torch.Tensor | str]:
        result: Dict[str, torch.Tensor | str] = {"mode": self.mode}
        if self.fsa is not None:
            result["fsa_gain_max"] = (
                self.fsa.max_scale * torch.tanh(self.fsa.output_gain.detach())
            ).abs().max()
        if self.rcm is not None:
            result["rcm_gain_max"] = (
                self.rcm.max_scale * torch.tanh(self.rcm.residual_gain.detach())
            ).abs().max()
        return result


def forward_frozen_p3_with_sar_adapter(
    p3: nn.Module,
    adapter: SARFeatureAdapter,
    pre_data: torch.Tensor,
    post_data: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | str]]:
    """Run a frozen P3 with SAR adaptation only in the pixel damage path.

    The encoders and localization path run without gradients.  The frozen
    fusion/readout operations remain outside ``no_grad`` so gradients can flow
    from the damage loss back to the adapter.  MICB/DSMIL and any memory
    residual are intentionally bypassed in this isolated pilot.
    """

    if getattr(p3, "mil_residual_scale", 0.0) != 0.0:
        raise ValueError("isolated SAR adapter requires mil_residual_scale=0")
    if any(parameter.requires_grad for parameter in p3.parameters()):
        raise ValueError("freeze P3 first with p3.requires_grad_(False)")

    with torch.no_grad():
        rgb_feature, sar_feature = p3._encode_and_fuse(pre_data, post_data)
        loc_logits_feature = p3.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if p3.detach_building_prior else p_building
        roi_weight = p3.roi_alpha + (1.0 - p3.roi_alpha) * building_prior
        base_fused = p3.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        base_damage_logits = p3.clf_2(base_fused * roi_weight)
        base_output_damage = F.interpolate(
            base_damage_logits,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    adapted_sar = adapter(sar_feature.detach())
    fused_feature = p3.fusion_layer_3(torch.cat((rgb_feature.detach(), adapted_sar), dim=1))
    fused_roi = fused_feature * roi_weight.detach()
    damage_logits_feature = p3.clf_2(fused_roi)

    output_loc = F.interpolate(
        loc_logits_feature,
        size=pre_data.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    output_damage = F.interpolate(
        damage_logits_feature,
        size=post_data.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    diagnostics: Dict[str, torch.Tensor | str] = {
        "rgb_feature": rgb_feature,
        "sar_feature": sar_feature,
        "adapted_sar_feature": adapted_sar,
        "p_building": p_building,
        "roi_weight": roi_weight,
        "base_output_damage": base_output_damage,
        **adapter.diagnostics(),
    }
    return output_loc, output_damage, diagnostics
