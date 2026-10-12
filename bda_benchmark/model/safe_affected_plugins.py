from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _filter(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    weight = kernel.to(device=x.device, dtype=x.dtype).expand(channels, 1, -1, -1)
    return F.conv2d(x, weight, padding=kernel.shape[-1] // 2, groups=channels)


class RGBInputStructureStem(nn.Module):
    """High-resolution RGB roof/boundary evidence, independent of P3 geometry."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        scharr = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]
        ).div(32.0)
        self.register_buffer("scharr_x", scharr[None, None])
        self.register_buffer("scharr_y", scharr.t().contiguous()[None, None])
        self.register_buffer(
            "log_kernel",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[
                None, None
            ],
        )

    def forward(self, rgb: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        gray = rgb.float().mean(1, keepdim=True)
        gray = F.interpolate(gray, size=size, mode="bilinear", align_corners=False)
        gx = _filter(gray, self.scharr_x)
        gy = _filter(gray, self.scharr_y)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        laplacian = _filter(gray, self.log_kernel)
        return self.encoder(torch.cat((gray, gx, gy, magnitude, laplacian), dim=1))


class SARInputTextureStem(nn.Module):
    """Robust shallow SAR evidence without assuming positive/raw amplitude units."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.register_buffer(
            "dx", torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])[None, None]
        )
        self.register_buffer("dy", self.dx.transpose(-1, -2).contiguous())
        self.register_buffer(
            "laplacian",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[
                None, None
            ],
        )

    def forward(self, sar: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        value = sar.float().mean(1, keepdim=True)
        value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
        mean3 = F.avg_pool2d(value, 3, stride=1, padding=1)
        mean7 = F.avg_pool2d(value, 7, stride=1, padding=3)
        mad3 = (value - mean3).abs()
        mad7 = (value - mean7).abs()
        gx = _filter(value, self.dx).abs()
        gy = _filter(value, self.dy).abs()
        high = _filter(value, self.laplacian).abs()
        return self.encoder(torch.cat((value, mean3, mad3, mean7, mad7, gx + gy, high), dim=1))


@dataclass
class LocalEvidence:
    matched: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    offset: torch.Tensor


class RGBAnchoredLocalEvidence(nn.Module):
    def __init__(self, channels: int = 32, match_channels: int = 16, window: int = 3) -> None:
        super().__init__()
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be a positive odd number")
        self.window = window
        self.query = nn.Conv2d(channels, match_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, match_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        radius = window // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij"
        )
        self.register_buffer(
            "offsets", torch.stack((xx.flatten(), yy.flatten()), dim=1).float(), persistent=False
        )

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> LocalEvidence:
        batch, _, height, width = rgb.shape
        locations, neighbours = height * width, self.window**2
        query = F.normalize(self.query(rgb).float(), dim=1).flatten(2)
        key = F.normalize(self.key(sar).float(), dim=1)
        patches = F.unfold(key, self.window, padding=self.window // 2)
        patches = patches.view(batch, key.shape[1], neighbours, locations)
        score = torch.einsum("bcl,bckl->bkl", query, patches) / math.sqrt(key.shape[1])
        valid = F.unfold(
            torch.ones((batch, 1, height, width), device=rgb.device, dtype=score.dtype),
            self.window,
            padding=self.window // 2,
        ).view(batch, neighbours, locations)
        weight = torch.softmax(score.masked_fill(valid < 0.5, -1e4), dim=1)
        value = self.value(sar).float()
        values = F.unfold(value, self.window, padding=self.window // 2)
        values = values.view(batch, value.shape[1], neighbours, locations)
        matched = torch.einsum("bkl,bckl->bcl", weight, values)
        matched = matched.view(batch, value.shape[1], height, width).to(rgb.dtype)
        entropy = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(1)
        entropy = entropy / valid.sum(1).clamp_min(2).log()
        confidence = (1.0 - entropy).clamp(0, 1).view(batch, 1, height, width)
        offset = torch.einsum("bkl,kd->bdl", weight, self.offsets.to(weight))
        return LocalEvidence(
            matched, confidence.to(rgb.dtype), entropy.view(batch, 1, height, width).to(rgb.dtype),
            offset.view(batch, 2, height, width).to(rgb.dtype),
        )


class ShallowAffectedEvidence(nn.Module):
    """Build occurrence evidence; never writes into the parent's 4-class feature."""

    def __init__(
        self,
        hidden: int = 32,
        use_rgb: bool = True,
        use_sar: bool = True,
        local_match: bool = True,
        confidence_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_rgb = bool(use_rgb)
        self.use_sar = bool(use_sar)
        self.local_match = bool(local_match)
        self.confidence_gate = bool(confidence_gate)
        self.rgb = RGBInputStructureStem(hidden)
        self.sar = SARInputTextureStem(hidden)
        self.matcher = RGBAnchoredLocalEvidence(hidden, max(hidden // 2, 4), 3)
        self.head = nn.Sequential(
            nn.Conv2d(hidden * 4 + 1, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, rgb_input: torch.Tensor, sar_input: torch.Tensor, size: tuple[int, int]):
        rgb = self.rgb(rgb_input, size)
        sar = self.sar(sar_input, size)
        if not self.use_rgb:
            rgb = torch.zeros_like(rgb)
        if not self.use_sar:
            sar = torch.zeros_like(sar)
        if self.local_match and self.use_rgb and self.use_sar:
            local = self.matcher(rgb, sar)
        else:
            ones = torch.ones_like(rgb[:, :1])
            local = LocalEvidence(sar, ones, torch.zeros_like(ones), torch.zeros_like(rgb[:, :2]))
        relation = (rgb - local.matched).abs()
        direct = (rgb - sar).abs()
        logits = self.head(torch.cat((rgb, sar, direct, relation, local.confidence), dim=1))
        gate = local.confidence if self.confidence_gate else torch.ones_like(local.confidence)
        return logits, gate, {
            "rgb_structure": rgb,
            "sar_texture": sar,
            "match_confidence": local.confidence,
            "match_entropy": local.entropy,
            "match_offset": local.offset,
        }


class SafeAffectedResidual(nn.Module):
    """Non-negative occurrence residual preserving Damaged/Destroyed ordering."""

    def __init__(self, maximum_shift: float = 4.0) -> None:
        super().__init__()
        self.maximum_shift = float(maximum_shift)
        self.gain = nn.Parameter(torch.zeros(()))

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.gain.clamp_(0.0, 1.0)

    def forward(
        self,
        parent_logits: torch.Tensor,
        occurrence_logits: torch.Tensor,
        support: torch.Tensor,
        confidence: torch.Tensor,
    ):
        gain = self.gain.clamp(0.0, 1.0)
        delta = gain * self.maximum_shift * torch.sigmoid(occurrence_logits) * support * confidence
        output = parent_logits.clone()
        output[:, 2:4] = output[:, 2:4] + delta
        return output, delta


__all__ = [
    "RGBAnchoredLocalEvidence", "RGBInputStructureStem", "SARInputTextureStem",
    "SafeAffectedResidual", "ShallowAffectedEvidence",
]
