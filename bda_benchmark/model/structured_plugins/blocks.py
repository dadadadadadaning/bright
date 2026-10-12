from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class StripSpatialBlock(nn.Module):
    """Large axial receptive field without changing tensor resolution.

    Intended use: the auxiliary RGID geometry head, not the locked P3 encoder.
    """

    def __init__(self, channels: int, strip_kernel: int = 19) -> None:
        super().__init__()
        if strip_kernel < 3 or strip_kernel % 2 == 0:
            raise ValueError("strip_kernel must be an odd integer >= 3")
        pad = strip_kernel // 2
        self.local = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            (1, strip_kernel),
            padding=(0, pad),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            (strip_kernel, 1),
            padding=(pad, 0),
            groups=channels,
            bias=False,
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.local(x)
        residual = self.horizontal(residual)
        residual = self.vertical(residual)
        residual = self.norm(self.project(residual))
        return x + self.act(residual)


class CrossScaleBoundaryFusion(nn.Module):
    """Inject lower-stage boundary detail into a high-level feature safely.

    ``gain`` starts at zero, so construction is exactly equivalent to returning
    ``high``.  This property allows a checkpoint-compatible controlled ablation.
    """

    def __init__(self, high_channels: int, detail_channels: int) -> None:
        super().__init__()
        self.detail_projection = nn.Sequential(
            nn.Conv2d(detail_channels, high_channels, 1, bias=False),
            nn.GroupNorm(_groups(high_channels), high_channels),
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(high_channels * 2, high_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(high_channels), high_channels),
            nn.GELU(),
            nn.Conv2d(high_channels, 1, 1),
        )
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        high: torch.Tensor,
        detail: torch.Tensor,
        boundary_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        detail = self.detail_projection(detail)
        detail = F.interpolate(
            detail, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        gate = torch.sigmoid(self.boundary_gate(torch.cat((high, detail), dim=1)))
        if boundary_prior is not None:
            boundary_prior = F.interpolate(
                boundary_prior.float(),
                size=high.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            gate = gate * boundary_prior.clamp(0.0, 1.0)
        return high + torch.tanh(self.gain) * gate * detail


@dataclass
class LocalMatchOutput:
    matched_sar: torch.Tensor
    residual: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    offset: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "matched_sar": self.matched_sar,
            "residual": self.residual,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "offset": self.offset,
        }


class LocalCrossModalMatcher(nn.Module):
    """Parameter-free soft matching inside a small SAR neighbourhood.

    It creates evidence only.  It does not overwrite P3 features or logits.
    """

    def __init__(self, window_size: int = 5, temperature: float = 0.07) -> None:
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.window_size = window_size
        self.temperature = temperature

        radius = window_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.register_buffer(
            "offset_table",
            torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).float(),
            persistent=False,
        )

    def forward(
        self, rgb_feature: torch.Tensor, sar_feature: torch.Tensor
    ) -> LocalMatchOutput:
        if rgb_feature.shape != sar_feature.shape or rgb_feature.ndim != 4:
            raise ValueError("RGB and SAR features must share [B,C,H,W]")
        batch, channels, height, width = rgb_feature.shape
        kernel = self.window_size
        locations = height * width

        query = F.normalize(rgb_feature.float(), dim=1, eps=1e-6)
        key = F.normalize(sar_feature.float(), dim=1, eps=1e-6)
        patches = F.unfold(key, kernel_size=kernel, padding=kernel // 2)
        patches = patches.view(batch, channels, kernel * kernel, locations)
        scores = torch.einsum("bcl,bckl->bkl", query.flatten(2), patches)

        valid = F.unfold(
            torch.ones(
                (batch, 1, height, width), device=scores.device, dtype=scores.dtype
            ),
            kernel_size=kernel,
            padding=kernel // 2,
        ).view(batch, kernel * kernel, locations)
        scores = scores.masked_fill(valid < 0.5, -1e4)
        weights = torch.softmax(scores / self.temperature, dim=1)

        raw_patches = F.unfold(
            sar_feature.float(), kernel_size=kernel, padding=kernel // 2
        ).view(batch, channels, kernel * kernel, locations)
        matched = torch.einsum("bkl,bckl->bcl", weights, raw_patches)
        matched = matched.view(batch, channels, height, width)

        entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=1)
        normalizer = valid.sum(dim=1).clamp_min(2.0).log()
        entropy = (entropy / normalizer).clamp(0.0, 1.0)
        top2 = scores.topk(k=min(2, scores.shape[1]), dim=1).values
        margin = top2[:, 0] if top2.shape[1] == 1 else top2[:, 0] - top2[:, 1]
        confidence = (1.0 - entropy) * torch.sigmoid(margin / self.temperature)

        offset_table = self.offset_table.to(device=weights.device, dtype=weights.dtype)
        offset = torch.einsum("bkl,kd->bdl", weights, offset_table)
        offset = offset.view(batch, 2, height, width)
        residual = 1.0 - F.cosine_similarity(
            rgb_feature.float(), matched, dim=1, eps=1e-6
        ).unsqueeze(1)

        return LocalMatchOutput(
            matched_sar=matched.to(sar_feature.dtype),
            residual=residual.to(rgb_feature.dtype),
            confidence=confidence.view(batch, 1, height, width).to(rgb_feature.dtype),
            entropy=entropy.view(batch, 1, height, width).to(rgb_feature.dtype),
            offset=offset.to(rgb_feature.dtype),
        )


class RegionRelationBlock(nn.Module):
    """One geometry-biased attention layer for region/superpixel tokens."""

    def __init__(self, channels: int, num_heads: int = 4, spatial_tau: float = 0.25):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.spatial_tau = spatial_tau
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or coordinates.shape[:2] != tokens.shape[:2]:
            raise ValueError("tokens must be [B,N,C] and coordinates [B,N,2]")
        batch, count, _ = tokens.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, count), dtype=torch.bool, device=tokens.device
            )
        qkv = self.qkv(self.norm(tokens)).reshape(
            batch, count, 3, self.num_heads, self.head_dim
        )
        q, k, value = qkv.unbind(dim=2)
        logits = torch.einsum("bnhd,bmhd->bhnm", q, k) / self.head_dim**0.5
        distance = torch.cdist(coordinates.float(), coordinates.float(), p=2)
        logits = logits - distance[:, None] / max(self.spatial_tau, 1e-6)
        logits = logits.masked_fill(~valid_mask[:, None, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid_mask[:, None, :, None].to(weights.dtype)
        update = torch.einsum("bhnm,bmhd->bnhd", weights, value)
        update = self.output(update.reshape(batch, count, self.channels))
        update = update * valid_mask[..., None].to(update.dtype)
        return tokens + torch.tanh(self.gain) * update


class AffectedOccurrenceResidual(nn.Module):
    """Safely add occurrence evidence without changing severity ordering.

    BRIGHT class order is assumed to be Background, Intact, Damaged, Destroyed.
    The same delta is added to Damaged and Destroyed, therefore their pairwise
    logit difference is mathematically preserved.
    """

    def __init__(self, maximum_shift: float = 1.0) -> None:
        super().__init__()
        self.maximum_shift = float(maximum_shift)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        logits: torch.Tensor,
        affected_evidence: torch.Tensor,
        building_support: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 4:
            raise ValueError("logits must be [B,4,H,W]")
        if affected_evidence.shape[1] != 1 or building_support.shape[1] != 1:
            raise ValueError("evidence and support must be single-channel maps")
        delta = (
            self.maximum_shift
            * torch.tanh(self.gain)
            * torch.tanh(affected_evidence)
            * building_support.detach().clamp(0.0, 1.0)
        )
        output = logits.clone()
        output[:, 2:4] = output[:, 2:4] + delta
        return output
