from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.core import (
    ROLE_NAMES,
    pool_role_tokens,
)


SCHARR_X = torch.tensor(
    [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]
) / 32.0
LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
GAUSSIAN_5 = torch.tensor(
    [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=torch.float32,
) / 256.0


def robust_standardize(value: torch.Tensor) -> torch.Tensor:
    """Per-image robust normalisation, independent of labels and event identity."""
    if value.ndim != 4:
        raise ValueError("value must be [N,C,H,W]")
    flat = value.float().flatten(2)
    median = flat.median(2).values[..., None, None]
    mad = (value.float() - median).abs().flatten(2).median(2).values[..., None, None]
    return ((value.float() - median) / (1.4826 * mad).clamp_min(1e-3)).clamp(-8, 8)


def _depthwise(value: torch.Tensor, kernel: torch.Tensor, padding: int) -> torch.Tensor:
    channels = value.shape[1]
    weight = kernel.to(value).reshape(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    return F.conv2d(value, weight, padding=padding, groups=channels)


def fixed_edge_maps(value: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Scharr/LoG structure maps. This branch has no learned parameter."""
    x = robust_standardize(value)
    gx = _depthwise(x, SCHARR_X, 1)
    gy = _depthwise(x, SCHARR_X.t(), 1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
    smooth = _depthwise(x, GAUSSIAN_5, 2)
    log = _depthwise(smooth, LAPLACIAN, 1).abs()
    contrast = (x - smooth).abs()
    maps = torch.cat((gx.abs(), gy.abs(), magnitude, log, contrast), dim=1)
    return robust_standardize(maps), ("scharr_x", "scharr_y", "gradient", "log", "contrast")


def fixed_sar_texture_maps(value: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Robust SAR texture maps without pretending RGB and SAR intensities align."""
    shifted = value.float() - value.float().flatten(2).amin(2)[..., None, None]
    logged = torch.log1p(shifted.clamp_min(0.0))
    x = robust_standardize(logged)
    edge, _ = fixed_edge_maps(x)
    padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
    patches = F.unfold(padded, kernel_size=5).reshape(
        x.shape[0], x.shape[1], 25, x.shape[2], x.shape[3]
    )
    local_median = patches.median(2).values
    local_mad = (patches - local_median[:, :, None]).abs().median(2).values
    residual = (x - local_median).abs()
    maps = torch.cat((edge[:, :4], residual, local_mad), dim=1)
    names = ("scharr_x", "scharr_y", "gradient", "log", "median_residual", "local_mad")
    return robust_standardize(maps), names


def pooled_map_features(
    maps: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    prefix: str,
    channel_names: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    pooled = pool_role_tokens(maps, masks)
    values = pooled.reshape(-1)
    names = tuple(f"{prefix}_{role}_{name}" for role in ROLE_NAMES for name in channel_names)
    return values, names


def structural_relation_features(
    rgb_maps: torch.Tensor,
    sar_maps: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Compare modality-internal role contrasts, not raw RGB/SAR intensities."""
    rgb = pool_role_tokens(rgb_maps[:, :4], masks)
    sar = pool_role_tokens(sar_maps[:, :4], masks)
    values, names = [], []
    for left_index, right_index, relation in ((0, 1, "change_anchor"), (0, 2, "change_boundary")):
        rgb_delta = rgb[left_index] - rgb[right_index]
        sar_delta = sar[left_index] - sar[right_index]
        values.extend(((rgb_delta - sar_delta).abs(), rgb_delta.abs(), sar_delta.abs()))
        for prefix in ("cross_gap", "rgb_internal", "sar_internal"):
            names.extend(f"{relation}_{prefix}_{index}" for index in range(4))
    return torch.cat(values), tuple(names)


def selective_histogram_from_fixed(
    fixed_histograms: torch.Tensor,
    delta_hist: torch.Tensor,
    accepted: torch.Tensor,
) -> torch.Tensor:
    """Eligible clusters rejected by verifier are reverted from S2 to P3."""
    fixed = fixed_histograms.sum(0).double()
    rejected = ~accepted.bool()
    if not bool(rejected.any()):
        return fixed
    return fixed - delta_hist[rejected].sum(0).double()


__all__ = [
    "fixed_edge_maps", "fixed_sar_texture_maps", "pooled_map_features",
    "robust_standardize", "selective_histogram_from_fixed",
    "structural_relation_features",
]
