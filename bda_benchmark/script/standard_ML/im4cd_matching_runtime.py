# -*- coding: utf-8 -*-
"""Parameter-free IM4CD-style local matching utilities.

This module deliberately does not warp an image and does not change P3 logits.
It measures whether the *location of a hard local response peak* contains
building-damage evidence beyond a co-located RGB--SAR feature difference.

The original IM4CD paper estimates a projective transform from sparse control
points.  BRIGHT crops are already approximately registered, so this controlled
probe uses the dominant discrete feature-grid translation as the only global
motion model.  The distinction is recorded in every cache manifest; the code
must not be described as a full IM4CD reproduction.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


COLOCATED_STAT_NAMES = (
    "cosine_distance_mean",
    "cosine_distance_std",
    "cosine_distance_q90",
)

MATCHING_STAT_NAMES = (
    "one_minus_peak_mean",
    "one_minus_peak_std",
    "one_minus_peak_q90",
    "peak_gain_mean",
    "peak_gain_q90",
    "hard_residual_mean",
    "hard_residual_q90",
    "margin_mean",
    "entropy_mean",
    "confidence_weighted_residual_mean",
)


def _validate_features(rgb: torch.Tensor, sar: torch.Tensor) -> None:
    if rgb.ndim != 4 or sar.ndim != 4:
        raise ValueError("RGB and SAR features must be [B,C,H,W]")
    if rgb.shape != sar.shape:
        raise ValueError(
            f"local matching requires equal feature shapes, got "
            f"{tuple(rgb.shape)} and {tuple(sar.shape)}"
        )
    if min(rgb.shape[-2:]) < 3:
        raise ValueError("feature maps are too small for local matching")


@torch.no_grad()
def hard_local_matching_maps(
    rgb: torch.Tensor,
    sar: torch.Tensor,
    *,
    window_size: int = 5,
    temperature: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Return co-located and hard-peak matching evidence maps.

    A cosine response volume is computed between every RGB feature and a local
    SAR search window.  Unlike the historical CCU soft-alignment branch, the
    change geometry uses the *argmax location*.  A dominant tile translation
    is estimated from confident interior matches and subtracted before the
    residual magnitude is calculated.
    """

    _validate_features(rgb, sar)
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("window_size must be an odd integer >= 3")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch, channels, height, width = rgb.shape
    radius = window_size // 2
    neighbours = window_size * window_size
    locations = height * width

    rgb_unit = F.normalize(rgb.float(), dim=1, eps=1e-6)
    sar_unit = F.normalize(sar.float(), dim=1, eps=1e-6)
    patches = F.unfold(
        sar_unit, kernel_size=window_size, padding=radius
    ).reshape(batch, channels, neighbours, locations)
    scores = torch.einsum(
        "bcl,bckl->bkl", rgb_unit.reshape(batch, channels, locations), patches
    ).clamp(-1.0, 1.0)

    valid = F.unfold(
        torch.ones((batch, 1, height, width), device=rgb.device),
        kernel_size=window_size,
        padding=radius,
    ).reshape(batch, neighbours, locations)
    scores = scores.masked_fill(valid < 0.5, -1e4)

    centre_index = neighbours // 2
    centre = scores[:, centre_index : centre_index + 1]
    top = scores.topk(min(2, neighbours), dim=1).values
    peak = top[:, 0:1]
    margin = peak - top[:, 1:2] if top.shape[1] > 1 else peak
    argmax = scores.argmax(dim=1)

    probability = (scores / float(temperature)).softmax(dim=1)
    entropy = -(
        probability.clamp_min(1e-8).log() * probability
    ).sum(dim=1, keepdim=True) / math.log(float(neighbours))
    confidence = ((1.0 - entropy) * torch.sigmoid(5.0 * margin)).clamp(0.0, 1.0)

    offsets = torch.tensor(
        [
            (float(dx), float(dy))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ],
        device=rgb.device,
        dtype=torch.float32,
    )
    offset_x = offsets[argmax, 0]
    offset_y = offsets[argmax, 1]

    # Estimate a translation-only global motion from confident interior
    # matches.  This is a deliberately smaller claim than IM4CD's robust
    # projective fit and is appropriate for already georeferenced BRIGHT tiles.
    yy, xx = torch.meshgrid(
        torch.arange(height, device=rgb.device),
        torch.arange(width, device=rgb.device),
        indexing="ij",
    )
    interior = (
        (yy >= radius)
        & (yy < height - radius)
        & (xx >= radius)
        & (xx < width - radius)
    ).reshape(-1)
    dominant_indices = []
    flat_confidence = confidence[:, 0]
    for item in range(batch):
        weights = flat_confidence[item, interior]
        selected = argmax[item, interior]
        counts = torch.zeros(neighbours, device=rgb.device, dtype=torch.float32)
        counts.scatter_add_(0, selected, weights)
        if float(counts.sum()) <= 1e-8:
            counts.scatter_add_(0, selected, torch.ones_like(weights))
        dominant_indices.append(counts.argmax())
    dominant_index = torch.stack(dominant_indices)
    dominant_x = offsets[dominant_index, 0].view(batch, 1, 1)
    dominant_y = offsets[dominant_index, 1].view(batch, 1, 1)

    residual = torch.sqrt(
        (offset_x - dominant_x).square() + (offset_y - dominant_y).square()
    ) / max(math.sqrt(2.0) * radius, 1.0)
    residual = residual.clamp(0.0, 1.0)

    def spatial(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, 1, height, width).to(rgb.dtype)

    return {
        "colocated_cosine_distance": spatial((1.0 - centre).clamp(0.0, 2.0)),
        "one_minus_peak": spatial((1.0 - peak).clamp(0.0, 2.0)),
        "peak_gain": spatial((peak - centre).clamp_min(0.0)),
        "hard_offset_x": spatial(offset_x),
        "hard_offset_y": spatial(offset_y),
        "hard_residual": spatial(residual),
        "correlation_margin": spatial(margin.clamp_min(0.0)),
        "correlation_entropy": spatial(entropy.clamp(0.0, 1.0)),
        "matching_confidence": spatial(confidence),
        "confidence_weighted_residual": spatial(confidence[:, 0] * residual),
        "dominant_offset": torch.stack((dominant_x[:, 0, 0], dominant_y[:, 0, 0]), dim=1),
    }


def _weighted_mean_std(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weights.sum().clamp_min(1e-8)
    mean = (values * weights).sum() / denominator
    variance = ((values - mean).square() * weights).sum() / denominator
    return mean, variance.clamp_min(0.0).sqrt()


@torch.no_grad()
def pool_region_matching_statistics(
    maps: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Pool matching maps inside one feature-grid building support."""

    if weights.ndim != 2:
        raise ValueError("region weights must be [H,W]")
    selected = weights > 1e-6
    if int(selected.sum()) == 0:
        raise ValueError("region has no feature-grid support")

    def values(name: str) -> torch.Tensor:
        value = maps[name]
        if value.ndim == 4:
            if value.shape[0] != 1 or value.shape[1] != 1:
                raise ValueError(f"map {name!r} must be [1,1,H,W]")
            value = value[0, 0]
        if value.shape != weights.shape:
            raise ValueError(f"map {name!r} shape does not match region weights")
        return value.float()

    colocated = values("colocated_cosine_distance")
    colocated_mean, colocated_std = _weighted_mean_std(colocated, weights.float())
    colocated_q90 = torch.quantile(colocated[selected], 0.90)

    one_minus_peak = values("one_minus_peak")
    omp_mean, omp_std = _weighted_mean_std(one_minus_peak, weights.float())
    peak_gain = values("peak_gain")
    residual = values("hard_residual")
    margin = values("correlation_margin")
    entropy = values("correlation_entropy")
    weighted_residual = values("confidence_weighted_residual")

    def weighted_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * weights).sum() / weights.sum().clamp_min(1e-8)

    return {
        "colocated_stats": torch.stack(
            (colocated_mean, colocated_std, colocated_q90)
        ).cpu(),
        "matching_stats": torch.stack(
            (
                omp_mean,
                omp_std,
                torch.quantile(one_minus_peak[selected], 0.90),
                weighted_mean(peak_gain),
                torch.quantile(peak_gain[selected], 0.90),
                weighted_mean(residual),
                torch.quantile(residual[selected], 0.90),
                weighted_mean(margin),
                weighted_mean(entropy),
                weighted_mean(weighted_residual),
            )
        ).cpu(),
    }


def vector_normalizer(vectors: torch.Tensor) -> Dict[str, list[float]]:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty [N,D] tensor")
    vectors = vectors.float()
    return {
        "mean": vectors.mean(dim=0).tolist(),
        "std": vectors.std(dim=0, unbiased=False).clamp_min(1e-6).tolist(),
    }


def build_augmented_keys(
    visual: torch.Tensor,
    auxiliary: torch.Tensor,
    normalizer: Mapping[str, Sequence[float]],
    weight: float,
) -> torch.Tensor:
    visual = F.normalize(visual.float(), dim=1)
    if float(weight) == 0.0:
        return visual
    mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32)
    std = torch.as_tensor(normalizer["std"], dtype=torch.float32).clamp_min(1e-6)
    if auxiliary.ndim != 2 or auxiliary.shape[1] != mean.numel():
        raise ValueError("auxiliary vector dimension does not match normalizer")
    z = (auxiliary.float() - mean) / std
    return F.normalize(torch.cat((visual, float(weight) * z), dim=1), dim=1)


def binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Tie-aware AUROC without a sklearn dependency."""

    labels = labels.long().flatten().cpu()
    scores = scores.float().flatten().cpu()
    if labels.numel() != scores.numel() or labels.numel() == 0:
        raise ValueError("labels and scores must be equal non-empty vectors")
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores, dtype=torch.float64)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and bool(sorted_scores[end] == sorted_scores[start]):
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = ranks[positive].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)

