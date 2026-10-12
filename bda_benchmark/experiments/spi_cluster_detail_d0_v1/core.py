from __future__ import annotations

import torch


AREA_BINS = (
    ("area_1", 1, 1), ("area_2", 2, 2), ("area_3_4", 3, 4),
    ("area_5_8", 5, 8), ("area_9_16", 9, 16),
    ("area_17_32", 17, 32), ("area_gt_32", 33, 2**31 - 1),
)


def area_bin_masks(area: torch.Tensor) -> dict[str, torch.Tensor]:
    area = area.long()
    return {name: (area >= low) & (area <= high) for name, low, high in AREA_BINS}


def hard_token_coverage(
    local_mask: torch.Tensor,
    slices: tuple[slice, slice],
    image_shape: tuple[int, int],
    token_shape: tuple[int, int],
) -> int:
    yy, xx = torch.nonzero(local_mask.bool(), as_tuple=True)
    if not len(yy):
        return 0
    yy = yy + int(slices[0].start)
    xx = xx + int(slices[1].start)
    image_h, image_w = image_shape
    token_h, token_w = token_shape
    ty = torch.floor((yy.float() + 0.5) * token_h / image_h).long().clamp(0, token_h - 1)
    tx = torch.floor((xx.float() + 0.5) * token_w / image_w).long().clamp(0, token_w - 1)
    return int(torch.unique(ty * token_w + tx).numel())


def reconstruct_oracle_histograms(cache: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p3 = cache["p3_histograms"].sum(0).double()
    fixed = cache["fixed_s2_histograms"].sum(0).double()
    positive = cache["utility"]["net"].long() > 0
    cluster_oracle = p3 + cache["delta_hist"][positive].sum(0).double()
    return p3, fixed, cluster_oracle


def duplicate_fraction(keys: torch.Tensor, features: torch.Tensor, decimals: int = 3) -> float:
    """Fraction of clusters sharing a quantized feature vector in one pseudo-building."""
    if not len(keys):
        return float("nan")
    quantized = torch.round(features.float() * (10**decimals)).to(torch.int32)
    duplicates = 0
    for key in torch.unique(keys):
        rows = quantized[keys == key]
        duplicates += len(rows) - len(torch.unique(rows, dim=0))
    return duplicates / max(len(keys), 1)


__all__ = [
    "AREA_BINS", "area_bin_masks", "duplicate_fraction", "hard_token_coverage",
    "reconstruct_oracle_histograms",
]
