from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

GEOMETRY_NAMES = (
    "area_score", "elongation_score", "holes_score", "entropy_score",
    "peak_score", "boundary_score", "normalized_child_count",
)
EDGE_NAMES = (
    "band_mean", "band_std", "band_q75", "band_to_parent_mean",
    "band_coverage",
)
PAIR_NAMES = (
    "edge_abs_difference_mean", "edge_min_support_mean",
    "edge_product_mean", "edge_correlation",
)


def candidate_cut_band(
    parent_labels: np.ndarray | torch.Tensor,
    child_labels: np.ndarray | torch.Tensor,
    parent_id: int,
    radius: int = 2,
) -> torch.Tensor:
    """Return only the internal boundary proposed by RGID, dilated in its parent.

    The external LocCC outline is intentionally excluded: this experiment asks
    whether a *candidate split* is supported, not whether the building exists.
    """

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    parents = np.asarray(torch.as_tensor(parent_labels).cpu(), dtype=np.int32)
    children = np.asarray(torch.as_tensor(child_labels).cpu(), dtype=np.int32)
    if parents.ndim != 2 or children.shape != parents.shape:
        raise ValueError("parent and child maps must be aligned [H,W]")
    parent = parents == int(parent_id)
    if not parent.any():
        raise ValueError(f"parent_id {parent_id} is absent")
    cut = np.zeros_like(parent)
    horizontal = parent[:, 1:] & parent[:, :-1] & (children[:, 1:] != children[:, :-1])
    vertical = parent[1:, :] & parent[:-1, :] & (children[1:, :] != children[:-1, :])
    cut[:, 1:] |= horizontal
    cut[:, :-1] |= horizontal
    cut[1:, :] |= vertical
    cut[:-1, :] |= vertical
    if radius and cut.any():
        cut = ndimage.binary_dilation(cut, iterations=int(radius)) & parent
    return torch.from_numpy(cut.astype(bool, copy=False))


def _robust_unit(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float()
    if image.ndim != 4:
        raise ValueError("image must be [B,C,H,W]")
    gray = image.mean(1, keepdim=True)
    flat = gray.flatten(2)
    median = flat.median(2).values[..., None]
    mad = (flat - median).abs().median(2).values[..., None].clamp_min(1e-4)
    return ((flat - median) / (1.4826 * mad)).clamp(-6, 6).view_as(gray)


def scharr_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Parameter-free robust edge map; suitable for RGB and speckled SAR."""

    gray = _robust_unit(image)
    kernel_x = gray.new_tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 16.0
    kernel_y = kernel_x.t()
    gx = F.conv2d(gray, kernel_x[None, None], padding=1)
    gy = F.conv2d(gray, kernel_y[None, None], padding=1)
    magnitude = (gx.square() + gy.square() + 1e-8).sqrt()
    scale = torch.quantile(magnitude.flatten(2), 0.95, dim=2)[..., None].clamp_min(1e-4)
    return (magnitude.flatten(2) / scale).clamp(0, 2).view_as(magnitude)


def summarize_map(values: torch.Tensor, band: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        values = values.mean(0)
    if values.ndim != 2 or values.shape != band.shape or parent.shape != band.shape:
        raise ValueError("map, band and parent must be aligned [H,W]")
    selected = values.float()[band]
    parent_values = values.float()[parent]
    if not selected.numel() or not parent_values.numel():
        raise ValueError("cut band and parent must be non-empty")
    return torch.stack((
        selected.mean(), selected.std(unbiased=False), torch.quantile(selected, 0.75),
        selected.mean() / parent_values.mean().abs().clamp_min(1e-6),
        band.sum().float() / parent.sum().float().clamp_min(1),
    ))


def paired_edge_features(rgb_edge: torch.Tensor, sar_edge: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    rgb = rgb_edge.float().squeeze()[band]
    sar = sar_edge.float().squeeze()[band]
    if not rgb.numel():
        raise ValueError("cut band is empty")
    rgb_centered, sar_centered = rgb - rgb.mean(), sar - sar.mean()
    correlation = (rgb_centered * sar_centered).mean() / (
        rgb_centered.square().mean().sqrt() * sar_centered.square().mean().sqrt()
    ).clamp_min(1e-6)
    return torch.stack(((rgb - sar).abs().mean(), torch.minimum(rgb, sar).mean(),
                        (rgb * sar).mean(), correlation))


def compressed_feature_channels(feature: torch.Tensor, channels: int) -> torch.Tensor:
    """Deterministically reduce channels before the expensive local search.

    Group averaging is deliberately parameter-free.  It changes only E0
    evidence extraction and never touches the frozen P3 forward path.
    """

    if feature.ndim != 4 or channels <= 0:
        raise ValueError("feature must be [B,C,H,W] and channels must be positive")
    source = feature.shape[1]
    if channels >= source:
        return feature
    boundaries = torch.linspace(0, source, channels + 1, device=feature.device).round().long()
    rows = []
    for index in range(channels):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        rows.append(feature[:, start:max(start + 1, end)].float().mean(1, keepdim=True))
    return torch.cat(rows, dim=1).to(feature.dtype)


def fast_geometry_features(
    *, parent_mask: torch.Tensor, probabilities: torch.Tensor,
    boundary_probability: torch.Tensor, child_count: int,
) -> torch.Tensor:
    """Cheap prediction-only replacement for per-parent full-map risk scoring."""

    mask = parent_mask.bool().cpu()
    yy, xx = torch.nonzero(mask, as_tuple=True)
    area = int(mask.sum())
    height, width = int(yy.max() - yy.min() + 1), int(xx.max() - xx.min() + 1)
    crop = mask[int(yy.min()):int(yy.max()) + 1, int(xx.min()):int(xx.max()) + 1].numpy()
    holes = int((ndimage.binary_fill_holes(crop) & ~crop).sum()) / max(area, 1)
    selected = probabilities.float()[:, mask].clamp_min(1e-8)
    entropy = float((-(selected * selected.log()).sum(0) / np.log(4.0)).mean())
    boundary = boundary_probability.float().cpu()[mask]
    aspect = max(height, width) / max(1, min(height, width))
    return torch.tensor((
        np.clip(np.log1p(area) / np.log1p(4096), 0, 1),
        np.clip((aspect - 1.0) / 3.0, 0, 1), np.clip(holes, 0, 1),
        np.clip(entropy, 0, 1), np.clip((child_count - 1) / 3.0, 0, 1),
        float(boundary.mean().clamp(0, 1)), min(child_count / 5.0, 1.0),
    ), dtype=torch.float32)


def build_feature_groups(
    *, geometry: torch.Tensor, parent_mask: torch.Tensor, band: torch.Tensor,
    rgb_edge: torch.Tensor, sar_edge: torch.Tensor,
    matching_maps: torch.Tensor, shuffled_matching_maps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    parent_mask = parent_mask.bool().cpu()
    band = band.bool().cpu()
    rgb_edge, sar_edge = rgb_edge.cpu().squeeze(), sar_edge.cpu().squeeze()
    result = {
        "geometry": geometry.float().cpu(),
        "rgb_edge": summarize_map(rgb_edge, band, parent_mask),
        "sar_edge": summarize_map(sar_edge, band, parent_mask),
        "same_position": paired_edge_features(rgb_edge, sar_edge, band),
    }
    for name, maps in (("local_matching", matching_maps),
                       ("shuffled_matching", shuffled_matching_maps)):
        rows = [summarize_map(channel, band, parent_mask) for channel in maps.cpu()]
        result[name] = torch.cat(rows)
    return result


def split_utility(
    loccc_prediction: torch.Tensor, split_prediction: torch.Tensor,
    target: torch.Tensor, parent_mask: torch.Tensor,
) -> dict[str, float | int]:
    valid = parent_mask.bool() & (target >= 0) & (target <= 3)
    changed = valid & (loccc_prediction != split_prediction)
    corrected = changed & (split_prediction == target) & (loccc_prediction != target)
    harmed = changed & (loccc_prediction == target) & (split_prediction != target)
    count = int(changed.sum())
    net = int(corrected.sum()) - int(harmed.sum())
    return {
        "changed": count, "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()), "net": net,
        "utility_ratio": net / max(count, 1),
    }


def feature_names(matching_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    expanded = tuple(f"{name}_{stat}" for name in matching_names for stat in EDGE_NAMES)
    return {
        "geometry": GEOMETRY_NAMES, "rgb_edge": EDGE_NAMES, "sar_edge": EDGE_NAMES,
        "same_position": PAIR_NAMES, "local_matching": expanded,
        "shuffled_matching": expanded,
    }


__all__ = [
    "candidate_cut_band", "scharr_magnitude", "build_feature_groups",
    "split_utility", "feature_names", "compressed_feature_channels",
    "fast_geometry_features",
]
