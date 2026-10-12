from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage


@dataclass
class RegionMaps:
    labels: torch.Tensor
    core_labels: torch.Tensor
    ring_labels: torch.Tensor
    region_count: int


def _one_image_maps(mask: np.ndarray, minimum_area: int, core_radius: int, ring_radius: int):
    structure = ndimage.generate_binary_structure(2, 1)  # exact 4-connectivity
    labels, count = ndimage.label(mask, structure=structure)
    if count:
        areas = np.bincount(labels.reshape(-1), minlength=count + 1)
        keep = areas >= int(minimum_area)
        keep[0] = False
        labels = labels * keep[labels]
        unique = np.unique(labels)
        unique = unique[unique > 0]
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        lut[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
        labels = lut[labels]
        count = int(unique.size)

    core = np.zeros_like(labels)
    if count:
        size = 2 * int(core_radius) + 1
        local_min = ndimage.minimum_filter(labels, size=size, mode="constant", cval=0)
        local_max = ndimage.maximum_filter(labels, size=size, mode="constant", cval=0)
        interior = (labels > 0) & (local_min == labels) & (local_max == labels)
        core[interior] = labels[interior]

    ring = np.zeros_like(labels)
    if count and ring_radius > 0 and np.any(labels == 0):
        distance, indices = ndimage.distance_transform_edt(
            labels == 0, return_indices=True
        )
        nearest = labels[indices[0], indices[1]]
        outside = (labels == 0) & (distance > 0) & (distance <= int(ring_radius))
        ring[outside] = nearest[outside]
    return labels, core, ring, count


def build_region_maps(
    p_building: torch.Tensor,
    threshold: float = 0.5,
    minimum_area: int = 2,
    core_radius: int = 1,
    ring_radius: int = 2,
) -> RegionMaps:
    """Build compact label maps on feature resolution; never materialize HxW masks/region."""
    if p_building.ndim != 4 or p_building.shape[1] != 1:
        raise ValueError(f"expected Bx1xHxW building probability, got {tuple(p_building.shape)}")
    binary = (p_building.detach().squeeze(1) >= threshold).cpu().numpy()
    label_batches, core_batches, ring_batches = [], [], []
    offset = 0
    for mask in binary:
        labels, core, ring, count = _one_image_maps(
            mask, minimum_area, core_radius, ring_radius
        )
        for array in (labels, core, ring):
            array[array > 0] += offset
        label_batches.append(labels)
        core_batches.append(core)
        ring_batches.append(ring)
        offset += count
    device = p_building.device
    convert = lambda xs: torch.as_tensor(np.stack(xs), device=device, dtype=torch.long)
    return RegionMaps(convert(label_batches), convert(core_batches), convert(ring_batches), offset)


def scatter_region_mean(features: torch.Tensor, labels: torch.Tensor, region_count: int):
    """GPU mean pooling into a compact (K+1)xC table; label zero is background."""
    if features.ndim != 4 or labels.shape != features.shape[:1] + features.shape[2:]:
        raise ValueError("feature/label shapes are incompatible")
    channels = features.shape[1]
    # Accumulate in FP32 even under BF16 autocast; BF16 counters become
    # inaccurate for large building regions.
    flat_features = features.permute(0, 2, 3, 1).reshape(-1, channels).float()
    flat_labels = labels.reshape(-1)
    table = torch.zeros((region_count + 1, channels), device=features.device, dtype=torch.float32)
    counts = torch.zeros((region_count + 1, 1), device=features.device, dtype=torch.float32)
    table.index_add_(0, flat_labels, flat_features)
    counts.index_add_(0, flat_labels, torch.ones((flat_labels.numel(), 1), device=features.device))
    table = table / counts.clamp_min(1.0)
    table[0] = 0
    return table.to(features.dtype), counts.squeeze(1)


def broadcast_table(table: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return table[labels].permute(0, 3, 1, 2).contiguous()
