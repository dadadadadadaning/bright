from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def core_and_ring(building: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation-only GT building core and its boundary ring."""
    if radius < 1:
        raise ValueError("radius must be positive")
    labels, _ = ndimage.label(building.cpu().numpy())
    eroded = ndimage.binary_erosion(labels > 0, iterations=radius)
    core = torch.from_numpy(eroded).bool()
    return core, building.bool() & ~core


def affected_component_area(target: torch.Tensor) -> torch.Tensor:
    """Pixel map containing the GT connected affected-component area."""
    labels, count = ndimage.label((target.cpu().numpy() >= 2))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    return torch.from_numpy(sizes[labels]).long()


def modification_cluster_area(changed: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Dense area map for prediction-only S2 modification clusters."""
    labels, count = ndimage.label(changed.detach().cpu().numpy())
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    return torch.from_numpy(sizes[labels]).long(), int(count)


def robust_sar_noise(sar: torch.Tensor, size: int = 5) -> torch.Tensor:
    """Local median absolute deviation, a descriptive SAR instability proxy."""
    source = sar.detach().float().cpu().numpy()
    median = ndimage.median_filter(source, size=size, mode="reflect")
    mad = ndimage.median_filter(np.abs(source - median), size=size, mode="reflect")
    return torch.from_numpy(np.ascontiguousarray(mad)).float()


def _edge_magnitude(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        value = value.mean(0)
    value = value.float()[None, None]
    x = value.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    y = x.transpose(-1, -2)
    gx = F.conv2d(value, x, padding=1); gy = F.conv2d(value, y, padding=1)
    return torch.sqrt(gx.square() + gy.square()).squeeze(0).squeeze(0)


def alignment_displacement_proxy(rgb: torch.Tensor, sar: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """How much a small shift improves RGB/SAR edge agreement.

    It is a sensor-agnostic *proxy*, not optical-flow ground truth. Large values
    indicate that edge agreement is substantially better after a local shift.
    """
    if radius < 1:
        raise ValueError("radius must be positive")
    rgb_edge, sar_edge = _edge_magnitude(rgb), _edge_magnitude(sar)
    rgb_edge = (rgb_edge - rgb_edge.mean()) / rgb_edge.std().clamp_min(1e-6)
    sar_edge = (sar_edge - sar_edge.mean()) / sar_edge.std().clamp_min(1e-6)
    zero = (rgb_edge - sar_edge).abs()
    padded = F.pad(sar_edge[None, None], (radius, radius, radius, radius), mode="replicate")[0, 0]
    best = zero.clone()
    height, width = zero.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
            best = torch.minimum(best, (rgb_edge - shifted).abs())
    return (zero - best).clamp_min(0).cpu()


def bucket_counts(affected: torch.Tensor, missed: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    relevant = affected.bool() & mask.bool()
    return int(relevant.sum()), int((missed.bool() & relevant).sum())


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


__all__ = ["affected_component_area", "alignment_displacement_proxy", "bucket_counts", "core_and_ring", "modification_cluster_area", "rate", "robust_sar_noise"]
