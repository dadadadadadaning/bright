"""GT-derived *training/evaluation* masks. They never enter inference."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def boundary_and_small_masks(location: torch.Tensor, damage: torch.Tensor, small_area: float, radius: int = 2):
    if location.ndim != 3 or damage.ndim != 3:
        raise ValueError("location and damage must be [B,H,W]")
    boundary = torch.zeros_like(location, dtype=torch.bool)
    small = torch.zeros_like(location, dtype=torch.bool)
    for index in range(location.shape[0]):
        building = (location[index] > 0).detach().cpu().numpy()
        eroded = ndimage.binary_erosion(building, iterations=radius)
        boundary[index] = torch.from_numpy(np.ascontiguousarray(building & ~eroded)).to(boundary.device)
        affected = (damage[index] >= 2).detach().cpu().numpy()
        labels, count = ndimage.label(affected)
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
        small[index] = torch.from_numpy(np.ascontiguousarray((labels > 0) & (sizes[labels] <= small_area))).to(small.device)
    return boundary, small


def local_sar_noise(post: torch.Tensor) -> torch.Tensor:
    value = post.float().mean(1, keepdim=True)
    mean = torch.nn.functional.avg_pool2d(value, 5, stride=1, padding=2)
    return (value - mean).abs()
