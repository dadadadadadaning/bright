from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import find_boundaries


@dataclass
class FastTargetStats:
    valid_instances: int = 0
    ignored_border_instances: int = 0
    center_collisions: int = 0


def _map_coordinate(value: int, source: int, target: int) -> int:
    mapped = round(((float(value) + 0.5) * target / source) - 0.5)
    return int(np.clip(mapped, 0, target - 1))


def build_center_target_fast(instance_ids, output_size, config):
    """Center target equivalent to the legacy implementation, using local EDTs."""

    labels = np.asarray(instance_ids)
    if labels.ndim != 2:
        raise ValueError("instance_ids must be two-dimensional")
    out_h, out_w = map(int, output_size)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_size must be positive")
    src_h, src_w = labels.shape
    target = np.zeros((out_h, out_w), dtype=np.float32)
    stats = FastTargetStats()
    objects = ndimage.find_objects(labels)
    y_grid = np.arange(out_h, dtype=np.float32)
    x_grid = np.arange(out_w, dtype=np.float32)

    for instance_id, bounding_slice in enumerate(objects, start=1):
        if bounding_slice is None:
            continue
        sy, sx = bounding_slice
        local = labels[sy, sx] == instance_id
        touches_border = (
            sy.start == 0 or sx.start == 0 or sy.stop == src_h or sx.stop == src_w
        )
        if touches_border:
            stats.ignored_border_instances += 1
            continue
        # One zero-valued pixel around the tight box preserves exactly the
        # distance to background while avoiding a full-image EDT per instance.
        padded = np.pad(local, 1, mode="constant", constant_values=False)
        distance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        candidates = np.argwhere(local & (distance == distance.max()))
        pixels = np.argwhere(local)
        centroid = pixels.mean(axis=0)
        squared = np.square(candidates - centroid[None]).sum(axis=1)
        order = np.lexsort((candidates[:, 1], candidates[:, 0], squared))
        local_y, local_x = candidates[int(order[0])]
        center_y = int(sy.start + local_y)
        center_x = int(sx.start + local_x)
        mapped_y = _map_coordinate(center_y, src_h, out_h)
        mapped_x = _map_coordinate(center_x, src_w, out_w)
        if target[mapped_y, mapped_x] >= 1.0:
            stats.center_collisions += 1
        scaled_area = float(local.sum()) * out_h / src_h * out_w / src_w
        radius = math.sqrt(max(scaled_area, 0.0) / math.pi)
        sigma = float(
            np.clip(
                radius / config.gaussian_radius_divisor,
                config.gaussian_sigma_min,
                config.gaussian_sigma_max,
            )
        )
        gy = np.exp(-np.square(y_grid - mapped_y) / (2.0 * sigma * sigma))
        gx = np.exp(-np.square(x_grid - mapped_x) / (2.0 * sigma * sigma))
        gaussian = (gy[:, None] * gx[None, :]).astype(np.float32, copy=False)
        np.maximum(target, gaussian, out=target)
        target[mapped_y, mapped_x] = 1.0
        stats.valid_instances += 1
    return target, stats


def build_boundary_target_fast(instance_ids, output_size, boundary_width=1):
    labels = np.asarray(instance_ids)
    boundary = find_boundaries(labels, mode="inner", connectivity=1)
    if boundary_width > 1:
        boundary = ndimage.binary_dilation(boundary, iterations=boundary_width - 1)
        boundary &= labels > 0
    tensor = torch.from_numpy(boundary.astype(np.float32))[None, None]
    return F.adaptive_max_pool2d(tensor, tuple(map(int, output_size)))[0, 0].numpy()


def build_rgid_targets_fast(instance_ids_batch, output_size, config, device=None):
    array = torch.as_tensor(instance_ids_batch).detach().cpu().numpy()
    if array.ndim == 2:
        array = array[None]
    centers, boundaries, stats = [], [], []
    for labels in array:
        center, row = build_center_target_fast(labels, output_size, config)
        boundary = build_boundary_target_fast(
            labels, output_size, boundary_width=config.boundary_width
        )
        centers.append(center)
        boundaries.append(boundary)
        stats.append(row)
    result = {
        "center_target": torch.from_numpy(np.stack(centers))[:, None].float(),
        "boundary_target": torch.from_numpy(np.stack(boundaries))[:, None].float(),
    }
    if device is not None:
        result = {key: value.to(device) for key, value in result.items()}
    return result, stats


__all__ = ["FastTargetStats", "build_center_target_fast", "build_rgid_targets_fast"]
