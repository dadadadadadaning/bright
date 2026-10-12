"""Unique Gaussian-center and instance-balanced offset targets."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from torch import Tensor


def _normalize_label_tensor(value: Tensor) -> Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"label tensor must be [B,H,W], got {tuple(value.shape)}")
    return value


def _resize_mask(mask: Tensor, output_size: Tuple[int, int]) -> Tensor:
    mask = mask.detach().to(dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError(f"instance mask must be two-dimensional, got {mask.shape}")
    return F.interpolate(
        mask[None, None],
        size=output_size,
        mode="nearest",
    )[0, 0] > 0.5


def _touches_border(mask: Tensor) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def deterministic_edt_center(mask: Tensor | np.ndarray) -> Tuple[int, int]:
    """Return the unique deepest interior point with deterministic tie breaks.

    Tie order is: nearest to the instance centroid, smaller y, then smaller x.
    """

    if torch.is_tensor(mask):
        array = mask.detach().to(device="cpu").numpy()
    else:
        array = np.asarray(mask)
    array = np.asarray(array, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"instance mask must be two-dimensional, got {array.shape}")
    points = np.argwhere(array)
    if points.size == 0:
        raise ValueError("cannot compute a center for an empty mask")
    distance = distance_transform_edt(array)
    maximum = float(distance[array].max())
    candidates = np.argwhere(array & np.isclose(distance, maximum, atol=1e-7))
    centroid = points.astype(np.float64).mean(axis=0)
    centroid_distance = ((candidates.astype(np.float64) - centroid) ** 2).sum(axis=1)
    order = np.lexsort((candidates[:, 1], candidates[:, 0], centroid_distance))
    best = candidates[int(order[0])]
    return int(best[0]), int(best[1])


def map_point_to_size(
    point: Tuple[int, int],
    source_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Map a source pixel center to a nearest output pixel center."""

    y, x = point
    mapped_y = int(round((y + 0.5) * output_size[0] / source_size[0] - 0.5))
    mapped_x = int(round((x + 0.5) * output_size[1] / source_size[1] - 0.5))
    return (
        min(output_size[0] - 1, max(0, mapped_y)),
        min(output_size[1] - 1, max(0, mapped_x)),
    )


def _snap_point_to_mask(
    point: Tuple[int, int],
    mask: Tensor,
) -> Tuple[int, int]:
    if bool(mask[point[0], point[1]]):
        return point
    points = torch.nonzero(mask, as_tuple=False)
    if points.numel() == 0:
        raise ValueError("cannot snap a point to an empty mask")
    target = torch.tensor(point, dtype=torch.float32, device=points.device)
    distances = ((points.float() - target[None]) ** 2).sum(dim=1)
    best = points[int(torch.argmin(distances).item())]
    return int(best[0].item()), int(best[1].item())


def _draw_gaussian(
    heatmap: Tensor,
    center_y: int,
    center_x: int,
    sigma: float,
) -> None:
    height, width = heatmap.shape
    radius = max(1, int(math.ceil(3.0 * sigma)))
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    yy = torch.arange(y0, y1, device=heatmap.device, dtype=torch.float32)
    xx = torch.arange(x0, x1, device=heatmap.device, dtype=torch.float32)
    gaussian = torch.exp(
        -(
            (yy[:, None] - float(center_y)) ** 2
            + (xx[None, :] - float(center_x)) ** 2
        )
        / (2.0 * sigma * sigma)
    ).to(dtype=heatmap.dtype)
    heatmap[y0:y1, x0:x1] = torch.maximum(
        heatmap[y0:y1, x0:x1],
        gaussian,
    )
    heatmap[center_y, center_x] = 1.0


def _boundary_from_mask(mask: Tensor) -> Tensor:
    value = mask.to(dtype=torch.float32)[None, None]
    interior = -F.max_pool2d(-value, kernel_size=3, stride=1, padding=1)
    return ((value - interior) > 0.0)[0, 0]


def _build_specs(
    labels_loc: Tensor,
    labels_damage: Tensor,
    instance_builder: Callable[..., Sequence[Any]],
    min_area: int,
) -> Sequence[Any]:
    try:
        return instance_builder(labels_loc, labels_damage, min_area=min_area)
    except TypeError:
        return instance_builder(labels_loc, labels_damage)


def build_unique_gaussian_targets(
    labels_loc: Tensor,
    labels_damage: Tensor,
    instance_builder: Callable[..., Sequence[Any]],
    *,
    output_size: Tuple[int, int],
    min_area: int = 32,
    min_sigma: float = 1.0,
    max_sigma: float = 3.0,
    include_offsets: bool = False,
) -> Dict[str, Any]:
    """Build one deterministic Gaussian center per complete building instance.

    Border-touching instances are ignored for center and offset supervision.
    Their mapped pixels are set to zero in ``center_valid``.
    """

    labels_loc = _normalize_label_tensor(labels_loc)
    labels_damage = _normalize_label_tensor(labels_damage)
    if labels_loc.shape != labels_damage.shape:
        raise ValueError("localization and damage labels must share a shape")
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size must be positive")

    specs = _build_specs(
        labels_loc.detach().to(device="cpu"),
        labels_damage.detach().to(device="cpu"),
        instance_builder,
        min_area,
    )
    batch_size = int(labels_loc.shape[0])
    device = labels_loc.device
    dtype = torch.float32
    center = torch.zeros((batch_size, 1, *output_size), dtype=dtype, device=device)
    boundary = torch.zeros_like(center)
    center_valid = torch.ones_like(center)
    offset = torch.zeros((batch_size, 2, *output_size), dtype=dtype, device=device)
    offset_valid = torch.zeros((batch_size, 1, *output_size), dtype=dtype, device=device)
    offset_weight_raw = torch.zeros_like(offset_valid)

    complete_count = 0
    border_touching = 0
    collisions = 0
    centers: List[Tuple[int, int, int, int]] = []
    raw_instance_weights: List[Tuple[int, Tensor, float]] = []

    for spec_index, spec in enumerate(specs):
        batch_index = int(spec.batch_index)
        if not 0 <= batch_index < batch_size:
            raise IndexError(f"invalid batch_index={batch_index}")
        source_mask = torch.as_tensor(spec.mask, dtype=torch.bool)
        if source_mask.numel() == 0 or not bool(source_mask.any()):
            continue
        resized = _resize_mask(source_mask, output_size).to(device=device)
        boundary[batch_index, 0] = torch.maximum(
            boundary[batch_index, 0],
            _boundary_from_mask(resized).to(device=device, dtype=dtype),
        )
        if _touches_border(source_mask):
            border_touching += 1
            center_valid[batch_index, 0][resized] = 0.0
            continue

        source_center = deterministic_edt_center(source_mask)
        center_y, center_x = map_point_to_size(
            source_center,
            tuple(source_mask.shape),
            output_size,
        )
        center_y, center_x = _snap_point_to_mask(
            (center_y, center_x),
            resized,
        )
        if center[batch_index, 0, center_y, center_x] == 1:
            collisions += 1
            continue

        area = int(resized.sum().item())
        radius = math.sqrt(max(1, area) / math.pi)
        sigma = min(
            max_sigma,
            max(min_sigma, radius / 3.0),
        )
        _draw_gaussian(
            center[batch_index, 0],
            center_y,
            center_x,
            sigma,
        )
        complete_count += 1
        centers.append((batch_index, center_y, center_x, spec_index))

        if include_offsets:
            yy, xx = torch.meshgrid(
                torch.arange(output_size[0], device=device, dtype=dtype),
                torch.arange(output_size[1], device=device, dtype=dtype),
                indexing="ij",
            )
            offset[batch_index, 0][resized] = float(center_x) - xx[resized]
            offset[batch_index, 1][resized] = float(center_y) - yy[resized]
            offset_valid[batch_index, 0][resized] = 1.0
            raw_weight = 1.0 / float(max(1, area))
            offset_weight_raw[batch_index, 0][resized] = raw_weight
            raw_instance_weights.append((batch_index, resized, raw_weight))

    offset_weight = offset_weight_raw
    if include_offsets:
        valid_count = float(offset_valid.sum().item())
        raw_sum = float(offset_weight_raw.sum().item())
        if raw_sum > 0:
            offset_weight = offset_weight_raw * (valid_count / raw_sum)

    result: Dict[str, Any] = {
        "center": center,
        "center_valid": center_valid,
        "boundary": boundary,
        "complete_instance_count": complete_count,
        "border_touching": border_touching,
        "unique_center_pixels": int((center == 1.0).sum().item()),
        "mapped_center_collisions": collisions,
        "instance_centers": centers,
        "instance_specs": list(specs),
    }
    if include_offsets:
        result.update(
            {
                "offset": offset,
                "offset_valid": offset_valid,
                "offset_weight": offset_weight,
            }
        )
    return result


def build_center_offset_targets(
    labels_loc: Tensor,
    labels_damage: Tensor,
    instance_builder: Callable[..., Sequence[Any]],
    *,
    output_size: Tuple[int, int],
    min_area: int = 32,
    min_sigma: float = 1.0,
    max_sigma: float = 3.0,
) -> Dict[str, Any]:
    return build_unique_gaussian_targets(
        labels_loc,
        labels_damage,
        instance_builder,
        output_size=output_size,
        min_area=min_area,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        include_offsets=True,
    )
