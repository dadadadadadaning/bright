"""Native-scale center NMS, marker watershed, and Center+Offset decoding."""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import GeometryDecoderConfig


Peak = Tuple[int, int, float]


def _as_2d_float(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
    return array


def _resize_nearest(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    return (
        F.interpolate(tensor, size=size, mode="nearest")[0, 0].numpy() > 0.5
    )


def _resize_labels(labels: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(labels.astype(np.float32, copy=False))[None, None]
    return F.interpolate(tensor, size=size, mode="nearest")[0, 0].numpy().astype(
        np.int32
    )


def connected_components_4(mask: np.ndarray) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    height, width = mask.shape
    visited = np.zeros_like(mask)
    output: List[np.ndarray] = []
    for y0, x0 in np.argwhere(mask):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        queue: deque[Tuple[int, int]] = deque([(int(y0), int(x0))])
        points: List[Tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= yy < height
                    and 0 <= xx < width
                    and mask[yy, xx]
                    and not visited[yy, xx]
                ):
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        component = np.zeros_like(mask)
        yy, xx = zip(*points)
        component[np.asarray(yy), np.asarray(xx)] = True
        output.append(component)
    return output


def native_local_maxima(
    center_probability: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.35,
    nms_kernel: int = 3,
) -> List[Peak]:
    """Apply local-maximum NMS at the center head's native resolution."""

    if nms_kernel < 1 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer")
    probability = _as_2d_float(center_probability, "center_probability")
    tensor = torch.from_numpy(probability)[None, None]
    pooled = F.max_pool2d(
        tensor,
        kernel_size=nms_kernel,
        stride=1,
        padding=nms_kernel // 2,
    )
    keep = (tensor >= float(threshold)) & (tensor >= pooled - 1e-12)
    points = torch.nonzero(keep[0, 0], as_tuple=False)
    peaks = [
        (int(y), int(x), float(probability[int(y), int(x)]))
        for y, x in points
    ]
    peaks.sort(key=lambda item: (-item[2], item[0], item[1]))
    return peaks


def _greedy_distance_nms(
    peaks: Sequence[Peak],
    min_distance: int,
    max_count: int,
) -> List[Peak]:
    selected: List[Peak] = []
    minimum_squared = float(min_distance * min_distance)
    for candidate in sorted(peaks, key=lambda item: (-item[2], item[0], item[1])):
        if min_distance > 0 and any(
            (candidate[0] - current[0]) ** 2
            + (candidate[1] - current[1]) ** 2
            < minimum_squared
            for current in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_count:
            break
    return selected


def extract_region_peaks(
    center_probability: np.ndarray | torch.Tensor,
    region_mask: np.ndarray | torch.Tensor,
    config: GeometryDecoderConfig,
) -> List[Peak]:
    """Return native-resolution center peaks inside one full-resolution region."""

    probability = _as_2d_float(center_probability, "center_probability")
    if torch.is_tensor(region_mask):
        region_mask = region_mask.detach().to(device="cpu").numpy()
    region_native = _resize_nearest(
        np.asarray(region_mask, dtype=np.bool_),
        probability.shape,
    )
    peaks = [
        peak
        for peak in native_local_maxima(
            probability,
            threshold=config.peak_threshold,
            nms_kernel=config.nms_kernel,
        )
        if region_native[peak[0], peak[1]]
    ]
    peaks = _greedy_distance_nms(
        peaks,
        config.min_peak_distance_native,
        config.max_markers_per_region,
    )
    if not peaks and config.fallback_to_region_max and region_native.any():
        masked = np.where(region_native, probability, -np.inf)
        flat_index = int(np.argmax(masked))
        y, x = np.unravel_index(flat_index, probability.shape)
        if np.isfinite(masked[y, x]):
            peaks = [(int(y), int(x), float(probability[y, x]))]
    return peaks


def _scale_peak(
    peak: Peak,
    native_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> Tuple[int, int]:
    y = min(
        output_size[0] - 1,
        max(0, int(round((peak[0] + 0.5) * output_size[0] / native_size[0] - 0.5))),
    )
    x = min(
        output_size[1] - 1,
        max(0, int(round((peak[1] + 0.5) * output_size[1] / native_size[1] - 0.5))),
    )
    return y, x


def _snap_to_region(point: Tuple[int, int], region: np.ndarray) -> Tuple[int, int]:
    if region[point]:
        return point
    points = np.argwhere(region)
    if points.size == 0:
        return point
    distances = (
        (points[:, 0] - point[0]) ** 2 + (points[:, 1] - point[1]) ** 2
    )
    best = points[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def _boundary_gate_accepts(
    labels: np.ndarray,
    region: np.ndarray,
    boundary_probability_full: np.ndarray,
    threshold: float,
) -> Tuple[bool, float]:
    """Score the three-pixel band around provisional watershed cut lines."""

    cut = np.zeros(labels.shape, dtype=np.bool_)
    vertical = (
        region[:-1]
        & region[1:]
        & (labels[:-1] > 0)
        & (labels[1:] > 0)
        & (labels[:-1] != labels[1:])
    )
    horizontal = (
        region[:, :-1]
        & region[:, 1:]
        & (labels[:, :-1] > 0)
        & (labels[:, 1:] > 0)
        & (labels[:, :-1] != labels[:, 1:])
    )
    cut[:-1] |= vertical
    cut[1:] |= vertical
    cut[:, :-1] |= horizontal
    cut[:, 1:] |= horizontal
    if not cut.any():
        return False, 0.0
    band = F.max_pool2d(
        torch.from_numpy(cut.astype(np.float32))[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0].numpy() > 0.0
    band &= region
    score = float(boundary_probability_full[band].mean()) if band.any() else 0.0
    return score >= threshold, score


def _marker_watershed(
    region: np.ndarray,
    markers: Sequence[Tuple[int, int]],
    elevation: np.ndarray,
) -> np.ndarray:
    """Dependency-free marker watershed using minimax topographic flooding."""

    region = np.asarray(region, dtype=np.bool_)
    labels = np.zeros(region.shape, dtype=np.int32)
    best_cost = np.full(region.shape, np.inf, dtype=np.float64)
    best_distance = np.full(region.shape, np.iinfo(np.int32).max, dtype=np.int32)
    heap: List[Tuple[float, int, int, int, int]] = []
    for label, marker in enumerate(markers, start=1):
        y, x = _snap_to_region(marker, region)
        cost = float(elevation[y, x])
        if cost < best_cost[y, x] or label < labels[y, x]:
            best_cost[y, x] = cost
            best_distance[y, x] = 0
            labels[y, x] = label
            heapq.heappush(heap, (cost, 0, label, y, x))

    height, width = region.shape
    while heap:
        cost, distance, label, y, x = heapq.heappop(heap)
        if (
            cost > best_cost[y, x] + 1e-12
            or distance != best_distance[y, x]
            or labels[y, x] != label
        ):
            continue
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= yy < height and 0 <= xx < width and region[yy, xx]):
                continue
            next_cost = max(cost, float(elevation[yy, xx]))
            next_distance = distance + 1
            if (
                next_cost < best_cost[yy, xx] - 1e-12
                or (
                    abs(next_cost - best_cost[yy, xx]) <= 1e-12
                    and (
                        next_distance < best_distance[yy, xx]
                        or (
                            next_distance == best_distance[yy, xx]
                            and (labels[yy, xx] == 0 or label < labels[yy, xx])
                        )
                    )
                )
            ):
                best_cost[yy, xx] = next_cost
                best_distance[yy, xx] = next_distance
                labels[yy, xx] = label
                heapq.heappush(
                    heap,
                    (next_cost, next_distance, label, yy, xx),
                )
    return labels


def _split_rollback_reasons(
    labels: np.ndarray,
    region: np.ndarray,
    markers: Sequence[Tuple[int, int]],
    region_area: int,
    config: GeometryDecoderConfig,
) -> List[str]:
    reasons: List[str] = []
    child_areas = [
        int((labels == label).sum())
        for label in range(1, int(labels.max()) + 1)
    ]
    if len(child_areas) < 2:
        reasons.append("fewer_than_two_children")
    if any(area < config.min_instance_area for area in child_areas):
        reasons.append("child_too_small")
    ratio_limit = int(math.ceil(region_area * config.rollback_min_child_ratio))
    if any(area < ratio_limit for area in child_areas):
        reasons.append("child_ratio_too_small")
    labeled_area = int(((labels > 0) & region).sum())
    if labeled_area != region_area:
        reasons.append("area_not_conserved")

    marker_labels = []
    for marker in markers:
        y, x = _snap_to_region(marker, region)
        marker_labels.append(int(labels[y, x]))
    positive_marker_labels = [label for label in marker_labels if label > 0]
    child_labels = set(range(1, int(labels.max()) + 1))
    if (
        len(positive_marker_labels) != len(markers)
        or len(set(positive_marker_labels)) != len(markers)
        or set(positive_marker_labels) != child_labels
    ):
        reasons.append("center_child_mismatch")
    return reasons


def decode_center_watershed_instances(
    building_probability: np.ndarray | torch.Tensor,
    center_probability: np.ndarray | torch.Tensor,
    boundary_probability: Optional[np.ndarray | torch.Tensor] = None,
    *,
    config: GeometryDecoderConfig,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Decode semantic building regions with native-scale centers and watershed."""

    building = _as_2d_float(building_probability, "building_probability")
    center = _as_2d_float(center_probability, "center_probability")
    boundary = (
        np.zeros_like(center)
        if boundary_probability is None
        else _as_2d_float(boundary_probability, "boundary_probability")
    )
    if boundary.shape != center.shape:
        raise ValueError("center and boundary maps must share the native size")

    regions = [
        component
        for component in connected_components_4(
            building >= config.building_threshold
        )
        if int(component.sum()) >= config.min_instance_area
    ]
    instances: List[np.ndarray] = []
    diagnostics: Dict[str, Any] = {
        "regions": len(regions),
        "regions_without_peak": 0,
        "regions_with_one_peak": 0,
        "regions_with_multiple_peaks": 0,
        "peaks_after_nms": 0,
        "watershed_split_regions": 0,
        "watershed_disabled_regions": 0,
        "boundary_gate_accepted_regions": 0,
        "boundary_gate_rejected_regions": 0,
        "boundary_gate_score_sum": 0.0,
        "boundary_gate_evaluated_regions": 0,
        "rollback_regions": 0,
        "rollback_candidate_regions": 0,
        "rollback_accepted_regions": 0,
        "rollback_rejected_regions": 0,
        "rollback_reason_fewer_than_two_children": 0,
        "rollback_reason_child_too_small": 0,
        "rollback_reason_child_ratio_too_small": 0,
        "rollback_reason_center_child_mismatch": 0,
        "rollback_reason_area_not_conserved": 0,
        "rollback_reason_boundary_insufficient": 0,
        "watershed_enabled": config.watershed_enabled,
        "boundary_gate_enabled": config.boundary_gate_enabled,
        "rollback_enabled": config.rollback_enabled,
    }

    boundary_full = torch.from_numpy(boundary)[None, None]
    boundary_full = F.interpolate(
        boundary_full,
        size=building.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    for region in regions:
        raw_peaks = [
            peak
            for peak in native_local_maxima(
                center,
                threshold=config.peak_threshold,
                nms_kernel=config.nms_kernel,
            )
            if _resize_nearest(region, center.shape)[peak[0], peak[1]]
        ]
        peaks = extract_region_peaks(center, region, config)
        diagnostics["peaks_after_nms"] += len(peaks)
        if not raw_peaks:
            diagnostics["regions_without_peak"] += 1
        if len(peaks) <= 1:
            diagnostics["regions_with_one_peak"] += 1
            instances.append(region)
            continue
        diagnostics["regions_with_multiple_peaks"] += 1

        if int(region.sum()) < config.min_region_area_to_split:
            instances.append(region)
            continue
        if not config.watershed_enabled:
            diagnostics["watershed_disabled_regions"] += 1
            instances.append(region)
            continue
        markers = [
            _scale_peak(peak, center.shape, building.shape)
            for peak in peaks
        ]
        elevation = (
            -building
            + config.watershed_boundary_weight * boundary_full
        )
        labels = _marker_watershed(region, markers, elevation)
        if config.boundary_gate_enabled:
            accepted, score = _boundary_gate_accepts(
                labels,
                region,
                boundary_full,
                config.boundary_gate_threshold,
            )
            diagnostics["boundary_gate_score_sum"] += score
            diagnostics["boundary_gate_evaluated_regions"] += 1
            if not accepted:
                diagnostics["boundary_gate_rejected_regions"] += 1
                if config.rollback_enabled:
                    diagnostics["rollback_candidate_regions"] += 1
                    diagnostics["rollback_rejected_regions"] += 1
                    diagnostics["rollback_regions"] += 1
                    diagnostics["rollback_reason_boundary_insufficient"] += 1
                instances.append(region)
                continue
            diagnostics["boundary_gate_accepted_regions"] += 1

        if config.rollback_enabled:
            diagnostics["rollback_candidate_regions"] += 1
            reasons = _split_rollback_reasons(
                labels,
                region,
                markers,
                int(region.sum()),
                config,
            )
            if reasons:
                diagnostics["rollback_regions"] += 1
                diagnostics["rollback_rejected_regions"] += 1
                for reason in reasons:
                    diagnostics[f"rollback_reason_{reason}"] += 1
                instances.append(region)
                continue
            diagnostics["rollback_accepted_regions"] += 1
        children = [
            labels == label
            for label in range(1, int(labels.max()) + 1)
            if int((labels == label).sum()) >= config.min_instance_area
        ]
        if len(children) >= 2:
            diagnostics["watershed_split_regions"] += 1
            instances.extend(children)
        else:
            instances.append(region)
    diagnostics["instance_count"] = len(instances)
    return instances, diagnostics


def decode_center_offset_instances(
    building_probability: np.ndarray | torch.Tensor,
    center_probability: np.ndarray | torch.Tensor,
    offset: np.ndarray | torch.Tensor,
    *,
    config: GeometryDecoderConfig,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Assign each building pixel to the center reached by its predicted offset."""

    building = _as_2d_float(building_probability, "building_probability")
    center = _as_2d_float(center_probability, "center_probability")
    if torch.is_tensor(offset):
        offset = offset.detach().to(device="cpu", dtype=torch.float32).numpy()
    offset_array = np.asarray(offset, dtype=np.float32)
    if offset_array.ndim == 4 and offset_array.shape[0] == 1:
        offset_array = offset_array[0]
    if offset_array.shape != (2, *center.shape):
        raise ValueError(
            f"offset must be [2,{center.shape[0]},{center.shape[1]}], "
            f"got {offset_array.shape}"
        )

    building_native = (
        F.interpolate(
            torch.from_numpy(building)[None, None],
            size=center.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .numpy()
        >= config.building_threshold
    )
    peaks = native_local_maxima(
        center,
        threshold=config.peak_threshold,
        nms_kernel=config.nms_kernel,
    )
    peaks = _greedy_distance_nms(
        peaks,
        config.min_peak_distance_native,
        max_count=max(config.max_markers_per_region, len(peaks)),
    )
    if not peaks and config.fallback_to_region_max and building_native.any():
        masked = np.where(building_native, center, -np.inf)
        y, x = np.unravel_index(int(np.argmax(masked)), center.shape)
        peaks = [(int(y), int(x), float(center[y, x]))]
    if not peaks:
        return [], {
            "center_count": 0,
            "assigned_pixels": 0,
            "unassigned_pixels": int(building_native.sum()),
            "instance_count": 0,
        }

    centers_xy = np.asarray([(peak[1], peak[0]) for peak in peaks], dtype=np.float32)
    yy, xx = np.mgrid[: center.shape[0], : center.shape[1]]
    endpoint_x = xx.astype(np.float32) + offset_array[0]
    endpoint_y = yy.astype(np.float32) + offset_array[1]
    pixels = np.argwhere(building_native)
    labels_native = np.zeros(center.shape, dtype=np.int32)
    assigned = 0
    for y, x in pixels:
        endpoint = np.asarray([endpoint_x[y, x], endpoint_y[y, x]], dtype=np.float32)
        distances = np.sqrt(((centers_xy - endpoint[None, :]) ** 2).sum(axis=1))
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= config.offset_assignment_radius:
            labels_native[y, x] = nearest + 1
            assigned += 1

    labels_full = _resize_labels(labels_native, building.shape)
    instances = [
        (labels_full == label) & (building >= config.building_threshold)
        for label in range(1, len(peaks) + 1)
    ]
    instances = [
        mask for mask in instances if int(mask.sum()) >= config.min_instance_area
    ]
    diagnostics = {
        "center_count": len(peaks),
        "assigned_pixels": assigned,
        "unassigned_pixels": int(building_native.sum()) - assigned,
        "assignment_rate": assigned / max(1, int(building_native.sum())),
        "instance_count": len(instances),
        "watershed_used": False,
    }
    return instances, diagnostics
