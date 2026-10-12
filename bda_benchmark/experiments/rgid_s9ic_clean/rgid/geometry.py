from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import watershed

from .config import RGIDConfig
from .targets_losses import binary_mask_to_instances


@dataclass(frozen=True)
class Peak:
    y: int
    x: int
    score: float
    parent_id: int


@dataclass
class DecodeStats:
    center_candidates_total: int = 0
    centers_retained_total: int = 0
    parent_center_histogram: Dict[str, int] = field(
        default_factory=lambda: {"0": 0, "1": 0, ">1": 0}
    )
    split_candidate_regions: int = 0
    split_accepted_regions: int = 0
    boundary_gate_accepted_regions: int = 0
    boundary_gate_rejected_regions: int = 0
    rollback_candidate_regions: int = 0
    rollback_accepted_regions: int = 0
    rollback_rejected_regions: int = 0
    rollback_reason_counts: Dict[str, int] = field(default_factory=dict)
    markers_in_split_candidates: List[int] = field(default_factory=list)
    cut_scores: List[float] = field(default_factory=list)

    def to_dict(self):
        output = asdict(self)
        output["markers_per_split_region"] = (
            float(np.mean(self.markers_in_split_candidates))
            if self.markers_in_split_candidates
            else 0.0
        )
        return output


@dataclass
class DecodeResult:
    instance_labels: np.ndarray
    stats: DecodeStats


def _as_2d_array(value, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to a 2D array, got {array.shape}")
    return array


def _resize_nearest(labels: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(labels.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=size, mode="nearest")
    return resized[0, 0].numpy().astype(np.int32)


def _resize_bilinear(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def extract_native_peaks(
    center_prob: np.ndarray,
    parent_labels_original: np.ndarray,
    config: RGIDConfig,
) -> Tuple[List[Peak], int]:
    probability = _as_2d_array(center_prob, "center_prob").astype(np.float32)
    parents = _as_2d_array(
        parent_labels_original, "parent_labels_original"
    ).astype(np.int32)
    head_parents = _resize_nearest(parents, probability.shape)
    kernel = int(config.nms_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer")
    tensor = torch.from_numpy(probability)[None, None]
    local_max = F.max_pool2d(
        tensor,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )[0, 0].numpy()
    candidates = (
        (probability >= config.peak_threshold)
        & (probability == local_max)
        & (head_parents > 0)
    )
    raw_candidate_count = int(candidates.sum())

    plateau_peaks: List[Peak] = []
    for parent_id in np.unique(head_parents[candidates]):
        if int(parent_id) <= 0:
            continue
        plateau_labels, count = ndimage.label(
            candidates & (head_parents == parent_id),
            structure=ndimage.generate_binary_structure(2, 2),
        )
        for component_id in range(1, int(count) + 1):
            coords = np.argwhere(plateau_labels == component_id)
            scores = probability[coords[:, 0], coords[:, 1]]
            best_score = float(scores.max())
            tied = coords[scores == best_score]
            tied = tied[np.lexsort((tied[:, 1], tied[:, 0]))]
            y, x = tied[0]
            plateau_peaks.append(
                Peak(int(y), int(x), best_score, int(parent_id))
            )

    retained: List[Peak] = []
    for parent_id in sorted({peak.parent_id for peak in plateau_peaks}):
        region_peaks = sorted(
            (peak for peak in plateau_peaks if peak.parent_id == parent_id),
            key=lambda peak: (-peak.score, peak.y, peak.x),
        )
        selected: List[Peak] = []
        for peak in region_peaks:
            if all(
                (peak.y - old.y) ** 2 + (peak.x - old.x) ** 2
                >= config.min_peak_distance**2
                for old in selected
            ):
                selected.append(peak)
            if len(selected) >= config.max_markers_per_region:
                break
        retained.extend(selected)
    retained.sort(key=lambda peak: (peak.parent_id, -peak.score, peak.y, peak.x))
    return retained, raw_candidate_count


def _map_peak_to_parent(
    peak: Peak,
    head_size: Tuple[int, int],
    parent_labels: np.ndarray,
) -> Tuple[int, int]:
    head_h, head_w = head_size
    out_h, out_w = parent_labels.shape
    y = int(round(((peak.y + 0.5) * out_h / head_h) - 0.5))
    x = int(round(((peak.x + 0.5) * out_w / head_w) - 0.5))
    y = int(np.clip(y, 0, out_h - 1))
    x = int(np.clip(x, 0, out_w - 1))
    if parent_labels[y, x] == peak.parent_id:
        return y, x
    coords = np.argwhere(parent_labels == peak.parent_id)
    if coords.size == 0:
        raise RuntimeError(f"parent {peak.parent_id} disappeared during mapping")
    distances = np.square(coords[:, 0] - y) + np.square(coords[:, 1] - x)
    nearest = coords[int(np.argmin(distances))]
    return int(nearest[0]), int(nearest[1])


def boundary_cut_score(
    child_labels,
    parent_mask,
    boundary_prob,
    band_radius: int = 1,
) -> float:
    children = _as_2d_array(child_labels, "child_labels").astype(np.int32)
    parent = _as_2d_array(parent_mask, "parent_mask").astype(bool)
    boundary = _as_2d_array(boundary_prob, "boundary_prob").astype(np.float32)
    if boundary.shape != parent.shape:
        boundary = _resize_bilinear(boundary, parent.shape)
    cut = np.zeros_like(parent, dtype=bool)
    for dy, dx in ((1, 0), (0, 1)):
        shifted = np.roll(children, shift=(dy, dx), axis=(0, 1))
        valid = (
            parent
            & np.roll(parent, shift=(dy, dx), axis=(0, 1))
            & (children > 0)
            & (shifted > 0)
            & (children != shifted)
        )
        if dy:
            valid[:dy] = False
        if dx:
            valid[:, :dx] = False
        cut |= valid
        cut |= np.roll(valid, shift=(-dy, -dx), axis=(0, 1))
    cut &= parent
    if not cut.any():
        return 0.0
    if band_radius > 0:
        cut = ndimage.binary_dilation(cut, iterations=int(band_radius)) & parent
    return float(boundary[cut].mean()) if cut.any() else 0.0


def validate_geometry(
    child_labels,
    parent_mask,
    marker_coords,
    config: RGIDConfig,
) -> List[str]:
    children = _as_2d_array(child_labels, "child_labels").astype(np.int32)
    parent = _as_2d_array(parent_mask, "parent_mask").astype(bool)
    reasons: List[str] = []
    child_ids = [int(value) for value in np.unique(children[parent]) if value > 0]
    if len(child_ids) < 2:
        reasons.append("invalid_child_count")

    parent_area = int(parent.sum())
    for child_id in child_ids:
        area = int(((children == child_id) & parent).sum())
        if area < config.min_child_area and "child_too_small" not in reasons:
            reasons.append("child_too_small")
        if (
            parent_area > 0
            and area / parent_area < config.min_child_parent_ratio
            and "child_ratio_too_small" not in reasons
        ):
            reasons.append("child_ratio_too_small")

    assigned_area = int(((children > 0) & parent).sum())
    outside_area = int(((children > 0) & ~parent).sum())
    if (
        abs(assigned_area - parent_area) > config.area_tolerance_pixels
        or outside_area > config.area_tolerance_pixels
    ):
        reasons.append("area_not_conserved")

    marker_counts = {child_id: 0 for child_id in child_ids}
    invalid_marker = False
    for y, x in marker_coords:
        if not (0 <= y < children.shape[0] and 0 <= x < children.shape[1]):
            invalid_marker = True
            continue
        child_id = int(children[y, x])
        if child_id <= 0 or child_id not in marker_counts:
            invalid_marker = True
        else:
            marker_counts[child_id] += 1
    expected = (
        all(count == 1 for count in marker_counts.values())
        if config.require_exactly_one_center_per_child
        else all(count >= 1 for count in marker_counts.values())
    )
    if invalid_marker or len(marker_coords) != sum(marker_counts.values()) or not expected:
        reasons.append("center_child_mismatch")
    return reasons


class RGIDDecoder:
    def __init__(self, config: RGIDConfig) -> None:
        self.config = config

    def decode(
        self,
        center_prediction,
        boundary_prediction,
        parent_mask,
        mode: str,
        inputs_are_logits: bool = True,
        boundary_gate_threshold: float | None = None,
    ) -> DecodeResult:
        mode = str(mode).lower()
        if mode not in {"g1", "g2", "g3"}:
            raise ValueError("mode must be g1, g2 or g3")
        center = _as_2d_array(center_prediction, "center_prediction").astype(
            np.float32
        )
        boundary = _as_2d_array(
            boundary_prediction, "boundary_prediction"
        ).astype(np.float32)
        if inputs_are_logits:
            center = 1.0 / (1.0 + np.exp(-center))
            boundary = 1.0 / (1.0 + np.exp(-boundary))
        parent_binary = _as_2d_array(parent_mask, "parent_mask") > 0
        parent_labels = binary_mask_to_instances(parent_binary)
        peaks, candidate_count = extract_native_peaks(
            center, parent_labels, self.config
        )
        stats = DecodeStats(
            center_candidates_total=candidate_count,
            centers_retained_total=len(peaks),
        )
        boundary_original = _resize_bilinear(boundary, parent_binary.shape)
        peaks_by_parent: Dict[int, List[Peak]] = {}
        for peak in peaks:
            peaks_by_parent.setdefault(peak.parent_id, []).append(peak)

        output = np.zeros(parent_binary.shape, dtype=np.int32)
        next_instance_id = 1
        threshold = (
            self.config.boundary_gate_threshold
            if boundary_gate_threshold is None
            else float(boundary_gate_threshold)
        )

        def append_region(region_labels: np.ndarray, region_mask: np.ndarray) -> None:
            nonlocal next_instance_id
            child_ids = [int(value) for value in np.unique(region_labels) if value > 0]
            if not child_ids:
                output[region_mask] = next_instance_id
                next_instance_id += 1
                return
            for child_id in child_ids:
                child = (region_labels == child_id) & region_mask
                if child.any():
                    output[child] = next_instance_id
                    next_instance_id += 1

        for parent_id in range(1, int(parent_labels.max()) + 1):
            parent = parent_labels == parent_id
            region_peaks = peaks_by_parent.get(parent_id, [])
            count = len(region_peaks)
            key = "0" if count == 0 else ("1" if count == 1 else ">1")
            stats.parent_center_histogram[key] += 1
            if count < 2 or int(parent.sum()) < self.config.min_split_region_area:
                append_region(np.zeros_like(parent_labels), parent)
                continue

            stats.split_candidate_regions += 1
            stats.markers_in_split_candidates.append(count)
            marker_coords = [
                _map_peak_to_parent(peak, center.shape, parent_labels)
                for peak in region_peaks
            ]
            marker_image = np.zeros(parent.shape, dtype=np.int32)
            for marker_id, (y, x) in enumerate(marker_coords, start=1):
                marker_image[y, x] = marker_id
            distance = ndimage.distance_transform_edt(parent)
            energy = (
                -self.config.watershed_distance_weight * distance
                + self.config.watershed_boundary_weight * boundary_original
            )
            children = watershed(energy, markers=marker_image, mask=parent).astype(
                np.int32
            )

            if mode in {"g2", "g3"}:
                cut_score = boundary_cut_score(
                    children,
                    parent,
                    boundary_original,
                    band_radius=self.config.cut_band_radius,
                )
                stats.cut_scores.append(cut_score)
                if cut_score < threshold:
                    stats.boundary_gate_rejected_regions += 1
                    append_region(np.zeros_like(parent_labels), parent)
                    continue
                stats.boundary_gate_accepted_regions += 1

            if mode == "g3":
                stats.rollback_candidate_regions += 1
                reasons = validate_geometry(
                    children, parent, marker_coords, self.config
                )
                if reasons:
                    stats.rollback_rejected_regions += 1
                    for reason in reasons:
                        stats.rollback_reason_counts[reason] = (
                            stats.rollback_reason_counts.get(reason, 0) + 1
                        )
                    append_region(np.zeros_like(parent_labels), parent)
                    continue
                stats.rollback_accepted_regions += 1

            stats.split_accepted_regions += 1
            append_region(children, parent)

        return DecodeResult(output.astype(np.int32, copy=False), stats)
