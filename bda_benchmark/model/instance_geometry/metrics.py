"""Diagnostics for center count, localization, and scale-dependent recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .decode import native_local_maxima
from .targets import deterministic_edt_center, map_point_to_size


def _mask_array(item: Any) -> np.ndarray:
    value = item.mask if hasattr(item, "mask") else item
    if torch.is_tensor(value):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"instance mask must be two-dimensional, got {array.shape}")
    return array


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return (
        F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None],
            size=size,
            mode="nearest",
        )[0, 0]
        .numpy()
        > 0.5
    )


def _mask_center_at_size(
    mask: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[float, float]:
    point = deterministic_edt_center(mask)
    mapped = map_point_to_size(point, tuple(mask.shape), size)
    return float(mapped[0]), float(mapped[1])


def _scale_bin(area: int) -> str:
    if area < 1024:
        return "small_lt1024"
    if area < 4096:
        return "medium_1024_4095"
    return "large_ge4096"


def _match_centers(
    predictions: Sequence[Tuple[float, float]],
    targets: Sequence[Tuple[float, float]],
    tolerance: float,
) -> Tuple[List[Tuple[int, int, float]], int, int]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            distance = float(
                np.hypot(
                    prediction[0] - target[0],
                    prediction[1] - target[1],
                )
            )
            if distance <= tolerance:
                candidates.append((distance, pred_index, target_index))
    candidates.sort()
    used_predictions = set()
    used_targets = set()
    matches: List[Tuple[int, int, float]] = []
    for distance, pred_index, target_index in candidates:
        if pred_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(target_index)
        matches.append((pred_index, target_index, distance))
    return (
        matches,
        len(predictions) - len(used_predictions),
        len(targets) - len(used_targets),
    )


def evaluate_center_mechanism(
    center_probability: np.ndarray | torch.Tensor,
    instance_masks: Sequence[Any],
    *,
    peak_threshold: float = 0.35,
    nms_kernel: int = 3,
    min_peak_distance_native: int = 3,
    match_tolerance_native: float = 3.0,
) -> Dict[str, Any]:
    """Evaluate 0/1/>1 peaks per building and center localization quality."""

    if torch.is_tensor(center_probability):
        center_probability = (
            center_probability.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
    probability = np.asarray(center_probability, dtype=np.float32)
    if probability.ndim == 3 and probability.shape[0] == 1:
        probability = probability[0]
    if probability.ndim != 2:
        raise ValueError("center_probability must be two-dimensional")

    raw_candidate_count = int((probability >= peak_threshold).sum())
    peaks = native_local_maxima(
        probability,
        threshold=peak_threshold,
        nms_kernel=nms_kernel,
    )
    selected = []
    minimum_squared = float(min_peak_distance_native**2)
    for peak in peaks:
        if any(
            (peak[0] - current[0]) ** 2 + (peak[1] - current[1]) ** 2
            < minimum_squared
            for current in selected
        ):
            continue
        selected.append(peak)

    masks = [_mask_array(item) for item in instance_masks]
    native_masks = [_resize_mask(mask, probability.shape) for mask in masks]
    valid_masks = [mask for mask in masks if mask.any()]
    target_centers = [
        _mask_center_at_size(mask, probability.shape)
        for mask in valid_masks
    ]
    target_areas = [int(mask.sum()) for mask in valid_masks]
    predicted_centers = [(float(peak[0]), float(peak[1])) for peak in selected]
    matches, false_positive, false_negative = _match_centers(
        predicted_centers,
        target_centers,
        match_tolerance_native,
    )

    peak_distribution = {"0": 0, "1": 0, ">1": 0}
    per_instance_peak_counts: List[int] = []
    for mask in native_masks:
        count = sum(bool(mask[peak[0], peak[1]]) for peak in selected)
        per_instance_peak_counts.append(count)
        if count == 0:
            peak_distribution["0"] += 1
        elif count == 1:
            peak_distribution["1"] += 1
        else:
            peak_distribution[">1"] += 1

    matched_target_indices = {target_index for _, target_index, _ in matches}
    scale_totals = {
        "small_lt1024": 0,
        "medium_1024_4095": 0,
        "large_ge4096": 0,
    }
    scale_matched = dict.fromkeys(scale_totals, 0)
    for target_index, area in enumerate(target_areas):
        key = _scale_bin(area)
        scale_totals[key] += 1
        if target_index in matched_target_indices:
            scale_matched[key] += 1

    true_positive = len(matches)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    errors = np.asarray([distance for _, _, distance in matches], dtype=np.float32)
    scale_recall = {
        key: scale_matched[key] / max(1, scale_totals[key])
        for key in scale_totals
    }
    return {
        "building_count": len(native_masks),
        "raw_threshold_candidates": raw_candidate_count,
        "peaks_after_nms": len(selected),
        "nms_suppressed": max(0, raw_candidate_count - len(selected)),
        "per_building_peak_count": per_instance_peak_counts,
        "building_peak_distribution": peak_distribution,
        "building_peak_rate_0": peak_distribution["0"] / max(1, len(native_masks)),
        "building_peak_rate_1": peak_distribution["1"] / max(1, len(native_masks)),
        "building_peak_rate_gt1": peak_distribution[">1"] / max(1, len(native_masks)),
        "center_true_positive": true_positive,
        "center_false_positive": false_positive,
        "center_false_negative": false_negative,
        "center_precision": precision,
        "center_recall": recall,
        "center_f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "localization_error_mean_native": float(errors.mean()) if errors.size else None,
        "localization_error_median_native": (
            float(np.median(errors)) if errors.size else None
        ),
        "localization_error_p90_native": (
            float(np.percentile(errors, 90)) if errors.size else None
        ),
        "localization_errors_native": errors.tolist(),
        "scale_total": scale_totals,
        "scale_matched": scale_matched,
        "scale_recall": scale_recall,
    }


@dataclass
class CenterMetricAccumulator:
    """Dataset-level accumulator for center diagnostics."""

    building_count: int = 0
    peak_zero: int = 0
    peak_one: int = 0
    peak_many: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    raw_candidates: int = 0
    peaks_after_nms: int = 0
    localization_errors: List[float] = field(default_factory=list)
    scale_total: Dict[str, int] = field(
        default_factory=lambda: {
            "small_lt1024": 0,
            "medium_1024_4095": 0,
            "large_ge4096": 0,
        }
    )
    scale_matched: Dict[str, int] = field(
        default_factory=lambda: {
            "small_lt1024": 0,
            "medium_1024_4095": 0,
            "large_ge4096": 0,
        }
    )

    def update(self, report: Mapping[str, Any]) -> None:
        distribution = report["building_peak_distribution"]
        self.building_count += int(report["building_count"])
        self.peak_zero += int(distribution["0"])
        self.peak_one += int(distribution["1"])
        self.peak_many += int(distribution[">1"])
        self.true_positive += int(report["center_true_positive"])
        self.false_positive += int(report["center_false_positive"])
        self.false_negative += int(report["center_false_negative"])
        self.raw_candidates += int(report["raw_threshold_candidates"])
        self.peaks_after_nms += int(report["peaks_after_nms"])
        exact_errors = report.get("localization_errors_native")
        if exact_errors is not None:
            self.localization_errors.extend(float(value) for value in exact_errors)
        else:
            mean_error = report.get("localization_error_mean_native")
            matched = int(report["center_true_positive"])
            if mean_error is not None:
                self.localization_errors.extend([float(mean_error)] * matched)
        for key in self.scale_total:
            self.scale_total[key] += int(report["scale_total"][key])
            self.scale_matched[key] += int(report["scale_matched"][key])

    def compute(self) -> Dict[str, Any]:
        precision = self.true_positive / max(
            1, self.true_positive + self.false_positive
        )
        recall = self.true_positive / max(
            1, self.true_positive + self.false_negative
        )
        errors = np.asarray(self.localization_errors, dtype=np.float32)
        return {
            "building_count": self.building_count,
            "building_peak_rate_0": self.peak_zero / max(1, self.building_count),
            "building_peak_rate_1": self.peak_one / max(1, self.building_count),
            "building_peak_rate_gt1": self.peak_many
            / max(1, self.building_count),
            "center_precision": precision,
            "center_recall": recall,
            "center_f1": 2.0
            * precision
            * recall
            / max(1e-12, precision + recall),
            "raw_threshold_candidates": self.raw_candidates,
            "peaks_after_nms": self.peaks_after_nms,
            "nms_suppressed": max(
                0, self.raw_candidates - self.peaks_after_nms
            ),
            "localization_error_mean_native": (
                float(errors.mean()) if errors.size else None
            ),
            "localization_error_median_native": (
                float(np.median(errors)) if errors.size else None
            ),
            "localization_error_p90_native": (
                float(np.percentile(errors, 90)) if errors.size else None
            ),
            "scale_recall": {
                key: self.scale_matched[key] / max(1, self.scale_total[key])
                for key in self.scale_total
            },
            "scale_total": dict(self.scale_total),
            "scale_matched": dict(self.scale_matched),
        }
