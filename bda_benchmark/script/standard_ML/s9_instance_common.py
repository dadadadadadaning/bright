"""Dependency-light instance utilities shared by S9 training/evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class MaskInstance:
    instance_id: int
    batch_index: int
    mask: np.ndarray
    area: int
    bbox_yxyx: Tuple[int, int, int, int]
    label: Optional[int] = None
    purity: float = 0.0


def normalize_label_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"label tensor must be [B,H,W], got {tuple(value.shape)}")
    return value


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("cannot build an instance from an empty mask")
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()) + 1,
        int(points[:, 1].max()) + 1,
    )


def instances_from_bag_specs(
    specs: Sequence[Any],
    *,
    keep_labels: bool = False,
) -> List[MaskInstance]:
    output = []
    for instance_id, spec in enumerate(specs, start=1):
        mask = torch.as_tensor(spec.mask).detach().to(device="cpu").numpy().astype(
            np.bool_,
            copy=False,
        )
        if not mask.any():
            continue
        output.append(
            MaskInstance(
                instance_id=instance_id,
                batch_index=int(spec.batch_index),
                mask=mask,
                area=int(mask.sum()),
                bbox_yxyx=_bbox(mask),
                label=(
                    int(spec.label)
                    if keep_labels and int(getattr(spec, "label", -1)) >= 0
                    else None
                ),
            )
        )
    return output


def instances_from_masks(
    masks: Sequence[Any],
    *,
    batch_index: int = 0,
) -> List[MaskInstance]:
    output = []
    for instance_id, value in enumerate(masks, start=1):
        if torch.is_tensor(value):
            value = value.detach().to(device="cpu").numpy()
        mask = np.asarray(value, dtype=np.bool_)
        if mask.ndim != 2:
            raise ValueError(f"predicted mask must be [H,W], got {mask.shape}")
        if not mask.any():
            continue
        output.append(
            MaskInstance(
                instance_id=instance_id,
                batch_index=int(batch_index),
                mask=mask,
                area=int(mask.sum()),
                bbox_yxyx=_bbox(mask),
            )
        )
    return output


def make_predicted_specs(
    localization_mask: torch.Tensor,
    builder: Callable[..., Sequence[Any]],
    *,
    min_area: int = 32,
) -> Sequence[Any]:
    localization_mask = normalize_label_tensor(localization_mask)
    damage_placeholder = torch.where(
        localization_mask > 0,
        torch.ones_like(localization_mask),
        torch.zeros_like(localization_mask),
    )
    return builder(
        localization_mask,
        damage_placeholder,
        min_area=min_area,
    )


def iou_matrix(
    predicted: Sequence[MaskInstance],
    target: Sequence[MaskInstance],
) -> np.ndarray:
    matrix = np.zeros((len(predicted), len(target)), dtype=np.float32)
    for pred_index, pred in enumerate(predicted):
        for target_index, truth in enumerate(target):
            if pred.batch_index != truth.batch_index:
                continue
            intersection = _intersection_area(pred, truth)
            union = pred.area + truth.area - intersection
            matrix[pred_index, target_index] = intersection / max(1, union)
    return matrix


def _intersection_area(first: MaskInstance, second: MaskInstance) -> int:
    y0 = max(first.bbox_yxyx[0], second.bbox_yxyx[0])
    x0 = max(first.bbox_yxyx[1], second.bbox_yxyx[1])
    y1 = min(first.bbox_yxyx[2], second.bbox_yxyx[2])
    x1 = min(first.bbox_yxyx[3], second.bbox_yxyx[3])
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(
        np.logical_and(
            first.mask[y0:y1, x0:x1],
            second.mask[y0:y1, x0:x1],
        ).sum()
    )


def match_instances(
    predicted: Sequence[MaskInstance],
    target: Sequence[MaskInstance],
    *,
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    matrix = iou_matrix(predicted, target)
    candidates = [
        (float(matrix[pred, truth]), pred, truth)
        for pred in range(matrix.shape[0])
        for truth in range(matrix.shape[1])
        if matrix[pred, truth] >= iou_threshold
    ]
    candidates.sort(reverse=True)
    used_predicted = set()
    used_target = set()
    matches = []
    for score, pred, truth in candidates:
        if pred in used_predicted or truth in used_target:
            continue
        used_predicted.add(pred)
        used_target.add(truth)
        matches.append((pred, truth, score))
    return matches


def merge_split_diagnostics(
    predicted: Sequence[MaskInstance],
    target: Sequence[MaskInstance],
    *,
    overlap_threshold: float = 0.10,
) -> Dict[str, int]:
    overlaps = np.zeros((len(predicted), len(target)), dtype=np.float32)
    for pred_index, pred in enumerate(predicted):
        for target_index, truth in enumerate(target):
            if pred.batch_index != truth.batch_index:
                continue
            intersection = _intersection_area(pred, truth)
            overlaps[pred_index, target_index] = intersection / max(
                1,
                min(pred.area, truth.area),
            )
    return {
        "merge_pred_instances": int(
            sum((row >= overlap_threshold).sum() > 1 for row in overlaps)
        ),
        "split_gt_instances": int(
            sum((column >= overlap_threshold).sum() > 1 for column in overlaps.T)
        ),
    }


def area_bin(area: int) -> str:
    if area < 1024:
        return "small_lt1024"
    if area < 4096:
        return "medium_1024_4095"
    return "large_ge4096"


def evaluate_instance_geometry(
    predicted: Sequence[MaskInstance],
    target: Sequence[MaskInstance],
    *,
    iou_threshold: float = 0.5,
    overlap_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Evaluate one image using the locked greedy IoU matching protocol."""

    matches = match_instances(
        predicted,
        target,
        iou_threshold=iou_threshold,
    )
    diagnostics = merge_split_diagnostics(
        predicted,
        target,
        overlap_threshold=overlap_threshold,
    )
    matched_target = {truth_index for _, truth_index, _ in matches}
    scale_total = {
        "small_lt1024": 0,
        "medium_1024_4095": 0,
        "large_ge4096": 0,
    }
    scale_matched = dict.fromkeys(scale_total, 0)
    for truth_index, truth in enumerate(target):
        key = area_bin(truth.area)
        scale_total[key] += 1
        if truth_index in matched_target:
            scale_matched[key] += 1
    return {
        "target_instances": len(target),
        "predicted_instances": len(predicted),
        "matched_instances": len(matches),
        "matched_iou_sum": float(sum(score for _, _, score in matches)),
        "merge_pred_instances": diagnostics["merge_pred_instances"],
        "split_gt_instances": diagnostics["split_gt_instances"],
        "scale_total": scale_total,
        "scale_matched": scale_matched,
    }


@dataclass
class InstanceMetricAccumulator:
    """Dataset-level instance P/R/F1, IoU, merge/split, and scale recall."""

    target_instances: int = 0
    predicted_instances: int = 0
    matched_instances: int = 0
    matched_iou_sum: float = 0.0
    merge_pred_instances: int = 0
    split_gt_instances: int = 0
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
        self.target_instances += int(report["target_instances"])
        self.predicted_instances += int(report["predicted_instances"])
        self.matched_instances += int(report["matched_instances"])
        self.matched_iou_sum += float(report["matched_iou_sum"])
        self.merge_pred_instances += int(report["merge_pred_instances"])
        self.split_gt_instances += int(report["split_gt_instances"])
        for key in self.scale_total:
            self.scale_total[key] += int(report["scale_total"][key])
            self.scale_matched[key] += int(report["scale_matched"][key])

    def compute(self) -> Dict[str, Any]:
        precision = self.matched_instances / max(1, self.predicted_instances)
        recall = self.matched_instances / max(1, self.target_instances)
        return {
            "target_instances": self.target_instances,
            "predicted_instances": self.predicted_instances,
            "matched_instances": self.matched_instances,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
            "mean_matched_iou": self.matched_iou_sum
            / max(1, self.matched_instances),
            "merge": self.merge_pred_instances,
            "split": self.split_gt_instances,
            "prediction_to_target_ratio": self.predicted_instances
            / max(1, self.target_instances),
            "scale_total": dict(self.scale_total),
            "scale_matched": dict(self.scale_matched),
            "scale_recall": {
                key: self.scale_matched[key] / max(1, self.scale_total[key])
                for key in self.scale_total
            },
        }
