from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def _ids(labels: np.ndarray) -> List[int]:
    return [int(value) for value in np.unique(labels) if int(value) > 0]


def evaluate_instance_labels(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    match_iou_threshold: float = 0.5,
    merge_split_overlap_threshold: float = 0.1,
) -> Dict[str, float | int]:
    gt = np.asarray(gt_labels)
    pred = np.asarray(pred_labels)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError("GT and prediction must be same-shape 2D label maps")
    gt_ids = _ids(gt)
    pred_ids = _ids(pred)
    gt_areas = np.asarray([(gt == value).sum() for value in gt_ids], dtype=np.float64)
    pred_areas = np.asarray(
        [(pred == value).sum() for value in pred_ids], dtype=np.float64
    )
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for gt_index, gt_id in enumerate(gt_ids):
        gt_mask = gt == gt_id
        for pred_index, pred_id in enumerate(pred_ids):
            intersections[gt_index, pred_index] = np.logical_and(
                gt_mask, pred == pred_id
            ).sum()
    if intersections.size:
        unions = gt_areas[:, None] + pred_areas[None, :] - intersections
        iou = intersections / np.maximum(unions, 1.0)
        rows, columns = linear_sum_assignment(1.0 - iou)
        matched = [
            float(iou[row, column])
            for row, column in zip(rows, columns)
            if iou[row, column] >= match_iou_threshold
        ]
    else:
        iou = np.zeros_like(intersections)
        matched = []

    true_positive = len(matched)
    false_positive = len(pred_ids) - true_positive
    false_negative = len(gt_ids) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    if intersections.size:
        # Intersection over the smaller instance is symmetric and detects
        # material many-to-one/one-to-many overlap without background bias.
        overlap = intersections / np.maximum(
            np.minimum(gt_areas[:, None], pred_areas[None, :]), 1.0
        )
        merge = int(
            sum(
                (overlap[:, pred_index] > merge_split_overlap_threshold).sum()
                >= 2
                for pred_index in range(len(pred_ids))
            )
        )
        split = int(
            sum(
                (overlap[gt_index] > merge_split_overlap_threshold).sum() >= 2
                for gt_index in range(len(gt_ids))
            )
        )
    else:
        merge = split = 0
    return {
        "gt_instances": len(gt_ids),
        "pred_instances": len(pred_ids),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(np.mean(matched)) if matched else 0.0,
        "matched_iou_sum": float(np.sum(matched)),
        "merge": merge,
        "split": split,
    }


def aggregate_metric_dicts(rows: Iterable[Dict[str, float | int]]):
    rows = list(rows)
    counts = {
        key: int(sum(int(row.get(key, 0)) for row in rows))
        for key in (
            "gt_instances",
            "pred_instances",
            "true_positive",
            "false_positive",
            "false_negative",
            "merge",
            "split",
        )
    }
    tp = counts["true_positive"]
    precision = tp / max(tp + counts["false_positive"], 1)
    recall = tp / max(tp + counts["false_negative"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    matched_sum = float(
        sum(
            float(
                row.get(
                    "matched_iou_sum",
                    float(row.get("mean_matched_iou", 0.0))
                    * int(row.get("true_positive", 0)),
                )
            )
            for row in rows
        )
    )
    return {
        **counts,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": matched_sum / max(tp, 1),
        "matched_iou_sum": matched_sum,
        "image_count": len(rows),
    }
