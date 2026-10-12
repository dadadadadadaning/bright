"""Pure Stage-0 operators and gates.

The module contains no checkpoint or dataset loading.  It deliberately reuses
the frozen H9 building-mask contract, while keeping all new decisions isolated
from H7, H9, RGID, CMR, and P3 training code.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import connected_components_4
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram,
    metrics_from_histogram,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (
    BuildingInstance,
    aggregate_building_probability,
    oracle_severity_labels,
    rasterize_instances,
)


STAGE0_VARIANTS = (
    "e0_raw_p3",
    "e1_binary_cc",
    "e2_damage_class_cc",
    "e3_learned_rgid",
    "e4_gt_instance_pool",
    "e5_gt_instance_p3_a2_oracle",
)


def _exclusive_instance_map(
    instances: Sequence[BuildingInstance],
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Return a dense 1-based id map, or None when masks overlap."""

    output = np.zeros((height, width), dtype=np.int32)
    for index, instance in enumerate(instances, start=1):
        mask = np.asarray(instance.mask, dtype=np.bool_)
        if tuple(mask.shape) != (height, width):
            raise ValueError("all geometry masks must have the same shape")
        if bool((output[mask] != 0).any()):
            return None
        output[mask] = index
    return output


def _pairwise_intersections(
    predicted: Sequence[BuildingInstance],
    target: Sequence[BuildingInstance],
    *,
    device: torch.device,
    chunk_size: int = 32,
) -> np.ndarray:
    """Compute every exact mask intersection once on the requested device.

    The common non-overlapping case uses two id maps and one bincount.  COCO
    predictions containing overlaps use chunked matrix products instead.
    """

    output = np.zeros((len(predicted), len(target)), dtype=np.int64)
    if not predicted or not target:
        return output
    shape = tuple(np.asarray(predicted[0].mask).shape)
    if len(shape) != 2:
        raise ValueError("geometry masks must be two-dimensional")
    height, width = int(shape[0]), int(shape[1])
    predicted_map = _exclusive_instance_map(predicted, height=height, width=width)
    target_map = _exclusive_instance_map(target, height=height, width=width)
    if predicted_map is not None and target_map is not None:
        predicted_ids = torch.as_tensor(predicted_map, dtype=torch.int64, device=device)
        target_ids = torch.as_tensor(target_map, dtype=torch.int64, device=device)
        active = (predicted_ids > 0) & (target_ids > 0)
        encoded = predicted_ids[active] * (len(target) + 1) + target_ids[active]
        counts = torch.bincount(
            encoded,
            minlength=(len(predicted) + 1) * (len(target) + 1),
        ).reshape(len(predicted) + 1, len(target) + 1)
        return counts[1:, 1:].cpu().numpy().astype(np.int64, copy=False)

    if chunk_size < 1:
        raise ValueError("intersection chunk size must be positive")
    pixels = height * width
    for pred_start in range(0, len(predicted), chunk_size):
        pred_stop = min(pred_start + chunk_size, len(predicted))
        pred_chunk = torch.as_tensor(
            np.stack([predicted[index].mask for index in range(pred_start, pred_stop)]),
            dtype=torch.float32,
            device=device,
        ).reshape(pred_stop - pred_start, pixels)
        for target_start in range(0, len(target), chunk_size):
            target_stop = min(target_start + chunk_size, len(target))
            target_chunk = torch.as_tensor(
                np.stack([target[index].mask for index in range(target_start, target_stop)]),
                dtype=torch.float32,
                device=device,
            ).reshape(target_stop - target_start, pixels)
            intersection = pred_chunk @ target_chunk.transpose(0, 1)
            output[pred_start:pred_stop, target_start:target_stop] = (
                intersection.round().to(torch.int64).cpu().numpy()
            )
    return output


def damage_class_components(
    localization: np.ndarray | torch.Tensor,
    severity_prediction: np.ndarray | torch.Tensor,
    *,
    minimum_area: int,
) -> list[BuildingInstance]:
    """Split the predicted building foreground by severity and 4-connectivity.

    This is intentionally named a component operator, not an instance oracle:
    two touching buildings with the same severity remain merged.
    """

    localization_array = np.asarray(localization, dtype=np.bool_)
    severity_array = np.asarray(severity_prediction, dtype=np.int64)
    if localization_array.shape != severity_array.shape or localization_array.ndim != 2:
        raise ValueError("localization and severity must be same-shape 2D arrays")
    if minimum_area < 1:
        raise ValueError("minimum_area must be positive")
    output: list[BuildingInstance] = []
    for severity in (1, 2, 3):
        mask = localization_array & (severity_array == severity)
        for component in connected_components_4(mask):
            if int(component.sum()) < minimum_area:
                continue
            output.append(
                BuildingInstance(
                    mask=component,
                    label=None,
                    instance_id=len(output) + 1,
                )
            )
    return output


def instances_from_id_map(
    instance_ids: np.ndarray | torch.Tensor,
    *,
    minimum_area: int,
) -> list[BuildingInstance]:
    """Convert a positive integer RGID map to the shared instance contract."""

    array = np.asarray(instance_ids, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("instance id map must be two-dimensional")
    if bool((array < 0).any()):
        raise ValueError("instance id map cannot contain negative ids")
    output: list[BuildingInstance] = []
    for instance_id in np.unique(array):
        if int(instance_id) <= 0:
            continue
        mask = array == int(instance_id)
        if int(mask.sum()) < minimum_area:
            continue
        output.append(
            BuildingInstance(
                mask=mask,
                label=None,
                instance_id=int(instance_id),
            )
        )
    return output


def _pooled_labels_and_confidence(
    probability: torch.Tensor,
    instances: Sequence[BuildingInstance],
) -> tuple[list[int], list[float], np.ndarray | None]:
    """Pool every exclusive region in one device-side reduction."""

    if probability.ndim != 3 or probability.shape[0] != 4:
        raise ValueError("probability must have shape [4,H,W]")
    height, width = int(probability.shape[-2]), int(probability.shape[-1])
    id_map = _exclusive_instance_map(instances, height=height, width=width)
    if not instances:
        return [], [], id_map
    if id_map is None:
        labels: list[int] = []
        confidence: list[float] = []
        for instance in instances:
            distribution = aggregate_building_probability(probability, instance.mask)
            value, index = distribution.max(dim=0)
            labels.append(int(index.item()) + 1)
            confidence.append(float(value.item()))
        return labels, confidence, None

    ids = torch.as_tensor(id_map, dtype=torch.int64, device=probability.device).reshape(-1)
    values = probability[1:4].permute(1, 2, 0).reshape(-1, 3)
    active = ids > 0
    sums = torch.zeros(
        (len(instances) + 1, 3),
        dtype=values.dtype,
        device=probability.device,
    )
    sums.index_add_(0, ids[active], values[active])
    counts = torch.bincount(ids[active], minlength=len(instances) + 1).clamp_min(1)
    means = sums[1:] / counts[1:, None]
    distributions = means / means.sum(dim=1, keepdim=True).clamp_min(1e-12)
    confidence_tensor, label_indices = distributions.max(dim=1)
    labels = (label_indices + 1).cpu().tolist()
    confidence = confidence_tensor.cpu().tolist()
    return [int(value) for value in labels], [float(value) for value in confidence], id_map


def pooled_instance_raster(
    probability: torch.Tensor,
    instances: Sequence[BuildingInstance],
    *,
    height: int,
    width: int,
    preserve_high_score: bool = True,
) -> tuple[torch.Tensor, list[int], list[float], int]:
    """Pool P3 severity probability inside each region and rasterize one label.

    For region R_j and severity c in {1,2,3}:

        s_j(c) = mean_{x in R_j} p(c|x)
        y_j    = argmax_c s_j(c)

    Background is excluded only after geometry has declared the region to be a
    building.  Pixels outside all regions remain background.
    """

    labels, confidence, exclusive_map = _pooled_labels_and_confidence(
        probability, instances
    )
    if exclusive_map is not None:
        lookup = np.asarray([0, *labels], dtype=np.int64)
        return torch.from_numpy(lookup[exclusive_map]), labels, confidence, 0
    raster, overlap = rasterize_instances(
        instances,
        labels,
        height=height,
        width=width,
        preserve_high_score=preserve_high_score,
    )
    return raster, labels, confidence, overlap


def gt_target_raster(
    instances: Sequence[BuildingInstance],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    labels = []
    for instance in instances:
        if instance.label not in (1, 2, 3):
            raise ValueError("official GT instances require severity labels 1..3")
        labels.append(int(instance.label))
    raster, _ = rasterize_instances(
        instances,
        labels,
        height=height,
        width=width,
        preserve_high_score=False,
    )
    return raster


def p3_a2_gt_instance_oracle_raster(
    p3_probability: torch.Tensor,
    a2_probability: torch.Tensor,
    instances: Sequence[BuildingInstance],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Label-aware P3/A2 oracle on official instances; never deployable."""

    targets = [int(instance.label) for instance in instances]
    p3_labels, _, _ = _pooled_labels_and_confidence(p3_probability, instances)
    a2_labels, _, _ = _pooled_labels_and_confidence(a2_probability, instances)
    labels = oracle_severity_labels(p3_labels, a2_labels, targets)
    raster, _ = rasterize_instances(
        instances,
        labels,
        height=height,
        width=width,
        preserve_high_score=False,
    )
    return raster


def building_histogram_from_map(
    prediction: torch.Tensor | np.ndarray,
    target_instances: Sequence[BuildingInstance],
) -> torch.Tensor:
    """Return a 4x4 building confusion using majority label per GT building.

    Rows are GT labels and columns are predicted labels.  A majority background
    prediction is retained as class 0, so a missed building is counted as an
    error rather than silently removed from building-level evaluation.
    """

    prediction_tensor = torch.as_tensor(prediction, dtype=torch.long).cpu()
    if prediction_tensor.ndim != 2:
        raise ValueError("prediction map must be two-dimensional")
    histogram = torch.zeros(4, 4, dtype=torch.int64)
    for instance in target_instances:
        target = int(instance.label) if instance.label is not None else -1
        if target not in (1, 2, 3):
            raise ValueError("target instance label must be 1, 2, or 3")
        mask = torch.as_tensor(instance.mask, dtype=torch.bool)
        if tuple(mask.shape) != tuple(prediction_tensor.shape):
            raise ValueError("prediction and target instance shapes differ")
        values = prediction_tensor[mask]
        if values.numel() == 0:
            predicted = 0
        else:
            if bool(((values < 0) | (values > 3)).any()):
                raise ValueError("prediction labels must lie in 0..3")
            predicted = int(torch.bincount(values, minlength=4).argmax().item())
        histogram[target, predicted] += 1
    return histogram


def building_metrics_from_histogram(histogram: torch.Tensor) -> Dict[str, object]:
    """Compute building severity F1 while retaining missed-background errors."""

    hist = torch.as_tensor(histogram, dtype=torch.float64)
    if tuple(hist.shape) != (4, 4):
        raise ValueError("building histogram must have shape [4,4]")
    tp = torch.diag(hist)
    support = hist.sum(1)
    predicted = hist.sum(0)
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    names = ("intact", "damaged", "destroyed")
    present = support[1:4] > 0
    per_class = {
        name: {
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
            "support": int(support[index].item()),
            "predicted": int(predicted[index].item()),
        }
        for index, name in enumerate(names, start=1)
    }
    macro = f1[1:4][present].mean() if bool(present.any()) else torch.tensor(0.0)
    return {
        "macro_f1_present_gt": float(macro.item()),
        "per_class": per_class,
        "confusion4": hist.to(torch.int64).tolist(),
        "missed_as_background": int(hist[1:4, 0].sum().item()),
    }


def evaluate_geometry(
    predicted: Sequence[BuildingInstance],
    target: Sequence[BuildingInstance],
    *,
    match_iou: float,
    merge_split_overlap: float = 0.1,
    device: torch.device | str | None = None,
    intersection_chunk_size: int = 32,
) -> Dict[str, float | int]:
    """One-to-one IoU matching plus merge/split diagnostics.

    Pair intersections are computed once on ``device`` and then reused by
    matching and merge/split accounting.  This replaces the legacy nested
    full-image ``mask_iou`` loop without changing the metric definition.
    """

    if not 0.0 < match_iou <= 1.0:
        raise ValueError("match_iou must lie in (0,1]")
    geometry_device = torch.device(device) if device is not None else torch.device("cpu")
    intersections = _pairwise_intersections(
        predicted,
        target,
        device=geometry_device,
        chunk_size=intersection_chunk_size,
    )
    target_areas = np.asarray([item.area for item in target], dtype=np.float64)
    predicted_areas = np.asarray([item.area for item in predicted], dtype=np.float64)
    if intersections.size:
        unions = (
            predicted_areas[:, None]
            + target_areas[None, :]
            - intersections.astype(np.float64)
        )
        ious = intersections / np.maximum(unions, 1.0)
        candidates = [
            (float(ious[pred_index, target_index]), pred_index, target_index)
            for pred_index, target_index in np.argwhere(ious >= match_iou)
        ]
        # Keep legacy reverse tuple ordering, including deterministic index
        # tie-breaking, so old and optimized audits remain comparable.
        candidates.sort(reverse=True)
        used_predicted: set[int] = set()
        used_target: set[int] = set()
        matches: list[tuple[int, int, float]] = []
        for score, pred_index, target_index in candidates:
            if pred_index in used_predicted or target_index in used_target:
                continue
            used_predicted.add(pred_index)
            used_target.add(target_index)
            matches.append((pred_index, target_index, score))
    else:
        matches = []
    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(target) - true_positive
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(target), 1)

    if intersections.size:
        overlap = intersections / np.maximum(
            np.minimum(predicted_areas[:, None], target_areas[None, :]), 1.0
        )
        merge = int(
            sum(
                int((overlap[index, :] > merge_split_overlap).sum()) >= 2
                for index in range(len(predicted))
            )
        )
        split = int(
            sum(
                int((overlap[:, index] > merge_split_overlap).sum()) >= 2
                for index in range(len(target))
            )
        )
    else:
        merge = split = 0
    return {
        "gt_instances": len(target),
        "pred_instances": len(predicted),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "matched_iou_sum": float(sum(row[2] for row in matches)),
        "mean_matched_iou": float(sum(row[2] for row in matches) / max(len(matches), 1)),
        "merge": merge,
        "split": split,
    }


def aggregate_geometry(rows: Sequence[Mapping[str, float | int]]) -> Dict[str, float | int]:
    keys = (
        "gt_instances",
        "pred_instances",
        "true_positive",
        "false_positive",
        "false_negative",
        "merge",
        "split",
    )
    counts = {key: int(sum(int(row.get(key, 0)) for row in rows)) for key in keys}
    tp = counts["true_positive"]
    precision = tp / max(tp + counts["false_positive"], 1)
    recall = tp / max(tp + counts["false_negative"], 1)
    matched_iou_sum = float(sum(float(row.get("matched_iou_sum", 0.0)) for row in rows))
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "matched_iou_sum": matched_iou_sum,
        "mean_matched_iou": matched_iou_sum / max(tp, 1),
        "image_count": len(rows),
    }


def new_variant_accumulator() -> Dict[str, object]:
    return {
        "semantic_histogram": torch.zeros(4, 4, dtype=torch.float64),
        "building_histogram": torch.zeros(4, 4, dtype=torch.int64),
        "geometry_rows": [],
        "images": 0,
        "overlap_pixels": 0,
    }


def update_variant_accumulator(
    accumulator: MutableMapping[str, object],
    prediction: torch.Tensor,
    target_map: torch.Tensor,
    target_instances: Sequence[BuildingInstance],
    *,
    geometry: Mapping[str, float | int] | None = None,
    overlap_pixels: int = 0,
) -> None:
    semantic = accumulator["semantic_histogram"]
    building = accumulator["building_histogram"]
    if not torch.is_tensor(semantic) or not torch.is_tensor(building):
        raise TypeError("invalid Stage-0 accumulator")
    semantic.add_(fast_histogram(prediction, target_map))
    building.add_(building_histogram_from_map(prediction, target_instances))
    if geometry is not None:
        rows = accumulator["geometry_rows"]
        if not isinstance(rows, list):
            raise TypeError("invalid geometry accumulator")
        rows.append(dict(geometry))
    accumulator["images"] = int(accumulator["images"]) + 1
    accumulator["overlap_pixels"] = int(accumulator["overlap_pixels"]) + int(overlap_pixels)


def merge_variant_accumulator(
    destination: MutableMapping[str, object],
    source: Mapping[str, object],
) -> None:
    for key in ("semantic_histogram", "building_histogram"):
        left, right = destination[key], source[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise TypeError("invalid Stage-0 accumulator")
        left.add_(right)
    left_rows, right_rows = destination["geometry_rows"], source["geometry_rows"]
    if not isinstance(left_rows, list) or not isinstance(right_rows, list):
        raise TypeError("invalid geometry accumulator")
    left_rows.extend(right_rows)
    destination["images"] = int(destination["images"]) + int(source["images"])
    destination["overlap_pixels"] = int(destination["overlap_pixels"]) + int(source["overlap_pixels"])


def finalize_variant_accumulator(accumulator: Mapping[str, object]) -> Dict[str, object]:
    semantic = accumulator["semantic_histogram"]
    building = accumulator["building_histogram"]
    rows = accumulator["geometry_rows"]
    if not torch.is_tensor(semantic) or not torch.is_tensor(building) or not isinstance(rows, list):
        raise TypeError("invalid Stage-0 accumulator")
    return {
        "images": int(accumulator["images"]),
        "semantic": metrics_from_histogram(semantic),
        "building": building_metrics_from_histogram(building),
        "geometry": aggregate_geometry(rows) if rows else None,
        "overlap_pixels": int(accumulator["overlap_pixels"]),
    }


def _gain_pp(candidate: float, baseline: float) -> float:
    return 100.0 * (float(candidate) - float(baseline))


def build_stage0_decision(
    reports: Mapping[str, Mapping[str, object]],
    event_reports: Sequence[Mapping[str, object]],
    *,
    minimum_valid_miou_gain_pp: float = 0.5,
    minimum_building_macro_f1_gain_pp: float = 1.0,
    minimum_damaged_f1_gain_pp: float = 1.0,
    minimum_nonnegative_events: int = 10,
    maximum_intact_iou_drop_pp: float = 0.5,
) -> Dict[str, object]:
    """Gate T1 using only E4-vs-E0 potential on the same public split."""

    baseline = reports["e0_raw_p3"]
    oracle_geometry = reports["e4_gt_instance_pool"]
    baseline_sem = baseline["semantic"]
    oracle_sem = oracle_geometry["semantic"]
    baseline_building = baseline["building"]
    oracle_building = oracle_geometry["building"]
    if not all(
        isinstance(value, Mapping)
        for value in (baseline_sem, oracle_sem, baseline_building, oracle_building)
    ):
        raise TypeError("Stage-0 reports have an invalid metric contract")
    observed = {
        "valid_miou_gain_pp": _gain_pp(oracle_sem["valid_miou"], baseline_sem["valid_miou"]),
        "building_macro_f1_gain_pp": _gain_pp(
            oracle_building["macro_f1_present_gt"],
            baseline_building["macro_f1_present_gt"],
        ),
        "damaged_f1_gain_pp": _gain_pp(
            oracle_sem["damaged_f1"], baseline_sem["damaged_f1"]
        ),
        "intact_iou_change_pp": _gain_pp(
            oracle_sem["intact_iou"], baseline_sem["intact_iou"]
        ),
        "event_count": len(event_reports),
        "nonnegative_event_count": sum(
            float(row["e4_vs_e0_building_macro_f1_gain_pp"]) >= 0.0
            for row in event_reports
        ),
    }
    checks = {
        "gt_pool_valid_miou_gain": observed["valid_miou_gain_pp"]
        >= minimum_valid_miou_gain_pp,
        "gt_pool_building_macro_f1_gain": observed["building_macro_f1_gain_pp"]
        >= minimum_building_macro_f1_gain_pp,
        "gt_pool_damaged_f1_gain": observed["damaged_f1_gain_pp"]
        >= minimum_damaged_f1_gain_pp,
        "event_nonnegative_count": observed["nonnegative_event_count"]
        >= minimum_nonnegative_events,
        "intact_does_not_collapse": observed["intact_iou_change_pp"]
        >= -maximum_intact_iou_drop_pp,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "observed": observed,
        "thresholds": {
            "minimum_valid_miou_gain_pp": minimum_valid_miou_gain_pp,
            "minimum_building_macro_f1_gain_pp": minimum_building_macro_f1_gain_pp,
            "minimum_damaged_f1_gain_pp": minimum_damaged_f1_gain_pp,
            "minimum_nonnegative_events": minimum_nonnegative_events,
            "maximum_intact_iou_drop_pp": maximum_intact_iou_drop_pp,
        },
        "recommendation": (
            "PROCEED_TO_T1_COMPONENT_BALANCED_JOINT_TRAINING"
            if passed
            else "STOP_SEMANTIC_FEEDBACK_STUDY_KEEP_INSTANCE_ONLY"
        ),
        "warning": "E4 is an official-instance diagnostic ceiling, not a deployable method.",
    }


__all__ = [
    "STAGE0_VARIANTS",
    "aggregate_geometry",
    "build_stage0_decision",
    "building_histogram_from_map",
    "building_metrics_from_histogram",
    "damage_class_components",
    "evaluate_geometry",
    "finalize_variant_accumulator",
    "gt_target_raster",
    "instances_from_id_map",
    "merge_variant_accumulator",
    "new_variant_accumulator",
    "p3_a2_gt_instance_oracle_raster",
    "pooled_instance_raster",
    "update_variant_accumulator",
]
