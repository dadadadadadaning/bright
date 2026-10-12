# -*- coding: utf-8 -*-
"""H9-B0 validation-only building-level P3/A2 oracle audit.

This program does not train or fuse a model.  It measures whether the frozen
P3 and H6-A2 experts remain complementary after predictions are aggregated to
one severity label per official BRIGHT building instance.

Two deliberately separate protocols are reported:

* B0-GT uses the released COCO ground-truth masks.  It isolates the ceiling of
  building-level severity selection and therefore is not deployable.
* B0-Deploy uses either an external COCO instance-prediction JSON or a clearly
  labelled P3 connected-component proxy.  Predicted masks are greedily matched
  to official instances before the label-aware oracle is applied.

Ground truth is used only for validation analysis.  Test/holdout modes are not
implemented by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import imageio.v2 as imageio
from torch.utils.data import DataLoader, Dataset

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    RGB_MEAN,
    RGB_STD,
    load_sar_stats,
    read_data_list,
    to_single_channel_sar,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    build_h6_sdcr_damageformer,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
    connected_components_4,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (  # noqa: E402
    EXPECTED_P3_SHA256,
    VALID_A2_PROTOCOLS,
    checkpoint_state,
    file_sha256,
    fast_histogram,
    metrics_from_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.summarize_H7_event_complementarity import (  # noqa: E402
    disaster_type,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


SEVERITY_NAMES = ("intact", "damaged", "destroyed")
EXPERT_NAMES = ("p3", "a2", "oracle")


class H9RGBSARInferenceDataset(Dataset):
    """RGB/SAR inference loader that does not require semantic target rasters."""

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        sar_stats_path: str,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.data_list = read_data_list(data_list_path)
        if not self.data_list:
            raise ValueError("H9 evaluation list is empty")
        self.sar_stats = load_sar_stats(sar_stats_path)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        name = self.data_list[index]
        pre_path = self.dataset_path / "pre-event" / f"{name}_pre_disaster.tif"
        post_path = self.dataset_path / "post-event" / f"{name}_post_disaster.tif"
        if not pre_path.is_file() or not post_path.is_file():
            raise FileNotFoundError(
                f"missing H9 RGB/SAR pair for {name}: pre={pre_path}, post={post_path}"
            )
        rgb = np.asarray(imageio.imread(pre_path), dtype=np.float32)[..., :3]
        raw_sar = to_single_channel_sar(
            np.asarray(imageio.imread(post_path), dtype=np.float32)
        )
        if rgb.shape[:2] != raw_sar.shape:
            raise ValueError(
                f"unaligned H9 sample {name}: RGB={rgb.shape}, SAR={raw_sar.shape}"
            )
        rgb_tensor = ((rgb - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        clipped = np.clip(
            raw_sar, self.sar_stats["p01"], self.sar_stats["p99"]
        )
        native_sar = (
            clipped - self.sar_stats["mean"]
        ) / self.sar_stats["std"]
        # H6-A2 consumes channel 0; the frozen P3 audit reconstructs its exact
        # historical pseudo-RGB view from the un-clipped display channel 1.
        sar_tensor = np.stack([native_sar, raw_sar], axis=0)
        return (
            np.ascontiguousarray(rgb_tensor, dtype=np.float32),
            np.ascontiguousarray(sar_tensor, dtype=np.float32),
            name,
        )


@dataclass(frozen=True)
class BuildingInstance:
    mask: np.ndarray
    label: int | None
    instance_id: int
    score: float = 1.0

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=np.bool_)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("building mask must be a non-empty [H,W] array")
        if self.label is not None and self.label not in (1, 2, 3):
            raise ValueError("building label must be 1, 2, 3, or None")
        object.__setattr__(self, "mask", mask)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def sample_id_from_file_name(file_name: str) -> str:
    name = Path(str(file_name)).name
    for suffix in (
        "_post_disaster.tif",
        "_pre_disaster.tif",
        "_instance_damage.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def event_name(sample_id: str) -> str:
    parts = str(sample_id).rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(sample_id)


def index_coco_annotations(
    payload: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    """Index a unified BRIGHT COCO file by sample id."""

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO payload requires images and annotations lists")
    if categories is not None:
        if not isinstance(categories, list):
            raise ValueError("COCO categories must be a list")
        observed = {
            int(row["id"]): str(row["name"]).lower()
            for row in categories
            if isinstance(row, Mapping) and "id" in row and "name" in row
        }
        expected = {1: "intact", 2: "damaged", 3: "destroyed"}
        for category_id, category_name in expected.items():
            if observed.get(category_id) != category_name:
                raise ValueError(
                    "COCO category contract mismatch: "
                    f"id={category_id}, expected={category_name!r}, "
                    f"observed={observed.get(category_id)!r}"
                )

    by_image_id: MutableMapping[int, list[Mapping[str, object]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("COCO annotation entries must be objects")
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id not in (1, 2, 3):
            raise ValueError(f"unexpected damage category id: {category_id}")
        by_image_id[image_id].append(annotation)

    output: Dict[str, Dict[str, object]] = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError("COCO image entries must be objects")
        image_id = int(image["id"])
        sample_id = sample_id_from_file_name(str(image["file_name"]))
        if sample_id in output:
            raise ValueError(f"duplicate COCO sample id: {sample_id}")
        output[sample_id] = {
            "image": dict(image),
            "annotations": list(by_image_id.get(image_id, [])),
        }
    return output


def load_official_instance_records(
    sample_ids: Sequence[str],
    *,
    unified_coco_json: str = "",
    per_image_json_dir: str = "",
) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Load exactly the requested split from official COCO annotations.

    The semantic standard_ML split and the cvprw26 challenge split are
    different.  Reading the per-image official JSONs avoids silently pairing
    the 450-image semantic validation list with an unrelated unified val.json.
    """

    if bool(unified_coco_json) == bool(per_image_json_dir):
        raise ValueError(
            "provide exactly one of unified_coco_json or per_image_json_dir"
        )
    requested = list(dict.fromkeys(str(value) for value in sample_ids))
    if unified_coco_json:
        path = Path(unified_coco_json)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        index = index_coco_annotations(payload)
        missing = [sample for sample in requested if sample not in index]
        if missing:
            raise RuntimeError(
                f"unified COCO file is missing {len(missing)} requested samples; "
                f"examples={missing[:5]}"
            )
        selected = {sample: index[sample] for sample in requested}
        return selected, {
            "source": "unified_coco_json",
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "requested_samples": len(requested),
        }

    directory = Path(per_image_json_dir)
    selected: Dict[str, Dict[str, object]] = {}
    digest_rows = []
    missing_paths = []
    for sample in requested:
        path = directory / f"{sample}_instance_damage.json"
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        local_index = index_coco_annotations(payload)
        if sample in local_index:
            record = local_index[sample]
        elif len(local_index) == 1:
            # Some early official per-image files used a generic file_name;
            # the split sample id remains the authoritative pairing key.
            record = next(iter(local_index.values()))
        else:
            raise RuntimeError(
                f"cannot resolve sample {sample!r} inside official JSON {path}"
            )
        selected[sample] = record
        digest_rows.append(f"{sample}\t{file_sha256(path)}")
    if missing_paths:
        raise RuntimeError(
            f"missing {len(missing_paths)} official per-image instance JSONs; "
            f"examples={missing_paths[:5]}"
        )
    import hashlib

    manifest_sha256 = hashlib.sha256(
        ("\n".join(digest_rows) + "\n").encode("utf-8")
    ).hexdigest()
    return selected, {
        "source": "official_per_image_coco_jsons",
        "path": str(directory.resolve()),
        "manifest_sha256": manifest_sha256,
        "requested_samples": len(requested),
    }


def _decode_segmentation(
    segmentation: object,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:  # pragma: no cover - exercised on the server
        raise RuntimeError(
            "H9 COCO mask decoding requires pycocotools. Install the cvprw26 "
            "dependencies before running smoke/val."
        ) from error

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, Mapping):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    else:
        raise ValueError("unsupported COCO segmentation representation")
    decoded = np.asarray(mask_utils.decode(rle))
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    decoded = decoded.astype(np.bool_, copy=False)
    if decoded.shape != (height, width):
        raise ValueError(
            f"decoded COCO mask has shape {decoded.shape}, expected {(height, width)}"
        )
    return decoded


def coco_instances(
    record: Mapping[str, object],
    *,
    minimum_area: int,
) -> list[BuildingInstance]:
    image = record["image"]
    annotations = record["annotations"]
    if not isinstance(image, Mapping) or not isinstance(annotations, list):
        raise TypeError("invalid indexed COCO record")
    height, width = int(image["height"]), int(image["width"])
    output = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        mask = _decode_segmentation(
            annotation["segmentation"], height=height, width=width
        )
        if int(mask.sum()) < minimum_area:
            continue
        output.append(
            BuildingInstance(
                mask=mask,
                label=int(annotation["category_id"]),
                instance_id=int(annotation.get("id", len(output) + 1)),
            )
        )
    return output


def index_prediction_json(payload: object) -> Dict[int, list[Mapping[str, object]]]:
    if isinstance(payload, Mapping):
        payload = payload.get("annotations", payload.get("predictions"))
    if not isinstance(payload, list):
        raise ValueError("instance prediction JSON must contain a list")
    output: MutableMapping[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in payload:
        if not isinstance(row, Mapping):
            raise ValueError("instance prediction entries must be objects")
        output[int(row["image_id"])].append(row)
    return dict(output)


def predicted_coco_instances(
    rows: Sequence[Mapping[str, object]],
    *,
    height: int,
    width: int,
    score_threshold: float,
    minimum_area: int,
    mask_nms_iou: float,
) -> list[BuildingInstance]:
    candidates = []
    for row in rows:
        score = float(row.get("score", 1.0))
        if score < score_threshold:
            continue
        mask = _decode_segmentation(row["segmentation"], height=height, width=width)
        if int(mask.sum()) < minimum_area:
            continue
        candidates.append(
            BuildingInstance(
                mask=mask,
                label=None,
                instance_id=int(row.get("id", len(candidates) + 1)),
                score=score,
            )
        )
    candidates.sort(key=lambda value: (-value.score, -value.area, value.instance_id))
    kept: list[BuildingInstance] = []
    for candidate in candidates:
        if all(mask_iou(candidate.mask, old.mask) < mask_nms_iou for old in kept):
            kept.append(candidate)
    return kept


def components_from_localization(
    localization: np.ndarray,
    *,
    minimum_area: int,
) -> list[BuildingInstance]:
    output = []
    for component in connected_components_4(np.asarray(localization) > 0):
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


def aggregate_building_probability(
    probability: torch.Tensor,
    mask: np.ndarray,
) -> torch.Tensor:
    """Return one normalized intact/damaged/destroyed distribution."""

    if probability.ndim != 3 or probability.shape[0] != 4:
        raise ValueError("probability must have shape [4,H,W]")
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=probability.device)
    if tuple(mask_tensor.shape) != tuple(probability.shape[-2:]):
        raise ValueError("instance mask and expert probability shapes differ")
    values = probability[1:4, mask_tensor]
    if values.numel() == 0:
        raise ValueError("cannot aggregate an empty building")
    mean_probability = values.mean(dim=1)
    return mean_probability / mean_probability.sum().clamp_min(1e-12)


def expert_labels_for_instances(
    probability: torch.Tensor,
    instances: Sequence[BuildingInstance],
) -> tuple[list[int], list[float]]:
    labels, confidence = [], []
    for instance in instances:
        distribution = aggregate_building_probability(probability, instance.mask)
        value, index = distribution.max(dim=0)
        labels.append(int(index.item()) + 1)
        confidence.append(float(value.item()))
    return labels, confidence


def oracle_severity_labels(
    p3_labels: Sequence[int],
    a2_labels: Sequence[int],
    target_labels: Sequence[int],
) -> list[int]:
    if not (len(p3_labels) == len(a2_labels) == len(target_labels)):
        raise ValueError("expert and target label lengths differ")
    return [
        a2 if a2 == target and p3 != target else p3
        for p3, a2, target in zip(p3_labels, a2_labels, target_labels)
    ]


def complementarity_counts(
    p3_labels: Sequence[int],
    a2_labels: Sequence[int],
    target_labels: Sequence[int],
) -> Dict[str, int]:
    counts: MutableMapping[str, int] = defaultdict(int)
    for p3, a2, target in zip(p3_labels, a2_labels, target_labels):
        p3_correct, a2_correct = p3 == target, a2 == target
        category = (
            "both_correct"
            if p3_correct and a2_correct
            else "p3_only_correct"
            if p3_correct
            else "a2_only_correct"
            if a2_correct
            else "neither_correct"
        )
        counts["valid"] += 1
        counts[category] += 1
        counts["disagreement"] += int(p3 != a2)
    for key in (
        "valid",
        "both_correct",
        "p3_only_correct",
        "a2_only_correct",
        "neither_correct",
        "disagreement",
    ):
        counts[key] += 0
    return dict(counts)


def merge_integer_counts(
    destination: MutableMapping[str, int], source: Mapping[str, int]
) -> None:
    for key, value in source.items():
        destination[key] = int(destination.get(key, 0)) + int(value)


def confusion3(
    targets: Sequence[int], predictions: Sequence[int]
) -> torch.Tensor:
    if len(targets) != len(predictions):
        raise ValueError("target and prediction lengths differ")
    histogram = torch.zeros(3, 3, dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        if target not in (1, 2, 3) or prediction not in (1, 2, 3):
            raise ValueError("severity confusion only accepts labels 1..3")
        histogram[target - 1, prediction - 1] += 1
    return histogram


def severity_metrics(histogram: torch.Tensor) -> Dict[str, object]:
    hist = histogram.to(torch.float64)
    true_positive = torch.diag(hist)
    support = hist.sum(1)
    predicted = hist.sum(0)
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    per_class = {
        name: {
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
            "support": int(support[index].item()),
            "predicted": int(predicted[index].item()),
        }
        for index, name in enumerate(SEVERITY_NAMES)
    }
    return {
        "accuracy": float((true_positive.sum() / hist.sum().clamp_min(1.0)).item()),
        "macro_f1_all3": float(f1.mean().item()),
        "macro_f1_present_gt": float(f1[present].mean().item()) if present.any() else 0.0,
        "per_class": per_class,
        "confusion": histogram.to(torch.int64).tolist(),
    }


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.bool_)
    second = np.asarray(second, dtype=np.bool_)
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / max(union, 1)


def greedy_match_masks(
    predicted: Sequence[BuildingInstance],
    target: Sequence[BuildingInstance],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, prediction in enumerate(predicted):
        for target_index, truth in enumerate(target):
            score = mask_iou(prediction.mask, truth.mask)
            if score >= iou_threshold:
                candidates.append((score, pred_index, target_index))
    candidates.sort(reverse=True)
    used_predicted, used_target = set(), set()
    output = []
    for score, pred_index, target_index in candidates:
        if pred_index in used_predicted or target_index in used_target:
            continue
        used_predicted.add(pred_index)
        used_target.add(target_index)
        output.append((pred_index, target_index, score))
    return output


def rasterize_instances(
    instances: Sequence[BuildingInstance],
    labels: Sequence[int],
    *,
    height: int,
    width: int,
    preserve_high_score: bool,
) -> tuple[torch.Tensor, int]:
    if len(instances) != len(labels):
        raise ValueError("instance and label lengths differ")
    canvas = np.zeros((height, width), dtype=np.int64)
    occupied = np.zeros((height, width), dtype=np.bool_)
    overlap_pixels = 0
    order = list(range(len(instances)))
    if preserve_high_score:
        order.sort(key=lambda index: (-instances[index].score, -instances[index].area))
    for index in order:
        mask = instances[index].mask
        overlap_pixels += int(np.logical_and(occupied, mask).sum())
        writable = np.logical_and(mask, ~occupied) if preserve_high_score else mask
        canvas[writable] = int(labels[index])
        occupied |= mask
    return torch.from_numpy(canvas), overlap_pixels


def new_accumulator() -> Dict[str, object]:
    return {
        "images": 0,
        "building_histograms": {
            name: torch.zeros(3, 3, dtype=torch.int64) for name in EXPERT_NAMES
        },
        "semantic_histograms": {
            name: torch.zeros(4, 4, dtype=torch.float64) for name in EXPERT_NAMES
        },
        "counts": defaultdict(int),
        "geometry": defaultdict(float),
        "overlap_pixels": defaultdict(int),
    }


def merge_accumulator(destination: Dict[str, object], source: Dict[str, object]) -> None:
    destination["images"] = int(destination["images"]) + int(source["images"])
    for group in ("building_histograms", "semantic_histograms"):
        left, right = destination[group], source[group]
        assert isinstance(left, Mapping) and isinstance(right, Mapping)
        for name in EXPERT_NAMES:
            left[name].add_(right[name])  # type: ignore[index,union-attr]
    for group in ("counts", "overlap_pixels"):
        left, right = destination[group], source[group]
        assert isinstance(left, MutableMapping) and isinstance(right, Mapping)
        for key, value in right.items():
            left[key] = left.get(key, 0) + value
    left_geometry, right_geometry = destination["geometry"], source["geometry"]
    assert isinstance(left_geometry, MutableMapping) and isinstance(right_geometry, Mapping)
    for key, value in right_geometry.items():
        left_geometry[key] = float(left_geometry.get(key, 0.0)) + float(value)


def update_gt_accumulator(
    accumulator: Dict[str, object],
    instances: Sequence[BuildingInstance],
    p3_probability: torch.Tensor,
    a2_probability: torch.Tensor,
    *,
    height: int,
    width: int,
) -> None:
    targets = [int(instance.label) for instance in instances if instance.label is not None]
    if len(targets) != len(instances):
        raise ValueError("GT audit requires labels for every instance")
    p3_labels, _ = expert_labels_for_instances(p3_probability, instances)
    a2_labels, _ = expert_labels_for_instances(a2_probability, instances)
    oracle_labels = oracle_severity_labels(p3_labels, a2_labels, targets)
    building_histograms = accumulator["building_histograms"]
    semantic_histograms = accumulator["semantic_histograms"]
    assert isinstance(building_histograms, Mapping)
    assert isinstance(semantic_histograms, Mapping)
    for name, labels in (("p3", p3_labels), ("a2", a2_labels), ("oracle", oracle_labels)):
        building_histograms[name].add_(confusion3(targets, labels))  # type: ignore[index,union-attr]
    counts = accumulator["counts"]
    assert isinstance(counts, MutableMapping)
    merge_integer_counts(counts, complementarity_counts(p3_labels, a2_labels, targets))

    target_map, target_overlap = rasterize_instances(
        instances, targets, height=height, width=width, preserve_high_score=False
    )
    overlap_pixels = accumulator["overlap_pixels"]
    assert isinstance(overlap_pixels, MutableMapping)
    overlap_pixels["target"] += target_overlap
    for name, labels in (("p3", p3_labels), ("a2", a2_labels), ("oracle", oracle_labels)):
        prediction_map, overlap = rasterize_instances(
            instances, labels, height=height, width=width, preserve_high_score=False
        )
        overlap_pixels[name] += overlap
        semantic_histograms[name].add_(fast_histogram(prediction_map, target_map))  # type: ignore[index,union-attr]
    accumulator["images"] = int(accumulator["images"]) + 1


def update_deploy_accumulator(
    accumulator: Dict[str, object],
    predicted: Sequence[BuildingInstance],
    target: Sequence[BuildingInstance],
    p3_probability: torch.Tensor,
    a2_probability: torch.Tensor,
    *,
    height: int,
    width: int,
    match_iou: float,
) -> None:
    p3_labels, _ = expert_labels_for_instances(p3_probability, predicted)
    a2_labels, _ = expert_labels_for_instances(a2_probability, predicted)
    matches = greedy_match_masks(predicted, target, iou_threshold=match_iou)
    matched_pred = [row[0] for row in matches]
    matched_target = [row[1] for row in matches]
    matched_targets = [int(target[index].label) for index in matched_target]
    matched_p3 = [p3_labels[index] for index in matched_pred]
    matched_a2 = [a2_labels[index] for index in matched_pred]
    matched_oracle = oracle_severity_labels(matched_p3, matched_a2, matched_targets)

    building_histograms = accumulator["building_histograms"]
    semantic_histograms = accumulator["semantic_histograms"]
    assert isinstance(building_histograms, Mapping)
    assert isinstance(semantic_histograms, Mapping)
    for name, labels in (("p3", matched_p3), ("a2", matched_a2), ("oracle", matched_oracle)):
        building_histograms[name].add_(confusion3(matched_targets, labels))  # type: ignore[index,union-attr]
    counts = accumulator["counts"]
    assert isinstance(counts, MutableMapping)
    merge_integer_counts(
        counts, complementarity_counts(matched_p3, matched_a2, matched_targets)
    )

    oracle_all = list(p3_labels)
    for local_index, pred_index in enumerate(matched_pred):
        oracle_all[pred_index] = matched_oracle[local_index]
    target_labels = [int(instance.label) for instance in target]
    target_map, target_overlap = rasterize_instances(
        target, target_labels, height=height, width=width, preserve_high_score=False
    )
    overlap_pixels = accumulator["overlap_pixels"]
    assert isinstance(overlap_pixels, MutableMapping)
    overlap_pixels["target"] += target_overlap
    for name, labels in (("p3", p3_labels), ("a2", a2_labels), ("oracle", oracle_all)):
        prediction_map, overlap = rasterize_instances(
            predicted, labels, height=height, width=width, preserve_high_score=True
        )
        overlap_pixels[name] += overlap
        semantic_histograms[name].add_(fast_histogram(prediction_map, target_map))  # type: ignore[index,union-attr]

    geometry = accumulator["geometry"]
    assert isinstance(geometry, MutableMapping)
    geometry["target_instances"] += len(target)
    geometry["predicted_instances"] += len(predicted)
    geometry["matched_instances"] += len(matches)
    geometry["matched_iou_sum"] += sum(row[2] for row in matches)
    accumulator["images"] = int(accumulator["images"]) + 1


def finalize_accumulator(accumulator: Mapping[str, object]) -> Dict[str, object]:
    building_histograms = accumulator["building_histograms"]
    semantic_histograms = accumulator["semantic_histograms"]
    geometry = accumulator["geometry"]
    assert isinstance(building_histograms, Mapping)
    assert isinstance(semantic_histograms, Mapping)
    assert isinstance(geometry, Mapping)
    target_instances = float(geometry.get("target_instances", 0.0))
    predicted_instances = float(geometry.get("predicted_instances", 0.0))
    matched_instances = float(geometry.get("matched_instances", 0.0))
    precision = matched_instances / max(predicted_instances, 1.0)
    recall = matched_instances / max(target_instances, 1.0)
    geometry_report = {
        "target_instances": int(target_instances),
        "predicted_instances": int(predicted_instances),
        "matched_instances": int(matched_instances),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "mean_matched_iou": float(geometry.get("matched_iou_sum", 0.0))
        / max(matched_instances, 1.0),
    }
    return {
        "images": int(accumulator["images"]),
        "building_metrics": {
            name: severity_metrics(building_histograms[name])  # type: ignore[index]
            for name in EXPERT_NAMES
        },
        "semantic_metrics": {
            name: metrics_from_histogram(semantic_histograms[name])  # type: ignore[index]
            for name in EXPERT_NAMES
        },
        "counts": dict(accumulator["counts"]),  # type: ignore[arg-type]
        "geometry": geometry_report,
        "overlap_pixels": dict(accumulator["overlap_pixels"]),  # type: ignore[arg-type]
    }


def oracle_gains(report: Mapping[str, object]) -> Dict[str, float]:
    building = report["building_metrics"]
    semantic = report["semantic_metrics"]
    assert isinstance(building, Mapping) and isinstance(semantic, Mapping)
    p3_building, oracle_building = building["p3"], building["oracle"]
    p3_semantic, oracle_semantic = semantic["p3"], semantic["oracle"]
    assert isinstance(p3_building, Mapping) and isinstance(oracle_building, Mapping)
    assert isinstance(p3_semantic, Mapping) and isinstance(oracle_semantic, Mapping)
    p3_class = p3_building["per_class"]
    oracle_class = oracle_building["per_class"]
    assert isinstance(p3_class, Mapping) and isinstance(oracle_class, Mapping)
    return {
        "building_macro_f1_pp": 100.0
        * (
            float(oracle_building["macro_f1_present_gt"])
            - float(p3_building["macro_f1_present_gt"])
        ),
        "building_damaged_f1_pp": 100.0
        * (
            float(oracle_class["damaged"]["f1"])  # type: ignore[index]
            - float(p3_class["damaged"]["f1"])  # type: ignore[index]
        ),
        "semantic_valid_miou_pp": 100.0
        * (float(oracle_semantic["valid_miou"]) - float(p3_semantic["valid_miou"])),
        "semantic_miou4_pp": 100.0
        * (float(oracle_semantic["miou4"]) - float(p3_semantic["miou4"])),
    }


def build_decision(
    gt_report: Mapping[str, object],
    deploy_report: Mapping[str, object],
    event_reports: Sequence[Mapping[str, object]],
    *,
    deploy_is_proxy: bool,
    minimum_macro_f1_gain_pp: float,
    minimum_damaged_f1_gain_pp: float,
    minimum_positive_events: int,
    minimum_deploy_valid_gain_pp: float,
) -> Dict[str, object]:
    gt_gain = oracle_gains(gt_report)
    deploy_gain = oracle_gains(deploy_report)
    positive_events = sum(
        float(row["gt_oracle_gains"]["building_macro_f1_pp"]) > 1e-9  # type: ignore[index]
        for row in event_reports
    )
    checks = {
        "gt_building_macro_f1": gt_gain["building_macro_f1_pp"]
        >= minimum_macro_f1_gain_pp,
        "gt_damaged_f1": gt_gain["building_damaged_f1_pp"]
        >= minimum_damaged_f1_gain_pp,
        "positive_events": positive_events >= minimum_positive_events,
        "deploy_semantic_valid_miou": deploy_gain["semantic_valid_miou_pp"]
        >= minimum_deploy_valid_gain_pp,
    }
    passed = all(checks.values())
    if not checks["gt_building_macro_f1"] or not checks["gt_damaged_f1"]:
        recommendation = "STOP_H9_BUILDING_SELECTION_UPPER_BOUND_TOO_LOW"
    elif not checks["positive_events"]:
        recommendation = "STOP_OR_REDESIGN_FOR_CROSS_EVENT_STABILITY"
    elif not checks["deploy_semantic_valid_miou"]:
        recommendation = "INSTANCE_GEOMETRY_IS_THE_CURRENT_BOTTLENECK"
    elif deploy_is_proxy:
        recommendation = "PROVISIONAL_PASS_REQUIRE_REAL_INSTANCE_PREDICTIONS"
    else:
        recommendation = "IMPLEMENT_H9_B1_BUILDING_SELECTOR"
    return {
        "passed": passed and not deploy_is_proxy,
        "provisional_pass": passed,
        "recommendation": recommendation,
        "checks": checks,
        "observed": {
            "gt_oracle_gains": gt_gain,
            "deploy_oracle_gains": deploy_gain,
            "positive_event_count": positive_events,
            "event_count": len(event_reports),
            "deploy_is_proxy": deploy_is_proxy,
        },
        "thresholds": {
            "minimum_macro_f1_gain_pp": minimum_macro_f1_gain_pp,
            "minimum_damaged_f1_gain_pp": minimum_damaged_f1_gain_pp,
            "minimum_positive_events": minimum_positive_events,
            "minimum_deploy_valid_gain_pp": minimum_deploy_valid_gain_pp,
        },
    }


def report_row(
    name: str,
    gt_accumulator: Mapping[str, object],
    deploy_accumulator: Mapping[str, object],
) -> Dict[str, object]:
    gt_report = finalize_accumulator(gt_accumulator)
    deploy_report = finalize_accumulator(deploy_accumulator)
    return {
        "name": name,
        "disaster_type": disaster_type(name),
        "gt": gt_report,
        "deploy": deploy_report,
        "gt_oracle_gains": oracle_gains(gt_report),
        "deploy_oracle_gains": oracle_gains(deploy_report),
    }


def aggregate_disaster_rows(
    events: Mapping[str, tuple[Dict[str, object], Dict[str, object]]]
) -> list[Dict[str, object]]:
    grouped: MutableMapping[str, tuple[Dict[str, object], Dict[str, object]]] = {}
    for event, (gt_accumulator, deploy_accumulator) in events.items():
        type_name = disaster_type(event)
        if type_name not in grouped:
            grouped[type_name] = (new_accumulator(), new_accumulator())
        merge_accumulator(grouped[type_name][0], gt_accumulator)
        merge_accumulator(grouped[type_name][1], deploy_accumulator)
    return [
        report_row(type_name, pair[0], pair[1])
        for type_name, pair in sorted(grouped.items())
    ]


def write_event_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "event",
        "disaster_type",
        "gt_p3_macro_f1",
        "gt_a2_macro_f1",
        "gt_oracle_macro_f1",
        "gt_oracle_gain_pp",
        "gt_damaged_gain_pp",
        "gt_a2_only_correct",
        "gt_p3_only_correct",
        "deploy_oracle_valid_gain_pp",
        "deploy_geometry_recall",
        "deploy_matched_instances",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            gt, deploy = row["gt"], row["deploy"]
            assert isinstance(gt, Mapping) and isinstance(deploy, Mapping)
            gt_building = gt["building_metrics"]
            assert isinstance(gt_building, Mapping)
            writer.writerow(
                {
                    "event": row["name"],
                    "disaster_type": row["disaster_type"],
                    "gt_p3_macro_f1": gt_building["p3"]["macro_f1_present_gt"],  # type: ignore[index]
                    "gt_a2_macro_f1": gt_building["a2"]["macro_f1_present_gt"],  # type: ignore[index]
                    "gt_oracle_macro_f1": gt_building["oracle"]["macro_f1_present_gt"],  # type: ignore[index]
                    "gt_oracle_gain_pp": row["gt_oracle_gains"]["building_macro_f1_pp"],  # type: ignore[index]
                    "gt_damaged_gain_pp": row["gt_oracle_gains"]["building_damaged_f1_pp"],  # type: ignore[index]
                    "gt_a2_only_correct": gt["counts"].get("a2_only_correct", 0),  # type: ignore[union-attr]
                    "gt_p3_only_correct": gt["counts"].get("p3_only_correct", 0),  # type: ignore[union-attr]
                    "deploy_oracle_valid_gain_pp": row["deploy_oracle_gains"]["semantic_valid_miou_pp"],  # type: ignore[index]
                    "deploy_geometry_recall": deploy["geometry"]["recall"],  # type: ignore[index]
                    "deploy_matched_instances": deploy["geometry"]["matched_instances"],  # type: ignore[index]
                }
            )


def write_class_tsv(path: Path, reports: Mapping[str, Mapping[str, object]]) -> None:
    fields = ["protocol", "expert", "class", "precision", "recall", "f1", "support"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for protocol, report in reports.items():
            building = report["building_metrics"]
            assert isinstance(building, Mapping)
            for expert in EXPERT_NAMES:
                per_class = building[expert]["per_class"]  # type: ignore[index]
                for class_name in SEVERITY_NAMES:
                    metrics = per_class[class_name]
                    writer.writerow(
                        {
                            "protocol": protocol,
                            "expert": expert,
                            "class": class_name,
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1": metrics["f1"],
                            "support": metrics["support"],
                        }
                    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("H9-B0 validation-only building oracle audit")
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--a2_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    annotation_group = parser.add_mutually_exclusive_group(required=True)
    annotation_group.add_argument("--instance_coco_json", default="")
    annotation_group.add_argument("--instance_json_dir", default="")
    parser.add_argument("--deploy_predictions_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_event_tsv", default="")
    parser.add_argument("--output_disaster_tsv", default="")
    parser.add_argument("--output_class_tsv", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--minimum_gt_area", type=int, default=1)
    parser.add_argument("--minimum_predicted_area", type=int, default=32)
    parser.add_argument("--deploy_score_threshold", type=float, default=0.05)
    parser.add_argument("--deploy_mask_nms_iou", type=float, default=0.7)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_sha256", default=EXPECTED_P3_SHA256)
    parser.add_argument("--minimum_macro_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_damaged_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_positive_events", type=int, default=10)
    parser.add_argument("--minimum_deploy_valid_gain_pp", type=float, default=0.5)
    parser.add_argument("--evaluation_protocol", default="cvprw26_public_val")
    parser.add_argument("--expected_evaluation_images", type=int, default=697)
    args = parser.parse_args()

    if not 0.0 <= args.deploy_score_threshold <= 1.0:
        raise ValueError("deploy score threshold must lie in [0,1]")
    if not 0.0 < args.match_iou <= 1.0:
        raise ValueError("match IoU must lie in (0,1]")
    set_seed(args.seed)
    device = torch.device(args.device)

    p3_hash = file_sha256(args.p3_checkpoint)
    if args.expected_p3_sha256 and p3_hash != args.expected_p3_sha256:
        raise RuntimeError(
            f"P3 SHA256 mismatch: expected={args.expected_p3_sha256}, observed={p3_hash}"
        )
    p3_checkpoint = torch.load(args.p3_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL()
    incompatible = p3.load_state_dict(checkpoint_state(p3_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")

    a2_checkpoint = torch.load(args.a2_checkpoint, map_location="cpu", weights_only=False)
    if str(a2_checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(f"native expert must be H6-A2, got {a2_checkpoint.get('variant')!r}")
    if a2_checkpoint.get("protocol") not in VALID_A2_PROTOCOLS:
        raise RuntimeError(f"unexpected H6-A2 protocol: {a2_checkpoint.get('protocol')!r}")
    a2 = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = a2.load_state_dict(checkpoint_state(a2_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H6-A2 load failed: {incompatible}")
    for model in (p3, a2):
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    dataset = H9RGBSARInferenceDataset(
        args.dataset_path,
        args.val_data_list_path,
        args.sar_stats_path,
    )
    if (
        args.expected_evaluation_images > 0
        and len(dataset) != args.expected_evaluation_images
    ):
        raise RuntimeError(
            "H9 evaluation split size mismatch: "
            f"expected={args.expected_evaluation_images}, observed={len(dataset)}"
        )
    coco_index, annotation_manifest = load_official_instance_records(
        dataset.data_list,
        unified_coco_json=args.instance_coco_json,
        per_image_json_dir=args.instance_json_dir,
    )
    prediction_index: Dict[int, list[Mapping[str, object]]] | None = None
    deploy_is_proxy = not bool(args.deploy_predictions_json)
    if args.deploy_predictions_json:
        if args.instance_json_dir:
            raise RuntimeError(
                "external deploy predictions require --instance_coco_json built for "
                "the same validation split; per-image COCO files may reuse image_id"
            )
        prediction_payload = json.loads(
            Path(args.deploy_predictions_json).read_text(encoding="utf-8-sig")
        )
        prediction_index = index_prediction_json(prediction_payload)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    overall_gt, overall_deploy = new_accumulator(), new_accumulator()
    event_accumulators: MutableMapping[
        str, tuple[Dict[str, object], Dict[str, object]]
    ] = {}
    processed_images = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb, sar, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        p3_loc_logits, p3_damage_logits = p3(rgb, p3_sar_view(sar))
        a2_outputs = a2(rgb, sar, return_dict=True)
        a2_damage_logits = a2_outputs["native_output_dam"]
        p3_probabilities = p3_damage_logits.float().softmax(1).cpu()
        a2_probabilities = a2_damage_logits.float().softmax(1).cpu()
        p3_localization = p3_loc_logits.float().argmax(1).cpu().numpy()

        for sample_index, raw_name in enumerate(names):
            sample_name = str(raw_name)
            record = coco_index[sample_name]
            image = record["image"]
            assert isinstance(image, Mapping)
            height, width = int(image["height"]), int(image["width"])
            if tuple(p3_probabilities[sample_index].shape[-2:]) != (height, width):
                raise RuntimeError(
                    f"prediction/COCO geometry mismatch for {sample_name}: "
                    f"prediction={tuple(p3_probabilities[sample_index].shape[-2:])}, "
                    f"COCO={(height, width)}"
                )
            gt_instances = coco_instances(record, minimum_area=args.minimum_gt_area)
            if prediction_index is not None:
                predicted_instances = predicted_coco_instances(
                    prediction_index.get(int(image["id"]), []),
                    height=height,
                    width=width,
                    score_threshold=args.deploy_score_threshold,
                    minimum_area=args.minimum_predicted_area,
                    mask_nms_iou=args.deploy_mask_nms_iou,
                )
            else:
                predicted_instances = components_from_localization(
                    p3_localization[sample_index],
                    minimum_area=args.minimum_predicted_area,
                )

            sample_gt, sample_deploy = new_accumulator(), new_accumulator()
            update_gt_accumulator(
                sample_gt,
                gt_instances,
                p3_probabilities[sample_index],
                a2_probabilities[sample_index],
                height=height,
                width=width,
            )
            update_deploy_accumulator(
                sample_deploy,
                predicted_instances,
                gt_instances,
                p3_probabilities[sample_index],
                a2_probabilities[sample_index],
                height=height,
                width=width,
                match_iou=args.match_iou,
            )
            merge_accumulator(overall_gt, sample_gt)
            merge_accumulator(overall_deploy, sample_deploy)
            current_event = event_name(sample_name)
            if current_event not in event_accumulators:
                event_accumulators[current_event] = (new_accumulator(), new_accumulator())
            merge_accumulator(event_accumulators[current_event][0], sample_gt)
            merge_accumulator(event_accumulators[current_event][1], sample_deploy)
            processed_images += 1
            if processed_images % 50 == 0:
                print(f"H9 building oracle: {processed_images}/{len(dataset)}", flush=True)

    gt_report = finalize_accumulator(overall_gt)
    deploy_report = finalize_accumulator(overall_deploy)
    event_reports = [
        report_row(event, pair[0], pair[1])
        for event, pair in sorted(event_accumulators.items())
    ]
    disaster_reports = aggregate_disaster_rows(event_accumulators)
    decision = build_decision(
        gt_report,
        deploy_report,
        event_reports,
        deploy_is_proxy=deploy_is_proxy,
        minimum_macro_f1_gain_pp=args.minimum_macro_f1_gain_pp,
        minimum_damaged_f1_gain_pp=args.minimum_damaged_f1_gain_pp,
        minimum_positive_events=args.minimum_positive_events,
        minimum_deploy_valid_gain_pp=args.minimum_deploy_valid_gain_pp,
    )

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_tsv = Path(args.output_event_tsv) if args.output_event_tsv else output.with_suffix(".events.tsv")
    disaster_tsv = Path(args.output_disaster_tsv) if args.output_disaster_tsv else output.with_suffix(".disasters.tsv")
    class_tsv = Path(args.output_class_tsv) if args.output_class_tsv else output.with_suffix(".classes.tsv")
    payload = {
        "protocol": "H9_B0_building_oracle_validation_v2",
        "evaluation_protocol": args.evaluation_protocol,
        "split": "public_val",
        "test_accessed": False,
        "oracle_is_deployable": False,
        "processed_images": processed_images,
        "checkpoints": {
            "p3": {
                "path": str(Path(args.p3_checkpoint).resolve()),
                "sha256": p3_hash,
                "step": int(p3_checkpoint.get("step", 0)),
                "strict_load": True,
            },
            "a2": {
                "path": str(Path(args.a2_checkpoint).resolve()),
                "sha256": file_sha256(args.a2_checkpoint),
                "protocol": a2_checkpoint.get("protocol"),
                "variant": a2_checkpoint.get("variant"),
                "step": int(a2_checkpoint.get("step", 0)),
                "strict_load": True,
            },
        },
        "data_contract": {
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "validation_list": str(Path(args.val_data_list_path).resolve()),
            "instance_annotations": annotation_manifest,
            "gt_geometry": "official_BRIGHT_COCO_instance_masks",
            "severity_aggregation": "mean_class_probability_renormalized_over_1_2_3",
            "deploy_geometry": (
                "external_COCO_predictions"
                if prediction_index is not None
                else "P3_localization_4_connected_component_proxy"
            ),
            "deploy_predictions_json": (
                str(Path(args.deploy_predictions_json).resolve())
                if args.deploy_predictions_json
                else None
            ),
            "deploy_is_proxy": deploy_is_proxy,
            "match_iou": args.match_iou,
        },
        "b0_gt": gt_report,
        "b0_gt_oracle_gains": oracle_gains(gt_report),
        "b0_deploy": deploy_report,
        "b0_deploy_oracle_gains": oracle_gains(deploy_report),
        "events": event_reports,
        "disaster_types": disaster_reports,
        "decision": decision,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_event_tsv(event_tsv, event_reports)
    write_event_tsv(disaster_tsv, disaster_reports)
    write_class_tsv(class_tsv, {"b0_gt": gt_report, "b0_deploy": deploy_report})

    gt_gain, deploy_gain = oracle_gains(gt_report), oracle_gains(deploy_report)
    gt_building = gt_report["building_metrics"]
    assert isinstance(gt_building, Mapping)
    print("=" * 88)
    print("H9-B0 validation-only building-level P3/A2 oracle audit")
    print(f"Processed images       : {processed_images}")
    print(f"Official GT buildings  : {gt_report['counts'].get('valid', 0):,}")  # type: ignore[union-attr]
    for name in EXPERT_NAMES:
        metrics = gt_building[name]
        print(
            f"B0-GT {name.upper():<6} | Macro-F1={100*metrics['macro_f1_present_gt']:.4f} | "  # type: ignore[index]
            f"Dmg-F1={100*metrics['per_class']['damaged']['f1']:.4f}"  # type: ignore[index]
        )
    geometry = deploy_report["geometry"]
    print(
        f"Deploy geometry        : matched={geometry['matched_instances']:,}/"  # type: ignore[index]
        f"{geometry['target_instances']:,} | recall={100*geometry['recall']:.2f}% | "  # type: ignore[index]
        f"proxy={deploy_is_proxy}"
    )
    print(
        f"GT Oracle gains        : Macro-F1={gt_gain['building_macro_f1_pp']:+.4f} pp | "
        f"Dmg-F1={gt_gain['building_damaged_f1_pp']:+.4f} pp | "
        f"Valid-mIoU={gt_gain['semantic_valid_miou_pp']:+.4f} pp"
    )
    print(
        f"Deploy Oracle gains    : Valid-mIoU={deploy_gain['semantic_valid_miou_pp']:+.4f} pp"
    )
    observed = decision["observed"]
    print(
        f"Positive events        : {observed['positive_event_count']}/"  # type: ignore[index]
        f"{observed['event_count']}"
    )
    print(f"Decision               : {decision['recommendation']}")
    print(f"Saved JSON             : {output}")
    print(f"Saved event TSV        : {event_tsv}")
    print(f"Saved disaster TSV     : {disaster_tsv}")
    print(f"Saved class TSV        : {class_tsv}")
    print("Test accessed          : False")
    print("=" * 88)


if __name__ == "__main__":
    main()
