#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from variant_common import atomic_json, binary_cc, torch_load


def _aggregate_logits(
    logits: np.ndarray,
    pixel_prediction: np.ndarray,
    instances: np.ndarray,
    class_ids: tuple[int, ...],
) -> np.ndarray:
    output = pixel_prediction.copy()
    for instance_id in (int(v) for v in np.unique(instances) if int(v) > 0):
        mask = instances == instance_id
        scores = np.asarray([logits[class_id][mask].mean() for class_id in class_ids])
        output[mask] = class_ids[int(np.argmax(scores))]
    return output


def _confusion(gt: np.ndarray, pred: np.ndarray, classes: tuple[int, ...], ignore: int):
    valid = gt != ignore
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    index = {value: i for i, value in enumerate(classes)}
    for true_value, pred_value in zip(gt[valid].ravel(), pred[valid].ravel()):
        if int(true_value) in index and int(pred_value) in index:
            matrix[index[int(true_value)], index[int(pred_value)]] += 1
    return matrix


def _metrics(matrix: np.ndarray, classes: tuple[int, ...]) -> dict:
    rows = {}
    for index, class_id in enumerate(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1)
        rows[str(class_id)] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        }
    return {
        "classes": rows,
        "macro_f1": float(np.mean([row["f1"] for row in rows.values()])),
        "mean_iou": float(np.mean([row["iou"] for row in rows.values()])),
        "pixel_accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "confusion_matrix": matrix.tolist(),
    }


def _instance_votes(
    gt_instances: np.ndarray,
    damage_gt: np.ndarray,
    prediction: np.ndarray,
    classes: tuple[int, ...],
) -> Counter:
    counts = Counter()
    valid_classes = np.asarray(classes)
    for instance_id in (int(v) for v in np.unique(gt_instances) if int(v) > 0):
        mask = gt_instances == instance_id
        true_values = damage_gt[mask]
        pred_values = prediction[mask]
        true_values = true_values[np.isin(true_values, valid_classes)]
        pred_values = pred_values[np.isin(pred_values, valid_classes)]
        if not true_values.size or not pred_values.size:
            continue
        true_class = int(np.bincount(true_values.astype(np.int64)).argmax())
        pred_class = int(np.bincount(pred_values.astype(np.int64)).argmax())
        counts[(true_class, pred_class)] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        "Compare pixel damage, parent aggregation and RGID instance aggregation"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", default="1,2,3")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    classes = tuple(int(v) for v in args.classes.split(",") if v.strip())
    files = sorted(Path(args.input_dir).expanduser().resolve().glob("*.pt"))
    if args.limit is not None:
        files = files[: max(int(args.limit), 0)]
    if not files:
        raise FileNotFoundError(f"No .pt inputs: {args.input_dir}")

    matrices = {mode: np.zeros((len(classes), len(classes)), dtype=np.int64) for mode in "abc"}
    building_votes = {mode: Counter() for mode in "abc"}
    event_matrices = defaultdict(
        lambda: {mode: np.zeros((len(classes), len(classes)), dtype=np.int64) for mode in "abc"}
    )
    required = {"damage_logits", "damage_gt", "parent_mask", "rgid_instance_ids"}
    for index, path in enumerate(files, start=1):
        payload = torch_load(path, weights_only=False)
        missing = sorted(required.difference(payload))
        if missing:
            raise KeyError(
                f"{path} lacks {missing}. Export these fields from the base damage model "
                "and the selected RGID G3 decoder before Stage 4."
            )
        logits = np.asarray(torch.as_tensor(payload["damage_logits"]).float())
        damage_gt = np.asarray(torch.as_tensor(payload["damage_gt"]).long())
        parent = np.asarray(torch.as_tensor(payload["parent_mask"]).bool())
        rgid = np.asarray(torch.as_tensor(payload["rgid_instance_ids"]).long())
        if logits.ndim != 3 or damage_gt.shape != logits.shape[-2:]:
            raise ValueError(f"Bad damage tensor shape: {path}")
        pixel = logits.argmax(axis=0).astype(np.int32)
        parent_instances = binary_cc(parent, min_area=1)
        predictions = {
            "a": pixel,
            "b": _aggregate_logits(logits, pixel, parent_instances, classes),
            "c": _aggregate_logits(logits, pixel, rgid, classes),
        }
        event = str(payload.get("event", payload.get("metadata", {}).get("event", "unknown")))
        for mode, prediction in predictions.items():
            matrix = _confusion(damage_gt, prediction, classes, args.ignore_index)
            matrices[mode] += matrix
            event_matrices[event][mode] += matrix
            if "gt_instance_ids" in payload:
                gt_instances = np.asarray(torch.as_tensor(payload["gt_instance_ids"]).long())
                building_votes[mode].update(
                    _instance_votes(gt_instances, damage_gt, prediction, classes)
                )
        print(f"[{index}/{len(files)}] {path.name}", flush=True)

    labels = {"a": "pixel", "b": "parent_aggregation", "c": "rgid_aggregation"}
    report = {
        "protocol": "rgid_damage_aggregation_abc_v1",
        "classes": list(classes),
        "sample_count": len(files),
        "methods": {},
        "events": {},
    }
    for mode, label in labels.items():
        result = _metrics(matrices[mode], classes)
        if building_votes[mode]:
            matrix = np.zeros_like(matrices[mode])
            index = {value: i for i, value in enumerate(classes)}
            for (true_class, pred_class), count in building_votes[mode].items():
                if true_class in index and pred_class in index:
                    matrix[index[true_class], index[pred_class]] += count
            result["building_damage_conditional_on_gt_instances"] = _metrics(matrix, classes)
        report["methods"][label] = result
    for event, values in event_matrices.items():
        report["events"][event] = {
            labels[mode]: _metrics(matrix, classes)
            for mode, matrix in values.items()
        }
    atomic_json(report, args.output)
    print(json.dumps(report["methods"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
