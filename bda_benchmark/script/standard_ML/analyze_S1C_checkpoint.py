# -*- coding: utf-8 -*-
"""Comprehensive S1-C checkpoint analysis for BRIGHT.

Outputs:
- overall_metrics.json
- confusion_matrix_counts.csv
- confusion_matrix_row_normalized_percent.csv
- class_metrics.csv
- per_event_metrics.csv
- per_sample_metrics.csv
- val_test_event_comparison.csv (when --split both)

The event id is derived from the official BRIGHT sample id by removing the final
numeric suffix, e.g. bata-explosion_00000007 -> bata-explosion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"
for import_root in (PROJECT_ROOT, BDA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_S1C import DamageFormer_S1C

CLASS_NAMES = ["background", "intact", "damaged", "destroyed"]
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive S1-C checkpoint analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--guidance_beta", type=float, default=1.0)
    p.add_argument("--split", choices=["val", "test", "both"], default="both")
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--val_dataset_path", required=True)
    p.add_argument("--val_data_list_path", required=True)
    p.add_argument("--test_dataset_path", required=True)
    p.add_argument("--test_data_list_path", required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def read_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_state(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


def sample_to_event(sample_id: str) -> str:
    if "_" not in sample_id:
        return sample_id
    prefix, suffix = sample_id.rsplit("_", 1)
    return prefix if suffix.isdigit() else sample_id


def confusion_from_arrays(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (gt >= 0) & (gt < num_classes)
    encoded = num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def metrics_from_confusion(cm: np.ndarray, class_names: list[str]) -> Dict[str, Any]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt_support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)
    fp = pred_support - tp
    fn = gt_support - tp
    union = gt_support + pred_support - tp

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    iou = tp / (union + EPS)
    oa = tp.sum() / (cm.sum() + EPS)
    official_miou = float(np.mean(iou))
    present = union > 0
    present_miou = float(np.mean(iou[present])) if np.any(present) else float("nan")

    class_metrics: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(class_names):
        class_metrics[name] = {
            "precision_percent": 100.0 * float(precision[idx]),
            "recall_percent": 100.0 * float(recall[idx]),
            "f1_percent": 100.0 * float(f1[idx]),
            "iou_percent": 100.0 * float(iou[idx]),
            "gt_pixels": int(gt_support[idx]),
            "pred_pixels": int(pred_support[idx]),
            "tp_pixels": int(tp[idx]),
            "fp_pixels": int(fp[idx]),
            "fn_pixels": int(fn[idx]),
            "union_pixels": int(union[idx]),
        }

    return {
        "oa_percent": 100.0 * float(oa),
        "official_miou_4class_percent": 100.0 * official_miou,
        "present_class_miou_percent": safe_float(100.0 * present_miou),
        "class_metrics": class_metrics,
    }


def harmonic_damage_f1(damage_cm: np.ndarray) -> float:
    metrics = metrics_from_confusion(damage_cm, CLASS_NAMES)
    f1 = np.asarray([
        metrics["class_metrics"][name]["f1_percent"] / 100.0
        for name in CLASS_NAMES[1:]
    ], dtype=np.float64)
    if np.any(f1 <= 0):
        return 0.0
    return float(len(f1) / np.sum(1.0 / f1))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataset_path: str,
    data_names: list[str],
    batch_size: int,
    device: torch.device,
    num_workers: int,
    split_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    print(f"--------- comprehensive {split_name} analysis ---------", flush=True)
    dataset = MultimodalDamageAssessmentDatset(
        dataset_path, data_names, 1024, None, "test"
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    cm_total = np.zeros((4, 4), dtype=np.int64)
    cm_loc = np.zeros((2, 2), dtype=np.int64)
    cm_damage_gt_building = np.zeros((4, 4), dtype=np.int64)
    event_cms: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    event_samples: Dict[str, int] = defaultdict(int)
    sample_rows: list[Dict[str, Any]] = []

    model.eval()
    for pre, post, labels_loc, labels_clf, data_idx in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        output_loc, output_clf = model(pre, post)

        pred_loc = output_loc.argmax(dim=1).cpu().numpy()
        pred_clf = output_clf.argmax(dim=1).cpu().numpy()
        gt_loc = labels_loc.numpy().astype(np.int64)
        gt_clf = labels_clf.numpy().astype(np.int64)

        cm_loc += confusion_from_arrays(gt_loc, pred_loc, 2)
        cm_total += confusion_from_arrays(gt_clf, pred_clf, 4)
        building_mask = gt_loc > 0
        cm_damage_gt_building += confusion_from_arrays(
            gt_clf[building_mask], pred_clf[building_mask], 4
        )

        names = list(data_idx)
        for i, sample_id in enumerate(names):
            sample_cm = confusion_from_arrays(gt_clf[i], pred_clf[i], 4)
            event = sample_to_event(str(sample_id))
            event_cms[event] += sample_cm
            event_samples[event] += 1
            sample_metrics = metrics_from_confusion(sample_cm, CLASS_NAMES)
            destroyed = sample_metrics["class_metrics"]["destroyed"]
            sample_rows.append({
                "split": split_name,
                "sample_id": str(sample_id),
                "event": event,
                "miou_4class_percent": sample_metrics["official_miou_4class_percent"],
                "present_class_miou_percent": sample_metrics["present_class_miou_percent"],
                "destroyed_iou_percent": destroyed["iou_percent"] if destroyed["union_pixels"] > 0 else None,
                "destroyed_precision_percent": destroyed["precision_percent"],
                "destroyed_recall_percent": destroyed["recall_percent"],
                "destroyed_f1_percent": destroyed["f1_percent"],
                "destroyed_gt_pixels": destroyed["gt_pixels"],
                "destroyed_pred_pixels": destroyed["pred_pixels"],
            })

    overall = metrics_from_confusion(cm_total, CLASS_NAMES)
    loc_metrics = metrics_from_confusion(cm_loc, ["background", "building"])
    clf_f1 = harmonic_damage_f1(cm_damage_gt_building)
    row_den = np.maximum(cm_total.sum(axis=1, keepdims=True), 1)
    row_normalized = cm_total / row_den * 100.0

    destroyed_row = {
        pred_name: float(row_normalized[3, pred_idx])
        for pred_idx, pred_name in enumerate(CLASS_NAMES)
    }

    event_rows: list[Dict[str, Any]] = []
    event_json: Dict[str, Any] = {}
    for event in sorted(event_cms):
        metrics = metrics_from_confusion(event_cms[event], CLASS_NAMES)
        destroyed = metrics["class_metrics"]["destroyed"]
        row = {
            "split": split_name,
            "event": event,
            "num_samples": event_samples[event],
            "miou_4class_percent": metrics["official_miou_4class_percent"],
            "present_class_miou_percent": metrics["present_class_miou_percent"],
            "background_iou_percent": metrics["class_metrics"]["background"]["iou_percent"],
            "intact_iou_percent": metrics["class_metrics"]["intact"]["iou_percent"],
            "damaged_iou_percent": metrics["class_metrics"]["damaged"]["iou_percent"],
            "destroyed_iou_percent": destroyed["iou_percent"] if destroyed["union_pixels"] > 0 else None,
            "destroyed_precision_percent": destroyed["precision_percent"],
            "destroyed_recall_percent": destroyed["recall_percent"],
            "destroyed_f1_percent": destroyed["f1_percent"],
            "destroyed_gt_pixels": destroyed["gt_pixels"],
            "destroyed_pred_pixels": destroyed["pred_pixels"],
            "destroyed_tp_pixels": destroyed["tp_pixels"],
            "destroyed_fp_pixels": destroyed["fp_pixels"],
            "destroyed_fn_pixels": destroyed["fn_pixels"],
        }
        event_rows.append(row)
        event_json[event] = row

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        split_dir / "confusion_matrix_counts.csv",
        cm_total,
        delimiter=",",
        fmt="%d",
        header=",".join(CLASS_NAMES),
        comments="",
    )
    np.savetxt(
        split_dir / "confusion_matrix_row_normalized_percent.csv",
        row_normalized,
        delimiter=",",
        fmt="%.6f",
        header=",".join(CLASS_NAMES),
        comments="",
    )

    class_rows = []
    for name in CLASS_NAMES:
        class_rows.append({"class": name, **overall["class_metrics"][name]})
    write_csv(
        split_dir / "class_metrics.csv",
        class_rows,
        ["class", "precision_percent", "recall_percent", "f1_percent", "iou_percent",
         "gt_pixels", "pred_pixels", "tp_pixels", "fp_pixels", "fn_pixels", "union_pixels"],
    )
    write_csv(
        split_dir / "per_event_metrics.csv",
        event_rows,
        list(event_rows[0].keys()) if event_rows else ["split", "event"],
    )
    write_csv(
        split_dir / "per_sample_metrics.csv",
        sample_rows,
        list(sample_rows[0].keys()) if sample_rows else ["split", "sample_id"],
    )

    result = {
        "split": split_name,
        "loc_f1_percent": loc_metrics["class_metrics"]["building"]["f1_percent"],
        "clf_harmonic_f1_percent": 100.0 * clf_f1,
        "oa_percent": overall["oa_percent"],
        "miou_percent": overall["official_miou_4class_percent"],
        "class_metrics": overall["class_metrics"],
        "confusion_matrix_counts": cm_total.tolist(),
        "confusion_matrix_row_normalized_percent": row_normalized.tolist(),
        "gt_destroyed_prediction_percent": destroyed_row,
        "per_event": event_json,
    }
    (split_dir / "overall_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"{split_name}: loc_F1={result['loc_f1_percent']:.4f}, "
        f"clf_F1={result['clf_harmonic_f1_percent']:.4f}, "
        f"OA={result['oa_percent']:.4f}, mIoU={result['miou_percent']:.4f}",
        flush=True,
    )
    print("GT destroyed prediction row (%):", destroyed_row, flush=True)
    print("Worst events by destroyed IoU:", flush=True)
    sortable = [r for r in event_rows if r["destroyed_iou_percent"] is not None]
    for row in sorted(sortable, key=lambda x: x["destroyed_iou_percent"])[:10]:
        print(
            f"  {row['event']}: destroyed IoU={row['destroyed_iou_percent']:.4f}, "
            f"recall={row['destroyed_recall_percent']:.4f}, gt_pixels={row['destroyed_gt_pixels']}",
            flush=True,
        )

    return result


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda:0")
    model = DamageFormer_S1C(
        guidance_beta=args.guidance_beta,
        use_pretrained_swin=False,
    )
    model.load_state_dict(load_state(str(checkpoint)), strict=True)
    model = model.to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"checkpoint": str(checkpoint)}

    if args.split in {"val", "both"}:
        summary["val"] = evaluate_split(
            model,
            args.val_dataset_path,
            read_list(args.val_data_list_path),
            args.eval_batch_size,
            device,
            args.num_workers,
            "val",
            out_dir,
        )
    if args.split in {"test", "both"}:
        summary["test"] = evaluate_split(
            model,
            args.test_dataset_path,
            read_list(args.test_data_list_path),
            args.eval_batch_size,
            device,
            args.num_workers,
            "test",
            out_dir,
        )

    if "val" in summary and "test" in summary:
        events = sorted(set(summary["val"]["per_event"]) | set(summary["test"]["per_event"]))
        rows = []
        for event in events:
            val = summary["val"]["per_event"].get(event, {})
            test = summary["test"]["per_event"].get(event, {})
            val_iou = val.get("destroyed_iou_percent")
            test_iou = test.get("destroyed_iou_percent")
            gap = None if val_iou is None or test_iou is None else float(test_iou) - float(val_iou)
            rows.append({
                "event": event,
                "val_destroyed_iou_percent": val_iou,
                "test_destroyed_iou_percent": test_iou,
                "test_minus_val_destroyed_iou_gap": gap,
                "val_destroyed_gt_pixels": val.get("destroyed_gt_pixels"),
                "test_destroyed_gt_pixels": test.get("destroyed_gt_pixels"),
            })
        write_csv(
            out_dir / "val_test_event_comparison.csv",
            rows,
            ["event", "val_destroyed_iou_percent", "test_destroyed_iou_percent",
             "test_minus_val_destroyed_iou_gap", "val_destroyed_gt_pixels", "test_destroyed_gt_pixels"],
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved comprehensive analysis to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
