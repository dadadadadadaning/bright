#!/usr/bin/env python3
"""Unified BRIGHT evaluator for the six standard-model checkpoints.

The script deliberately uses one dataset implementation, one label mapping and
one metric implementation for every model.  It saves raw predictions so that a
later reporting step can render exactly the same qualitative cases for all
models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
BDA_ROOT = REPO_ROOT / "bda_benchmark"
CHANGE_MAMBA_ROOT = REPO_ROOT / "ChangeMamba-master"
for search_path in (REPO_ROOT, BDA_ROOT, CHANGE_MAMBA_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader import MultimodalDamageAssessmentDatset


MODEL_NAMES = (
    "unet",
    "siamcrnn",
    "siamattnunet",
    "deeplabv3plus",
    "damageformer",
    "changemamba",
)
DISPLAY_NAMES = {
    "unet": "U-Net",
    "siamcrnn": "SiamCRNN",
    "siamattnunet": "SiamAttnUNet",
    "deeplabv3plus": "DeepLabV3+",
    "damageformer": "DamageFormer",
    "changemamba": "ChangeMamba",
}
CLASS_NAMES = ("Background", "Intact", "Damaged", "Destroyed")
COLOR_MAP = np.asarray(
    [
        (245, 246, 248),
        (70, 181, 121),
        (228, 189, 89),
        (182, 70, 69),
    ],
    dtype=np.uint8,
)
EVENTS = (
    "turkey-earthquake",
    "hawaii-wildfire",
    "morocco-earthquake",
    "haiti-earthquake",
    "la_palma-volcano",
    "congo-volcano",
    "beirut-explosion",
    "bata-explosion",
    "libya-flood",
    "noto-earthquake",
    "marshall-wildfire",
    "ukraine-conflict",
    "myanmar-hurricane",
    "mexico-hurricane",
)
DISASTER_TYPES = (
    "earthquake",
    "wildfire",
    "volcano",
    "explosion",
    "flood",
    "conflict",
    "hurricane",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--list-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--suffix", default=".tif")
    parser.add_argument("--crop-size", type=int, default=1024,
                        help="Kept for dataset compatibility; test images are not randomly cropped.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference.")
    parser.add_argument("--tta", action="store_true",
                        help="Average identity, horizontal-flip and vertical-flip logits for every model.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: evaluate the first N images.")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument(
        "--changemamba-config",
        type=Path,
        default=CHANGE_MAMBA_ROOT / "changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml",
    )
    return parser.parse_args()


def read_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def event_from_name(name: str) -> str:
    for event in EVENTS:
        if event in name:
            return event
    return "unknown"


def disaster_type_from_name(name: str) -> str:
    for disaster_type in DISASTER_TYPES:
        if disaster_type in name:
            return disaster_type
    return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint) -> OrderedDict:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")
    state = OrderedDict(checkpoint)
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = OrderedDict((key[len(prefix):], value) for key, value in state.items())
    return state


def _build_siamcrnn_without_download():
    """Construct SiamCRNN without downloading ImageNet weights.

    Its historical constructor hard-codes ``pretrained=True`` even though a
    complete task checkpoint is loaded immediately afterwards.
    """
    siam_module = importlib.import_module("model.SiamCRNN")

    original = siam_module.torchvision.models.resnet18

    def local_resnet18(*_args, **_kwargs):
        try:
            return original(weights=None)
        except TypeError:
            return original(pretrained=False)

    siam_module.torchvision.models.resnet18 = local_resnet18
    try:
        return siam_module.SiamCRNN()
    finally:
        siam_module.torchvision.models.resnet18 = original


def build_model(model_name: str, changemamba_config: Path):
    if model_name == "unet":
        from model.UNet import UNet

        return UNet(in_channels=6, num_classes=4)
    if model_name == "siamcrnn":
        return _build_siamcrnn_without_download()
    if model_name == "siamattnunet":
        from model.SiamAttnUNet import SiamAttnUNet

        return SiamAttnUNet(in_channels=3, num_classes=4)
    if model_name == "deeplabv3plus":
        # The constructor also requests ImageNet weights.  Prevent a network
        # download because the full BRIGHT checkpoint replaces them.
        deeplab_module = importlib.import_module("model.DeepLabV3Plus")

        original = deeplab_module.models.resnet50

        def local_resnet50(*_args, **_kwargs):
            try:
                return original(weights=None)
            except TypeError:
                return original(pretrained=False)

        deeplab_module.models.resnet50 = local_resnet50
        try:
            return deeplab_module.DeepLabV3Plus(in_channels=6, num_classes=4)
        finally:
            deeplab_module.models.resnet50 = original
    if model_name == "damageformer":
        from model.DamageFormer import DamageFormer

        return DamageFormer(use_pretrained_swin=False)
    if model_name == "changemamba":
        from changedetection.configs.config import get_config
        from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
        from changedetection.script.script_utils import get_vssm_kwargs

        config_args = SimpleNamespace(
            cfg=str(changemamba_config),
            opts=None,
            encoder_pretrained_path=None,
            model_checkpoint_path=None,
            resume=None,
            resume_training_path=None,
        )
        config = get_config(config_args)
        return ChangeMambaMMBDA(
            output_building=2,
            output_damage=4,
            pretrained=None,
            **get_vssm_kwargs(config),
        )
    raise ValueError(model_name)


def load_checkpoint(model, model_name: str, path: Path, allow_partial: bool) -> dict:
    if model_name == "changemamba":
        from changedetection.checkpoints import load_model_weights

        report = load_model_weights(model, str(path))
        total = len(model.state_dict())
        report = dict(report)
        report["total_model_keys"] = total
        report["loaded_fraction"] = report["loaded_keys"] / max(total, 1)
        if report["loaded_fraction"] < 0.98 and not allow_partial:
            raise RuntimeError(
                f"ChangeMamba checkpoint loaded only {report['loaded_keys']}/{total} keys. "
                "Check --changemamba-config; use --allow-partial-checkpoint only for diagnosis."
            )
        return report

    state = _extract_state_dict(_torch_load(path))
    incompatible = model.load_state_dict(state, strict=False)
    total = len(model.state_dict())
    loaded = total - len(incompatible.missing_keys)
    report = {
        "loaded_keys": loaded,
        "total_model_keys": total,
        "loaded_fraction": loaded / max(total, 1),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if (incompatible.missing_keys or incompatible.unexpected_keys) and not allow_partial:
        raise RuntimeError(
            "Checkpoint does not strictly match the selected model. "
            f"Missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}."
        )
    return report


def forward_logits(model, model_name: str, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    if model_name in {"unet", "deeplabv3plus"}:
        output = model(torch.cat((pre, post), dim=1))
    else:
        output = model(pre, post)
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[1] != 4:
        shape = getattr(output, "shape", None)
        raise RuntimeError(f"Expected [B,4,H,W] damage logits, got {type(output)!r} with shape {shape}.")
    return output


def infer_logits(model, model_name: str, pre: torch.Tensor, post: torch.Tensor, tta: bool) -> torch.Tensor:
    logits = [forward_logits(model, model_name, pre, post)]
    if tta:
        for dims in ((3,), (2,)):
            aug_pre = pre.flip(dims=dims)
            aug_post = post.flip(dims=dims)
            aug_logits = forward_logits(model, model_name, aug_pre, aug_post).flip(dims=dims)
            logits.append(aug_logits)
    return torch.stack(logits, dim=0).mean(dim=0)


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = num_classes * gt[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=output, where=np.asarray(denominator) != 0)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    union = gt_count + pred_count - tp
    iou = safe_divide(tp, union)
    precision = safe_divide(tp, pred_count)
    recall = safe_divide(tp, gt_count)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    gt_building = gt_count[1:].sum()
    pred_building = pred_count[1:].sum()
    building_tp = cm[1:, 1:].sum()
    building_union = gt_building + pred_building - building_tp
    building_precision = float(building_tp / pred_building) if pred_building else np.nan
    building_recall = float(building_tp / gt_building) if gt_building else np.nan
    building_f1 = (
        2 * building_precision * building_recall / (building_precision + building_recall)
        if building_precision + building_recall > 0
        else np.nan
    )
    total = cm.sum()
    return {
        "pixel_accuracy": float(tp.sum() / total) if total else np.nan,
        "mean_iou_4class": float(np.nanmean(iou)),
        "foreground_miou_3class": float(np.nanmean(iou[1:])),
        "building_iou": float(building_tp / building_union) if building_union else np.nan,
        "building_f1": float(building_f1),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": gt_count.astype(np.int64),
        "damaged_to_intact_rate": float(cm[2, 1] / gt_count[2]) if gt_count[2] else np.nan,
        "destroyed_to_intact_rate": float(cm[3, 1] / gt_count[3]) if gt_count[3] else np.nan,
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def flatten_metric_row(prefix: dict, metrics: dict) -> dict:
    row = dict(prefix)
    for key in ("pixel_accuracy", "mean_iou_4class", "foreground_miou_3class", "building_iou", "building_f1",
                "damaged_to_intact_rate", "destroyed_to_intact_rate"):
        row[key] = metrics[key]
    for index, class_name in enumerate(CLASS_NAMES):
        slug = class_name.lower()
        row[f"iou_{slug}"] = metrics["iou"][index]
        row[f"precision_{slug}"] = metrics["precision"][index]
        row[f"recall_{slug}"] = metrics["recall"][index]
        row[f"f1_{slug}"] = metrics["f1"][index]
        row[f"gt_pixels_{slug}"] = int(metrics["support"][index])
    return row


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(prediction: np.ndarray, raw_dir: Path, color_dir: Path, name: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(raw_dir / f"{name}.png")
    Image.fromarray(COLOR_MAP[prediction.astype(np.uint8)], mode="RGB").save(color_dir / f"{name}.png")


def validate_paths(args: argparse.Namespace) -> None:
    required = (args.checkpoint, args.dataset_root, args.list_file)
    missing = [str(path) for path in required if not path.exists()]
    if args.model == "changemamba" and not args.changemamba_config.exists():
        missing.append(str(args.changemamba_config))
    if missing:
        raise FileNotFoundError("Required paths not found:\n  " + "\n  ".join(missing))


def main() -> None:
    args = parse_args()
    validate_paths(args)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    names = read_names(args.list_file)
    if args.limit > 0:
        names = names[: args.limit]
    dataset = MultimodalDamageAssessmentDatset(
        str(args.dataset_root), names, args.crop_size, None, "test", suffix=args.suffix
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(f"[INFO] Building {DISPLAY_NAMES[args.model]} ...", flush=True)
    model = build_model(args.model, args.changemamba_config)
    checkpoint_report = load_checkpoint(model, args.model, args.checkpoint, args.allow_partial_checkpoint)
    model.to(device).eval()

    output_dir = args.output_root / args.model
    raw_dir = output_dir / "predictions/raw"
    color_dir = output_dir / "predictions/colored"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_cm = np.zeros((4, 4), dtype=np.int64)
    event_cms: Dict[str, np.ndarray] = {}
    type_cms: Dict[str, np.ndarray] = {}
    per_image_rows = []
    forward_seconds = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    completed = 0
    with torch.inference_mode():
        for pre, post, _loc_gt, damage_gt, batch_names in loader:
            pre = pre.to(device, non_blocking=True).float()
            post = post.to(device, non_blocking=True).float()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            amp_enabled = args.amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = infer_logits(model, args.model, pre, post, args.tta)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            forward_seconds.extend([elapsed / len(batch_names)] * len(batch_names))

            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            targets = damage_gt.cpu().numpy().astype(np.int64)
            for batch_index, name in enumerate(batch_names):
                prediction = predictions[batch_index]
                target = targets[batch_index]
                cm = confusion_matrix(target, prediction)
                overall_cm += cm
                event = event_from_name(name)
                disaster_type = disaster_type_from_name(name)
                event_cms.setdefault(event, np.zeros((4, 4), dtype=np.int64))
                event_cms[event] += cm
                type_cms.setdefault(disaster_type, np.zeros((4, 4), dtype=np.int64))
                type_cms[disaster_type] += cm
                image_metrics = metrics_from_confusion(cm)
                row = flatten_metric_row(
                    {"name": name, "event": event, "disaster_type": disaster_type}, image_metrics
                )
                row["damaged_to_intact_pixels"] = int(cm[2, 1])
                row["destroyed_to_intact_pixels"] = int(cm[3, 1])
                per_image_rows.append(row)
                if not args.no_save_predictions:
                    save_prediction(prediction, raw_dir, color_dir, name)

            completed += len(batch_names)
            if completed % 25 == 0 or completed == len(dataset):
                print(f"[INFO] {args.model}: {completed}/{len(dataset)} images", flush=True)

    overall_metrics = metrics_from_confusion(overall_cm)
    event_rows = []
    for event in sorted(event_cms):
        event_rows.append(flatten_metric_row({"event": event}, metrics_from_confusion(event_cms[event])))
    type_rows = []
    for disaster_type in sorted(type_cms):
        type_rows.append(
            flatten_metric_row({"disaster_type": disaster_type}, metrics_from_confusion(type_cms[disaster_type]))
        )

    per_class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        per_class_rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "iou": overall_metrics["iou"][class_index],
                "precision": overall_metrics["precision"][class_index],
                "recall": overall_metrics["recall"][class_index],
                "f1": overall_metrics["f1"][class_index],
                "support_pixels": int(overall_metrics["support"][class_index]),
            }
        )

    timing = {
        "mean_seconds_per_image": float(np.mean(forward_seconds)),
        "median_seconds_per_image": float(np.median(forward_seconds)),
        "p95_seconds_per_image": float(np.percentile(forward_seconds, 95)),
        "throughput_images_per_second": float(1.0 / np.mean(forward_seconds)),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
        ),
    }
    payload = {
        "schema_version": 1,
        "model": args.model,
        "display_name": DISPLAY_NAMES[args.model],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_load": checkpoint_report,
        "dataset_root": str(args.dataset_root.resolve()),
        "list_file": str(args.list_file.resolve()),
        "num_images": len(dataset),
        "class_names": CLASS_NAMES,
        "protocol": {
            "num_classes": 4,
            "tta": bool(args.tta),
            "amp": bool(args.amp),
            "batch_size": args.batch_size,
            "suffix": args.suffix,
            "prediction_masked_by_localization_head": False,
            "metric_aggregation": "global pixel confusion matrix",
        },
        "metrics": overall_metrics,
        "timing": timing,
        "confusion_matrix_gt_rows_pred_columns": overall_cm,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "per_event_metrics.csv", event_rows)
    write_csv(output_dir / "per_disaster_type_metrics.csv", type_rows)
    np.savetxt(output_dir / "confusion_matrix.csv", overall_cm, delimiter=",", fmt="%d")

    print(json.dumps(json_ready({"model": args.model, "metrics": overall_metrics, "timing": timing}), indent=2))
    print(f"[DONE] Results: {output_dir}")


if __name__ == "__main__":
    main()
