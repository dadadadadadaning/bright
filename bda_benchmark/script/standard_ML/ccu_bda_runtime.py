# -*- coding: utf-8 -*-
"""Shared evaluation utilities for the CCU-BDA experimental line."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    fast_hist,
    metrics_from_hist,
)


DIAGNOSTIC_SUFFIXES = (
    "reliability",
    "correlation_entropy",
    "correlation_peak",
    "correlation_margin",
    "offset_magnitude",
    "cycle_error",
    "cycle_consistency",
    "change_probability",
    "unchanged_gate",
    "changed_gate",
    "rejected_gate",
    "conditional_transition",
    "hierarchy_transition",
    "controlled_transition",
    "controlled_residual_abs",
    "controlled_matched_difference_abs",
    "controlled_3d_difference_abs",
)


def _event_name(sample_name: object) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def _damage_metrics(histogram: torch.Tensor) -> Dict[str, float]:
    intersection = torch.diag(histogram)
    union = histogram.sum(1) + histogram.sum(0) - intersection
    iou = intersection / union.clamp_min(1.0)
    return {
        "miou4": float(iou.mean().item()),
        "valid_miou": float(iou[1:4].mean().item()),
        "background_iou": float(iou[0].item()),
        "intact_iou": float(iou[1].item()),
        "damaged_iou": float(iou[2].item()),
        "destroyed_iou": float(iou[3].item()),
    }


@torch.no_grad()
def evaluate_ccu(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
    details_sink: MutableMapping[str, object] | None = None,
) -> Dict[str, float]:
    """Evaluate official segmentation metrics plus CCU diagnostics.

    Diagnostic tensors are monitor-only.  They never affect model selection,
    which remains validation three-class mIoU for compatibility with the
    existing BRIGHT experiments.
    """

    was_training = model.training
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hist: Dict[str, torch.Tensor] = {}
    event_image_count: Dict[str, int] = defaultdict(int)
    event_diagnostic_sum: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    event_diagnostic_count: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    processed = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rgb, sar, location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar, return_dict=True)
        loc_prediction = outputs["output_loc"].argmax(1)
        damage_prediction = outputs["output_dam"].argmax(1)
        loc_hist += fast_hist(loc_prediction, location, 2, ignore_index)
        damage_hist += fast_hist(damage_prediction, damage, 4, ignore_index)

        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if any(key.endswith(suffix) for suffix in DIAGNOSTIC_SUFFIXES):
                finite = value.float()
                if torch.isfinite(finite).all():
                    sums[key] += float(finite.mean().item())
                    counts[key] += 1

        for sample_index in range(rgb.shape[0]):
            event = _event_name(names[sample_index])
            sample_hist = fast_hist(
                damage_prediction[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1],
                4,
                ignore_index,
            )
            event_hist[event] = event_hist.get(
                event, torch.zeros_like(sample_hist)
            ) + sample_hist
            event_image_count[event] += 1
            for key, value in outputs.items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0:
                    continue
                if not any(key.endswith(suffix) for suffix in DIAGNOSTIC_SUFFIXES):
                    continue
                sample_value = value
                if value.shape[0] == rgb.shape[0]:
                    sample_value = value[sample_index : sample_index + 1]
                finite = sample_value.float()
                if torch.isfinite(finite).all():
                    event_diagnostic_sum[event][key] += float(finite.mean().item())
                    event_diagnostic_count[event][key] += 1
        processed += int(rgb.shape[0])

    metrics = metrics_from_hist(loc_hist, damage_hist)
    location_tp = loc_hist[1, 1]
    metrics["loc_precision"] = float(
        (location_tp / loc_hist[:, 1].sum().clamp_min(1.0)).item()
    )
    metrics["loc_recall"] = float(
        (location_tp / loc_hist[1, :].sum().clamp_min(1.0)).item()
    )
    intact_total = damage_hist[1].sum().clamp_min(1.0)
    metrics["intact_to_damaged"] = float(
        (damage_hist[1, 2] / intact_total).item()
    )
    metrics["intact_to_destroyed"] = float(
        (damage_hist[1, 3] / intact_total).item()
    )
    metrics["processed_images"] = int(processed)

    event_values = []
    for histogram in event_hist.values():
        intersection = torch.diag(histogram)
        union = histogram.sum(1) + histogram.sum(0) - intersection
        event_values.append(
            float((intersection / union.clamp_min(1.0)).mean().item())
        )
    metrics["event_count"] = len(event_values)
    metrics["event_mean_official_style_4class_miou"] = (
        sum(event_values) / len(event_values) if event_values else 0.0
    )
    metrics["event_worst_official_style_4class_miou"] = (
        min(event_values) if event_values else 0.0
    )
    for key in sorted(sums):
        metrics[f"ccu_{key}"] = sums[key] / max(counts[key], 1)

    if details_sink is not None:
        event_records: Dict[str, Dict[str, object]] = {}
        disaster_groups: Dict[str, list[Dict[str, float]]] = defaultdict(list)
        for event in sorted(event_hist):
            event_metrics = _damage_metrics(event_hist[event])
            diagnostics = {
                f"ccu_{key}": event_diagnostic_sum[event][key]
                / max(event_diagnostic_count[event][key], 1)
                for key in sorted(event_diagnostic_sum[event])
            }
            disaster_type = event.rsplit("-", 1)[-1]
            event_records[event] = {
                "event": event,
                "disaster_type": disaster_type,
                "processed_images": event_image_count[event],
                **event_metrics,
                **diagnostics,
            }
            disaster_groups[disaster_type].append(event_metrics)
        disaster_records: Dict[str, Dict[str, float | int]] = {}
        for disaster_type, records in sorted(disaster_groups.items()):
            keys = tuple(records[0]) if records else ()
            disaster_records[disaster_type] = {
                "event_count": len(records),
                **{
                    key: sum(record[key] for record in records) / len(records)
                    for key in keys
                },
            }
        details_sink["event_metrics"] = event_records
        details_sink["disaster_type_macro_metrics"] = disaster_records

    if was_training:
        model.train()
    return metrics


def metrics_line(prefix: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{prefix} | valid_mIoU={100*metrics['valid_miou']:.4f} | "
        f"mIoU4={100*metrics['miou4']:.4f} | "
        f"Loc F1={100*metrics['loc_f1']:.2f} | "
        f"IoU(int/dmg/des)={100*metrics['intact_iou']:.2f}/"
        f"{100*metrics['damaged_iou']:.2f}/"
        f"{100*metrics['destroyed_iou']:.2f} | "
        f"Dmg->Int={100*metrics['damaged_to_intact']:.2f}% | "
        f"Des->Int={100*metrics['destroyed_to_intact']:.2f}% | "
        f"Event mean/worst="
        f"{100*metrics['event_mean_official_style_4class_miou']:.2f}/"
        f"{100*metrics['event_worst_official_style_4class_miou']:.2f}"
    )


__all__: Iterable[str] = ("DIAGNOSTIC_SUFFIXES", "evaluate_ccu", "metrics_line")
