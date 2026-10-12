# -*- coding: utf-8 -*-
"""Loss and evaluation utilities for H10-H3D."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    fast_hist,
    metrics_from_hist,
)


def hierarchical_damage_nll(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    """One conditional likelihood objective for three building states.

    Background and ignored pixels do not supervise the two damage decisions;
    building localisation remains the frozen P3 responsibility.  The loss is
    not a collection of auxiliary objectives: it is exactly the negative log
    likelihood of ``intact / damaged / destroyed`` under the factorisation.
    """

    occurrence = outputs["h10_affected_logits"].float().squeeze(1)
    severity = outputs["h10_severity_logits"].float().squeeze(1)
    valid = (labels != ignore_index) & (labels >= 1) & (labels <= 3)
    if not valid.any():
        zero = occurrence.sum() * 0.0
        return {
            "extra_supervised_loss": zero,
            "h10_hierarchical_nll": zero,
            "h10_occurrence_nll": zero,
            "h10_conditional_severity_nll": zero,
            "h10_supervised_building_pixels": zero.detach(),
        }

    occurrence_target = ((labels == 2) | (labels == 3)).float()
    occurrence_map = F.binary_cross_entropy_with_logits(
        occurrence, occurrence_target, reduction="none"
    )
    occurrence_loss = occurrence_map[valid].mean()

    affected = valid & ((labels == 2) | (labels == 3))
    if affected.any():
        severity_target = (labels == 3).float()
        severity_map = F.binary_cross_entropy_with_logits(
            severity, severity_target, reduction="none"
        )
        severity_loss = severity_map[affected].mean()
    else:
        severity_loss = severity.sum() * 0.0

    # Per-pixel conditional NLL: intact contributes occurrence only; affected
    # pixels contribute occurrence plus conditional severity.
    total = occurrence_map[valid].sum()
    if affected.any():
        total = total + severity_map[affected].sum()
    total = total / valid.sum().clamp_min(1)
    return {
        "extra_supervised_loss": total,
        "h10_hierarchical_nll": total,
        "h10_occurrence_nll": occurrence_loss,
        "h10_conditional_severity_nll": severity_loss,
        "h10_supervised_building_pixels": valid.sum().detach().float(),
    }


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
def evaluate_h10(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
    details_sink: MutableMapping[str, object] | None = None,
) -> Dict[str, float]:
    """Evaluate final segmentation and the two failure locations separately."""

    was_training = model.training
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hist: Dict[str, torch.Tensor] = {}
    event_images: Dict[str, int] = defaultdict(int)
    counters: Dict[str, float] = defaultdict(float)
    diagnostic_sums: Dict[str, float] = defaultdict(float)
    diagnostic_counts: Dict[str, int] = defaultdict(int)
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

        occurrence_prediction = outputs["h10_affected_logits"][:, 0] >= 0
        severity_prediction = outputs["h10_severity_logits"][:, 0] >= 0
        valid = damage != ignore_index
        gt_affected = valid & ((damage == 2) | (damage == 3))
        gt_intact = valid & (damage == 1)
        gt_damaged = valid & (damage == 2)
        gt_destroyed = valid & (damage == 3)
        counters["affected_tp"] += float((gt_affected & occurrence_prediction).sum())
        counters["affected_fp"] += float((gt_intact & occurrence_prediction).sum())
        counters["affected_fn"] += float((gt_affected & ~occurrence_prediction).sum())
        counters["damaged_total"] += float(gt_damaged.sum())
        counters["damaged_gate_miss"] += float((gt_damaged & ~occurrence_prediction).sum())
        counters["damaged_severity_to_destroyed"] += float(
            (gt_damaged & occurrence_prediction & severity_prediction).sum()
        )
        counters["destroyed_total"] += float(gt_destroyed.sum())
        counters["destroyed_gate_miss"] += float(
            (gt_destroyed & ~occurrence_prediction).sum()
        )
        counters["destroyed_severity_to_damaged"] += float(
            (gt_destroyed & occurrence_prediction & ~severity_prediction).sum()
        )
        counters["severity_tp"] += float((gt_destroyed & severity_prediction).sum())
        counters["severity_fp"] += float((gt_damaged & severity_prediction).sum())
        counters["severity_fn"] += float((gt_destroyed & ~severity_prediction).sum())

        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor) or not key.startswith("h10_"):
                continue
            finite = value.detach().float()
            if torch.isfinite(finite).all():
                diagnostic_sums[key] += float(finite.mean().item())
                diagnostic_counts[key] += 1

        for sample_index in range(rgb.shape[0]):
            event = _event_name(names[sample_index])
            sample_hist = fast_hist(
                damage_prediction[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1],
                4,
                ignore_index,
            )
            event_hist[event] = event_hist.get(event, torch.zeros_like(sample_hist)) + sample_hist
            event_images[event] += 1
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
    metrics["intact_to_damaged"] = float((damage_hist[1, 2] / intact_total).item())
    metrics["intact_to_destroyed"] = float((damage_hist[1, 3] / intact_total).item())
    metrics["processed_images"] = int(processed)

    tp = counters["affected_tp"]
    fp = counters["affected_fp"]
    fn = counters["affected_fn"]
    metrics["affected_iou"] = tp / max(tp + fp + fn, 1.0)
    metrics["affected_f1"] = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    metrics["affected_recall"] = tp / max(tp + fn, 1.0)
    metrics["damaged_gate_miss"] = counters["damaged_gate_miss"] / max(
        counters["damaged_total"], 1.0
    )
    metrics["damaged_severity_to_destroyed"] = counters[
        "damaged_severity_to_destroyed"
    ] / max(counters["damaged_total"], 1.0)
    metrics["destroyed_gate_miss"] = counters["destroyed_gate_miss"] / max(
        counters["destroyed_total"], 1.0
    )
    metrics["destroyed_severity_to_damaged"] = counters[
        "destroyed_severity_to_damaged"
    ] / max(counters["destroyed_total"], 1.0)
    severity_tp = counters["severity_tp"]
    severity_fp = counters["severity_fp"]
    severity_fn = counters["severity_fn"]
    metrics["conditional_destroyed_f1"] = 2.0 * severity_tp / max(
        2.0 * severity_tp + severity_fp + severity_fn, 1.0
    )

    event_values = []
    event_records: Dict[str, Dict[str, object]] = {}
    disaster_groups: Dict[str, list[Dict[str, float]]] = defaultdict(list)
    for event in sorted(event_hist):
        event_metrics = _damage_metrics(event_hist[event])
        event_values.append(event_metrics["miou4"])
        disaster_type = event.rsplit("-", 1)[-1]
        event_records[event] = {
            "event": event,
            "disaster_type": disaster_type,
            "processed_images": event_images[event],
            **event_metrics,
        }
        disaster_groups[disaster_type].append(event_metrics)
    metrics["event_count"] = len(event_values)
    metrics["event_mean_official_style_4class_miou"] = (
        sum(event_values) / len(event_values) if event_values else 0.0
    )
    metrics["event_worst_official_style_4class_miou"] = (
        min(event_values) if event_values else 0.0
    )
    metrics["damage_balance_score"] = float(
        0.5 * (metrics["damaged_iou"] + metrics["destroyed_iou"])
        - 0.25 * (metrics["damaged_to_intact"] + metrics["destroyed_to_intact"])
    )
    for key in sorted(diagnostic_sums):
        metrics[key] = diagnostic_sums[key] / max(diagnostic_counts[key], 1)

    if details_sink is not None:
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
        f"mIoU4={100*metrics['miou4']:.4f} | Loc F1={100*metrics['loc_f1']:.2f} | "
        f"IoU(int/dmg/des)={100*metrics['intact_iou']:.2f}/"
        f"{100*metrics['damaged_iou']:.2f}/{100*metrics['destroyed_iou']:.2f} | "
        f"Affected F1/Recall={100*metrics['affected_f1']:.2f}/"
        f"{100*metrics['affected_recall']:.2f} | "
        f"Dmg gate-miss/to-Des={100*metrics['damaged_gate_miss']:.2f}/"
        f"{100*metrics['damaged_severity_to_destroyed']:.2f}% | "
        f"Event mean/worst={100*metrics['event_mean_official_style_4class_miou']:.2f}/"
        f"{100*metrics['event_worst_official_style_4class_miou']:.2f}"
    )


__all__ = (
    "hierarchical_damage_nll",
    "evaluate_h10",
    "metrics_line",
)
