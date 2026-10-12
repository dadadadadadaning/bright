"""Training/evaluation primitives for the controlled Q2-B0/B1 chain."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_RGBSAR_Q2_BuildingResidual import RegionRecord
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    fast_hist,
    metrics_from_hist,
)


def region_targets(
    records: Sequence[RegionRecord],
    instance_map: torch.Tensor,
    damage: torch.Tensor,
    *,
    minimum_building_fraction: float = 0.50,
    minimum_class_purity: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected region indices and 0-based building damage labels."""
    selected = []
    labels = []
    for index, record in enumerate(records):
        mask = instance_map[record.batch_index] == record.instance_id
        values = damage[record.batch_index, mask]
        building = values[(values >= 1) & (values <= 3)]
        if building.numel() / max(int(values.numel()), 1) < minimum_building_fraction:
            continue
        counts = torch.stack([(building == value).sum() for value in (1, 2, 3)])
        label = int(counts.argmax())
        if float(counts[label] / max(int(building.numel()), 1)) < minimum_class_purity:
            continue
        selected.append(index)
        labels.append(label + 1)
    device = damage.device
    return (
        torch.tensor(selected, device=device, dtype=torch.long),
        torch.tensor(labels, device=device, dtype=torch.long),
    )


def region_cross_entropy(
    outputs: Mapping[str, object],
    damage: torch.Tensor,
    *,
    minimum_building_fraction: float,
    minimum_class_purity: float,
) -> tuple[torch.Tensor, int]:
    indices, targets = region_targets(
        outputs["q2_records"],
        outputs["q2_instance_map"],
        damage,
        minimum_building_fraction=minimum_building_fraction,
        minimum_class_purity=minimum_class_purity,
    )
    logits = outputs["q2_region_logits"]
    if indices.numel() == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[indices], targets), int(indices.numel())


def _event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def _add_hierarchical_metrics(metrics: Dict[str, float], damage_hist: torch.Tensor) -> None:
    affected_total = damage_hist[2:4].sum().clamp_min(1.0)
    affected_tp = damage_hist[2:4, 2:4].sum()
    affected_fp = damage_hist[0:2, 2:4].sum()
    affected_fn = damage_hist[2:4, 0:2].sum()
    metrics["affected_recall"] = float((affected_tp / affected_total).item())
    metrics["affected_iou"] = float(
        (affected_tp / (affected_tp + affected_fp + affected_fn).clamp_min(1.0)).item()
    )
    metrics["affected_f1"] = float(
        (2 * affected_tp / (2 * affected_tp + affected_fp + affected_fn).clamp_min(1.0)).item()
    )
    damaged_total = damage_hist[2].sum().clamp_min(1.0)
    destroyed_total = damage_hist[3].sum().clamp_min(1.0)
    metrics["damaged_gate_miss"] = float(
        (damage_hist[2, 0:2].sum() / damaged_total).item()
    )
    metrics["destroyed_gate_miss"] = float(
        (damage_hist[3, 0:2].sum() / destroyed_total).item()
    )


def _finalize_metrics(
    loc_hist: torch.Tensor,
    damage_hist: torch.Tensor,
    event_hist: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    metrics = metrics_from_hist(loc_hist, damage_hist)
    _add_hierarchical_metrics(metrics, damage_hist)
    event_values = {}
    for event, histogram in event_hist.items():
        intersection = torch.diag(histogram)
        union = histogram.sum(1) + histogram.sum(0) - intersection
        event_values[event] = float(
            (intersection / union.clamp_min(1.0)).mean().item()
        )
    values = list(event_values.values())
    metrics["event_count"] = len(values)
    metrics["event_mean_official_style_4class_miou"] = float(np.mean(values)) if values else 0.0
    metrics["event_worst_official_style_4class_miou"] = float(np.min(values)) if values else 0.0
    metrics["event_std_official_style_4class_miou"] = float(np.std(values)) if values else 0.0
    metrics["event_miou4"] = event_values
    return metrics


@torch.inference_mode()
def evaluate_q2(model, loader, device, ignore_index: int = 255, max_batches: int = 0):
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    final_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    base_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    final_events: Dict[str, torch.Tensor] = {}
    base_events: Dict[str, torch.Tensor] = {}
    pixel_counts = defaultdict(int)
    region_counts = defaultdict(int)
    transition_sum = delta_sum = 0.0
    batch_count = processed = 0
    for batch_index, (rgb, sar, location, damage, names) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar, return_dict=True)
        loc_prediction = outputs["output_loc"].argmax(1)
        final_prediction = outputs["output_dam"].argmax(1)
        base_prediction = outputs["p3_output_dam"].argmax(1)
        loc_hist += fast_hist(loc_prediction, location, 2, ignore_index)
        final_hist += fast_hist(final_prediction, damage, 4, ignore_index)
        base_hist += fast_hist(base_prediction, damage, 4, ignore_index)
        valid = damage != ignore_index
        base_correct = (base_prediction == damage) & valid
        final_correct = (final_prediction == damage) & valid
        changed = (final_prediction != base_prediction) & valid
        pixel_counts["changed"] += int(changed.sum())
        pixel_counts["corrected"] += int((~base_correct & final_correct).sum())
        pixel_counts["harmed"] += int((base_correct & ~final_correct).sum())
        pixel_counts["valid"] += int(valid.sum())
        for class_index, class_name in ((1, "intact"), (2, "damaged"), (3, "destroyed")):
            class_mask = damage == class_index
            pixel_counts[f"{class_name}_corrected"] += int((~base_correct & final_correct & class_mask).sum())
            pixel_counts[f"{class_name}_harmed"] += int((base_correct & ~final_correct & class_mask).sum())

        indices, targets = region_targets(
            outputs["q2_records"], outputs["q2_instance_map"], damage,
            minimum_building_fraction=0.50, minimum_class_purity=0.75,
        )
        if indices.numel():
            base_region = outputs["q2_base_region_logits"][indices].argmax(1)
            final_region = outputs["q2_region_logits"][indices].argmax(1)
            base_ok = base_region == targets
            final_ok = final_region == targets
            region_counts["eligible"] += int(indices.numel())
            region_counts["changed"] += int((base_region != final_region).sum())
            region_counts["corrected"] += int((~base_ok & final_ok).sum())
            region_counts["harmed"] += int((base_ok & ~final_ok).sum())
            for target_index, class_name in enumerate(
                ("intact", "damaged", "destroyed"), start=1
            ):
                mask = targets == target_index
                region_counts[f"{class_name}_corrected"] += int((~base_ok & final_ok & mask).sum())
                region_counts[f"{class_name}_harmed"] += int((base_ok & ~final_ok & mask).sum())

        transition_sum += float(outputs["q2_transition"].mean())
        delta_sum += float(outputs["q2_delta_abs"])
        batch_count += 1
        for sample_index, name in enumerate(names):
            event = _event_name(name)
            final_sample = fast_hist(
                final_prediction[sample_index:sample_index + 1],
                damage[sample_index:sample_index + 1], 4, ignore_index,
            )
            base_sample = fast_hist(
                base_prediction[sample_index:sample_index + 1],
                damage[sample_index:sample_index + 1], 4, ignore_index,
            )
            final_events[event] = final_events.get(event, torch.zeros_like(final_sample)) + final_sample
            base_events[event] = base_events.get(event, torch.zeros_like(base_sample)) + base_sample
        processed += len(names)
        if processed % 50 == 0:
            print(f"Q2 validation: {processed}", flush=True)

    final_metrics = _finalize_metrics(loc_hist, final_hist, final_events)
    base_metrics = _finalize_metrics(loc_hist, base_hist, base_events)
    final_metrics["processed_images"] = processed
    base_metrics["processed_images"] = processed
    event_delta = {
        event: 100.0 * (
            final_metrics["event_miou4"][event] - base_metrics["event_miou4"][event]
        ) for event in sorted(set(final_events) & set(base_events))
    }
    audit = {
        "pixels": dict(pixel_counts),
        "regions": dict(region_counts),
        "pixel_net_correction": pixel_counts["corrected"] - pixel_counts["harmed"],
        "region_net_correction": region_counts["corrected"] - region_counts["harmed"],
        "transition": transition_sum / max(batch_count, 1),
        "residual_abs": delta_sum / max(batch_count, 1),
        "event_miou4_delta_pp": event_delta,
        "nonnegative_events": sum(value >= 0 for value in event_delta.values()),
        "comparable_events": len(event_delta),
    }
    return final_metrics, base_metrics, audit


def advancement_gate(final: Mapping[str, object], base: Mapping[str, object], audit: Mapping[str, object]):
    delta = {
        "valid_miou": 100.0 * (float(final["valid_miou"]) - float(base["valid_miou"])),
        "miou4": 100.0 * (float(final["miou4"]) - float(base["miou4"])),
        "damaged_iou": 100.0 * (float(final["damaged_iou"]) - float(base["damaged_iou"])),
        "event_mean": 100.0 * (float(final["event_mean_official_style_4class_miou"]) - float(base["event_mean_official_style_4class_miou"])),
        "event_worst": 100.0 * (float(final["event_worst_official_style_4class_miou"]) - float(base["event_worst_official_style_4class_miou"])),
        "damaged_to_intact": 100.0 * (float(final["damaged_to_intact"]) - float(base["damaged_to_intact"])),
    }
    checks = {
        "valid_miou": delta["valid_miou"] >= 0.30,
        "damaged_iou": delta["damaged_iou"] >= 0.50,
        "event_mean_nondecreasing": delta["event_mean"] >= 0.0,
        "event_worst_nondecreasing": delta["event_worst"] >= 0.0,
        "nonnegative_events": int(audit["nonnegative_events"]) >= 10,
        "region_net_positive": int(audit["region_net_correction"]) > 0,
        "damaged_to_intact_decreasing": delta["damaged_to_intact"] < 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_pp": delta,
        "thresholds": {
            "valid_miou_pp": 0.30,
            "damaged_iou_pp": 0.50,
            "minimum_nonnegative_events": 10,
        },
        "next_step": "PROCEED_TO_Q2_B2_MULTISCALE" if all(checks.values()) else "STOP_Q2_B1_OR_KEEP_P3",
    }


def selection_key(metrics: Mapping[str, object]) -> tuple[float, float, float, float]:
    return (
        float(metrics["event_worst_official_style_4class_miou"]),
        float(metrics["event_mean_official_style_4class_miou"]),
        float(metrics["damaged_iou"]),
        float(metrics["valid_miou"]),
    )


__all__ = [
    "advancement_gate", "evaluate_q2", "region_cross_entropy",
    "region_targets", "selection_key",
]
