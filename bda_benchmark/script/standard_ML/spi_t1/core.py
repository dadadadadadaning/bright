"""Pure operators and preregistered gates for the SPI T0/T1 experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import connected_components_4


VARIANTS = ("t0_original_objective", "t1_component_balanced")
SOURCE_VARIANT = "p3_locked_source"


@dataclass(frozen=True)
class SemanticComponent:
    component_id: int
    batch_index: int
    class_label: int
    mask: torch.Tensor
    area: int


def class_component_partition(
    labels: torch.Tensor,
    *,
    minimum_area: int = 32,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, list[SemanticComponent]]:
    """Build 1-based IDs for class-separated 4-connected foreground regions.

    IDs are unique across the batch.  Components are constructed independently
    for labels 1/2/3, so touching regions with different damage classes split;
    same-class touching buildings deliberately remain one pseudo component.
    """

    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,H,W] or [B,1,H,W]")
    if minimum_area < 1:
        raise ValueError("minimum_area must be positive")
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
    output = torch.zeros_like(labels_cpu, dtype=torch.long)
    records: list[SemanticComponent] = []
    next_id = 1
    for batch_index in range(int(labels_cpu.shape[0])):
        valid = labels_cpu[batch_index] != int(ignore_index)
        for class_label in (1, 2, 3):
            mask = valid & (labels_cpu[batch_index] == class_label)
            for component in connected_components_4(mask):
                if int(component.sum()) < minimum_area:
                    continue
                component_mask = torch.from_numpy(component.copy())
                area = int(component_mask.sum().item())
                output[batch_index][component_mask] = next_id
                records.append(
                    SemanticComponent(
                        component_id=next_id,
                        batch_index=batch_index,
                        class_label=class_label,
                        mask=component_mask,
                        area=area,
                    )
                )
                next_id += 1
    return output, records


def class_component_id_map(
    labels: torch.Tensor,
    *,
    minimum_area: int = 32,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, int]:
    output, records = class_component_partition(
        labels,
        minimum_area=minimum_area,
        ignore_index=ignore_index,
    )
    return output, len(records)


def component_balanced_nll(
    damage_logits: torch.Tensor,
    damage_labels: torch.Tensor,
    *,
    minimum_area: int = 32,
    ignore_index: int = 255,
    component_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float | int]]:
    """Mean of per-component mean NLLs, excluding background and ignore pixels.

    L_cb = (1/N_R) sum_j (1/|R_j|) sum_{x in R_j} -log p_{y_j}(x)

    The reduction is performed on the logits device.  Connected-component
    extraction is non-differentiable and uses semantic training labels only.
    """

    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ValueError("damage_logits must have shape [B,4,H,W]")
    if damage_labels.ndim == 4 and damage_labels.shape[1] == 1:
        damage_labels = damage_labels[:, 0]
    if damage_labels.ndim != 3:
        raise ValueError("damage_labels must have shape [B,H,W]")
    if tuple(damage_logits.shape[-2:]) != tuple(damage_labels.shape[-2:]):
        raise ValueError("logits and labels must share spatial dimensions")
    if component_ids is None:
        component_ids, component_count = class_component_id_map(
            damage_labels,
            minimum_area=minimum_area,
            ignore_index=ignore_index,
        )
    else:
        if tuple(component_ids.shape) != tuple(damage_labels.shape):
            raise ValueError("component_ids and damage_labels must share a shape")
        component_count = int(component_ids.max().item()) if component_ids.numel() else 0

    nll = F.cross_entropy(
        damage_logits.float(),
        damage_labels.long(),
        ignore_index=ignore_index,
        reduction="none",
    )
    ids = component_ids.to(device=damage_logits.device, dtype=torch.long)
    active = ids > 0
    zero = damage_logits.float().sum() * 0.0
    if component_count == 0 or not bool(active.any().item()):
        return zero, {
            "component_count": 0,
            "component_pixels": 0,
            "mean_component_area": 0.0,
        }

    flat_ids = ids[active]
    flat_nll = nll[active]
    sums = torch.zeros(component_count + 1, device=damage_logits.device, dtype=torch.float32)
    sums.index_add_(0, flat_ids, flat_nll)
    counts = torch.bincount(flat_ids, minlength=component_count + 1).to(torch.float32)
    per_component = sums[1:] / counts[1:].clamp_min(1.0)
    return per_component.mean(), {
        "component_count": component_count,
        "component_pixels": int(active.sum().item()),
        "mean_component_area": float(counts[1:].mean().item()),
    }


def paired_gain(candidate: Mapping[str, object], control: Mapping[str, object]) -> Dict[str, float]:
    candidate_sem = candidate["semantic"]
    control_sem = control["semantic"]
    candidate_build = candidate["building"]
    control_build = control["building"]
    assert isinstance(candidate_sem, Mapping) and isinstance(control_sem, Mapping)
    assert isinstance(candidate_build, Mapping) and isinstance(control_build, Mapping)
    return {
        "valid_miou_pp": 100.0 * (float(candidate_sem["valid_miou"]) - float(control_sem["valid_miou"])),
        "damaged_f1_pp": 100.0 * (float(candidate_sem["damaged_f1"]) - float(control_sem["damaged_f1"])),
        "intact_iou_pp": 100.0 * (float(candidate_sem["intact_iou"]) - float(control_sem["intact_iou"])),
        "building_macro_f1_pp": 100.0 * (
            float(candidate_build["macro_f1_present_gt"])
            - float(control_build["macro_f1_present_gt"])
        ),
    }


def build_t1_decision(
    reports: Mapping[str, Mapping[str, object]],
    event_rows: Sequence[Mapping[str, object]],
    *,
    minimum_valid_miou_gain_pp: float = 0.5,
    minimum_damaged_f1_gain_pp: float = 1.0,
    minimum_building_macro_f1_gain_pp: float = 1.0,
    minimum_nonnegative_events: int = 10,
    maximum_intact_iou_drop_pp: float = 0.5,
) -> Dict[str, object]:
    """Apply the frozen T1 gate to raw T1 versus raw paired T0 outputs."""

    control = reports[VARIANTS[0]]
    candidate = reports[VARIANTS[1]]
    gain = paired_gain(candidate, control)
    if SOURCE_VARIANT not in reports:
        raise KeyError("T1 decision requires the locked source P3 report")
    source_gain = paired_gain(candidate, reports[SOURCE_VARIANT])
    nonnegative_events = sum(
        float(row["t1_vs_t0_building_macro_f1_gain_pp"]) >= 0.0
        for row in event_rows
    )
    checks = {
        "raw_valid_miou_gain": gain["valid_miou_pp"] >= minimum_valid_miou_gain_pp,
        "raw_damaged_f1_gain": gain["damaged_f1_pp"] >= minimum_damaged_f1_gain_pp,
        "raw_building_macro_f1_gain": gain["building_macro_f1_pp"] >= minimum_building_macro_f1_gain_pp,
        "event_nonnegative_count": nonnegative_events >= minimum_nonnegative_events,
        "intact_does_not_collapse": gain["intact_iou_pp"] >= -maximum_intact_iou_drop_pp,
        "t1_not_worse_than_locked_source": (
            source_gain["valid_miou_pp"] >= 0.0
            and source_gain["damaged_f1_pp"] >= 0.0
            and source_gain["building_macro_f1_pp"] >= 0.0
            and source_gain["intact_iou_pp"] >= -maximum_intact_iou_drop_pp
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": (
            "PROCEED_TO_T2_PSEUDO_GEOMETRY_LEARNING"
            if passed
            else "STOP_AT_T1_COMPONENT_BALANCING_INSUFFICIENT"
        ),
        "checks": checks,
        "gain_pp": gain,
        "gain_vs_locked_source_pp": source_gain,
        "nonnegative_events": nonnegative_events,
        "thresholds": {
            "minimum_valid_miou_gain_pp": minimum_valid_miou_gain_pp,
            "minimum_damaged_f1_gain_pp": minimum_damaged_f1_gain_pp,
            "minimum_building_macro_f1_gain_pp": minimum_building_macro_f1_gain_pp,
            "minimum_nonnegative_events": minimum_nonnegative_events,
            "maximum_intact_iou_drop_pp": maximum_intact_iou_drop_pp,
        },
    }
