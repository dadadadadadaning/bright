# -*- coding: utf-8 -*-
"""Losses and validation gate for isolated P3-HR experiments."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from model.DamageFormer_S8_MICB_DSMIL import connected_components_4
from script.standard_ML.p3_direct_occurrence_runtime import o1_validation_checks


def _squeeze_logit(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W]")
    return value.float()


def _hard_indices(
    base_values: torch.Tensor,
    count: int,
    *,
    largest: bool,
) -> torch.Tensor:
    return torch.topk(
        base_values.detach(),
        k=min(int(count), int(base_values.numel())),
        largest=largest,
        sorted=False,
    ).indices


def hierarchical_residual_loss(
    output_logits: torch.Tensor,
    base_logits: torch.Tensor,
    new_occurrence: torch.Tensor,
    base_occurrence: torch.Tensor,
    new_severity: torch.Tensor,
    base_severity: torch.Tensor,
    occurrence_delta: torch.Tensor,
    severity_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    hard_fraction: float = 0.25,
    minimum_hard_pixels: int = 64,
    required_margin: float = 0.0,
    four_class_weight: float = 0.25,
    occurrence_weight: float = 1.0,
    severity_weight: float = 0.5,
    protection_weight: float = 0.10,
    ignore_index: int = 255,
) -> tuple[torch.Tensor | None, Dict[str, float | int | bool]]:
    """Task-asymmetric hard Occurrence plus conditional Severity loss.

    Frozen-P3 margins define the hard samples.  An update is allowed only when
    both weak Damaged and risky Intact pixels are present.  Destroyed pixels do
    not dominate the Occurrence positive set; they are learned by the separate
    conditional Severity objective.
    """

    new_occurrence = _squeeze_logit(new_occurrence, "new_occurrence")
    base_occurrence = _squeeze_logit(base_occurrence, "base_occurrence")
    new_severity = _squeeze_logit(new_severity, "new_severity")
    base_severity = _squeeze_logit(base_severity, "base_severity")
    occurrence_delta = _squeeze_logit(occurrence_delta, "occurrence_delta")
    severity_delta = _squeeze_logit(severity_delta, "severity_delta")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must be [B,H,W] or [B,1,H,W]")
    expected = target.shape
    values = (
        new_occurrence,
        base_occurrence,
        new_severity,
        base_severity,
        occurrence_delta,
        severity_delta,
    )
    if any(value.shape != expected for value in values):
        raise ValueError("hierarchical logits and target must share [B,H,W]")
    if output_logits.shape != base_logits.shape or output_logits.shape[1] != 4:
        raise ValueError("output/base logits must share [B,4,H,W]")
    if not 0.0 < float(hard_fraction) <= 1.0:
        raise ValueError("hard_fraction must be in (0,1]")
    if int(minimum_hard_pixels) <= 0:
        raise ValueError("minimum_hard_pixels must be positive")

    valid = target != int(ignore_index)
    intact_mask = valid & (target == 1)
    damaged_mask = valid & (target == 2)
    destroyed_mask = valid & (target == 3)
    intact_count = int(intact_mask.sum().item())
    damaged_count = int(damaged_mask.sum().item())
    destroyed_count = int(destroyed_mask.sum().item())
    if intact_count == 0 or damaged_count == 0:
        return None, {
            "usable": False,
            "intact_pixels": intact_count,
            "damaged_pixels": damaged_count,
            "destroyed_pixels": destroyed_count,
            "selected_per_occurrence_class": 0,
        }

    desired = max(
        int(minimum_hard_pixels),
        int(round(float(hard_fraction) * damaged_count)),
    )
    selected = min(desired, intact_count, damaged_count)
    base_i = base_occurrence[intact_mask]
    base_d = base_occurrence[damaged_mask]
    new_i = new_occurrence[intact_mask]
    new_d = new_occurrence[damaged_mask]
    i_indices = _hard_indices(base_i, selected, largest=True)
    d_indices = _hard_indices(base_d, selected, largest=False)
    hard_i = new_i[i_indices]
    hard_d = new_d[d_indices]
    margin = float(required_margin)
    occurrence_i_loss = F.softplus(margin + hard_i).mean()
    occurrence_d_loss = F.softplus(margin - hard_d).mean()
    occurrence_loss = 0.5 * (occurrence_i_loss + occurrence_d_loss)

    severity_terms = []
    severity_d_loss = new_severity.sum() * 0.0
    severity_x_loss = new_severity.sum() * 0.0
    if damaged_count:
        severity_d_loss = F.softplus(new_severity[damaged_mask]).mean()
        severity_terms.append(severity_d_loss)
    if destroyed_count:
        severity_x_loss = F.softplus(-new_severity[destroyed_mask]).mean()
        severity_terms.append(severity_x_loss)
    severity_loss = torch.stack(severity_terms).mean()

    building_mask = valid & (target >= 1) & (target <= 3)
    building_logits = output_logits.permute(0, 2, 3, 1)[building_mask]
    building_target = target[building_mask]
    four_class_loss = F.cross_entropy(building_logits.float(), building_target.long())

    protect_values = []
    if intact_count > selected:
        easy_i = torch.ones_like(base_i, dtype=torch.bool)
        easy_i[i_indices] = False
        protect_values.append(occurrence_delta[intact_mask][easy_i])
    # Preserve P3's already-correct conditional D/X decisions.  The new head
    # is free to move the mistakes but may not rotate the complete deep space.
    base_prediction = base_logits.detach().argmax(1)
    easy_severity = (
        ((target == 2) & (base_prediction == 2))
        | ((target == 3) & (base_prediction == 3))
    ) & valid
    if bool(easy_severity.any()):
        protect_values.append(severity_delta[easy_severity])
    if protect_values:
        protection_loss = torch.cat(protect_values).pow(2).mean()
    else:
        protection_loss = occurrence_loss * 0.0

    loss = (
        float(four_class_weight) * four_class_loss
        + float(occurrence_weight) * occurrence_loss
        + float(severity_weight) * severity_loss
        + float(protection_weight) * protection_loss
    )
    diagnostics: Dict[str, float | int | bool] = {
        "usable": True,
        "intact_pixels": intact_count,
        "damaged_pixels": damaged_count,
        "destroyed_pixels": destroyed_count,
        "selected_per_occurrence_class": int(selected),
        "base_hard_intact_margin": float(base_i[i_indices].mean()),
        "base_hard_damaged_margin": float(base_d[d_indices].mean()),
        "new_hard_intact_margin": float(hard_i.detach().mean()),
        "new_hard_damaged_margin": float(hard_d.detach().mean()),
        "four_class_loss": float(four_class_loss.detach()),
        "occurrence_loss": float(occurrence_loss.detach()),
        "severity_loss": float(severity_loss.detach()),
        "severity_damaged_loss": float(severity_d_loss.detach()),
        "severity_destroyed_loss": float(severity_x_loss.detach()),
        "protection_loss": float(protection_loss.detach()),
    }
    return loss, diagnostics


def hr3o_region_occurrence_loss(
    base_logits: torch.Tensor,
    new_occurrence: torch.Tensor,
    base_occurrence: torch.Tensor,
    occurrence_delta: torch.Tensor,
    severity_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    repairable_min_margin: float = -2.0,
    repairable_max_margin: float = 0.0,
    required_margin: float = 0.0,
    minimum_component_area: int = 16,
    occurrence_weight: float = 1.0,
    protection_weight: float = 1.0,
    ignore_index: int = 255,
) -> tuple[torch.Tensor | None, Dict[str, float | int | bool]]:
    """Region-level, Occurrence-only objective for the isolated HR3-O screen.

    Frozen P3 defines three disjoint permissions:

    * repair: GT-Damaged components whose mean Occurrence margin is below the
      boundary but still reachable by the bounded residual;
    * hard Intact: an equal number of Intact components with the largest
      frozen-P3 Occurrence margins;
    * keep: all other region-level P3-correct I/D/X components.  Their
      Occurrence residual is explicitly kept at zero.

    Components, rather than individual boundary pixels, are the optimisation
    unit.  Severity must be exactly zero: HR3-O cannot trade Damaged against
    Destroyed while screening the Intact--Affected boundary.
    """

    new_occurrence = _squeeze_logit(new_occurrence, "new_occurrence")
    base_occurrence = _squeeze_logit(base_occurrence, "base_occurrence")
    occurrence_delta = _squeeze_logit(occurrence_delta, "occurrence_delta")
    severity_delta = _squeeze_logit(severity_delta, "severity_delta")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must be [B,H,W] or [B,1,H,W]")
    if base_logits.ndim != 4 or base_logits.shape[1] != 4:
        raise ValueError("base_logits must be [B,4,H,W]")
    expected = target.shape
    if base_logits.shape[0] != expected[0] or base_logits.shape[-2:] != expected[-2:]:
        raise ValueError("base_logits and target must share batch/spatial shape")
    values = (new_occurrence, base_occurrence, occurrence_delta, severity_delta)
    if any(value.shape != expected for value in values):
        raise ValueError("HR3-O logits and target must share [B,H,W]")
    if int(minimum_component_area) <= 0:
        raise ValueError("minimum_component_area must be positive")
    lower = float(repairable_min_margin)
    upper = float(repairable_max_margin)
    if not lower < upper <= float(required_margin):
        raise ValueError(
            "repairable margins require min < max <= required_margin"
        )
    if bool(torch.count_nonzero(severity_delta.detach()).item()):
        raise RuntimeError("HR3-O severity residual must remain exactly zero")

    target_numpy = target.detach().cpu().numpy()
    records: list[dict[str, object]] = []
    class_region_counts = {1: 0, 2: 0, 3: 0}
    for batch_index in range(int(target.shape[0])):
        sample = np.asarray(target_numpy[batch_index])
        for class_id in (1, 2, 3):
            class_mask = sample == class_id
            for component in connected_components_4(class_mask):
                area = int(component.sum())
                if area < int(minimum_component_area):
                    continue
                mask = torch.as_tensor(
                    component,
                    device=target.device,
                    dtype=torch.bool,
                )
                base_margin = base_occurrence[batch_index][mask].mean()
                new_margin = new_occurrence[batch_index][mask].mean()
                pooled_logits = base_logits[batch_index, :, mask].float().mean(dim=1)
                records.append(
                    {
                        "batch_index": batch_index,
                        "class_id": class_id,
                        "mask": mask,
                        "area": area,
                        "base_margin": base_margin,
                        "new_margin": new_margin,
                        "base_prediction": int(pooled_logits.argmax().item()),
                    }
                )
                class_region_counts[class_id] += 1

    damaged_records = [record for record in records if record["class_id"] == 2]
    repairable = [
        record
        for record in damaged_records
        if lower <= float(record["base_margin"].detach()) < upper
    ]
    intact_records = [record for record in records if record["class_id"] == 1]
    # Prefer the reachable D->I buildings closest to the boundary.  More
    # negative components would consume gradient without being repairable by
    # the declared bounded residual.
    repairable.sort(
        key=lambda record: float(record["base_margin"].detach()), reverse=True
    )
    intact_records.sort(
        key=lambda record: float(record["base_margin"].detach()), reverse=True
    )
    selected_count = min(len(repairable), len(intact_records))
    if selected_count == 0:
        return None, {
            "usable": False,
            "intact_regions": class_region_counts[1],
            "damaged_regions": class_region_counts[2],
            "destroyed_regions": class_region_counts[3],
            "repairable_damaged_regions": len(repairable),
            "selected_repair_regions": 0,
            "selected_hard_intact_regions": 0,
            "severity_delta_exact_zero": True,
        }

    selected_repair = repairable[:selected_count]
    selected_intact = intact_records[:selected_count]
    selected_ids = {id(record) for record in selected_repair + selected_intact}
    margin = float(required_margin)
    repair_loss = torch.stack(
        [F.softplus(margin - record["new_margin"]) for record in selected_repair]
    ).mean()
    hard_intact_loss = torch.stack(
        [F.softplus(margin + record["new_margin"]) for record in selected_intact]
    ).mean()
    occurrence_loss = 0.5 * (repair_loss + hard_intact_loss)

    protected_records = {
        1: [],
        2: [],
        3: [],
    }
    protection_terms = []
    for record in records:
        class_id = int(record["class_id"])
        if id(record) in selected_ids:
            continue
        if int(record["base_prediction"]) != class_id:
            continue
        batch_index = int(record["batch_index"])
        mask = record["mask"]
        protection_terms.append(
            occurrence_delta[batch_index][mask].float().pow(2).mean()
        )
        protected_records[class_id].append(record)
    if protection_terms:
        protection_loss = torch.stack(protection_terms).mean()
    else:
        protection_loss = occurrence_loss * 0.0

    loss = (
        float(occurrence_weight) * occurrence_loss
        + float(protection_weight) * protection_loss
    )

    def mean_margin(items: list[dict[str, object]], key: str) -> float:
        return float(
            torch.stack([record[key].detach() for record in items]).mean()
        )

    impossible = sum(
        float(record["base_margin"].detach()) < lower
        for record in damaged_records
    )
    diagnostics: Dict[str, float | int | bool] = {
        "usable": True,
        "intact_regions": class_region_counts[1],
        "damaged_regions": class_region_counts[2],
        "destroyed_regions": class_region_counts[3],
        "repairable_damaged_regions": len(repairable),
        "unreachable_damaged_regions": int(impossible),
        "selected_repair_regions": selected_count,
        "selected_hard_intact_regions": selected_count,
        "protected_intact_regions": len(protected_records[1]),
        "protected_damaged_regions": len(protected_records[2]),
        "protected_destroyed_regions": len(protected_records[3]),
        "base_repair_margin": mean_margin(selected_repair, "base_margin"),
        "new_repair_margin": mean_margin(selected_repair, "new_margin"),
        "base_hard_intact_margin": mean_margin(selected_intact, "base_margin"),
        "new_hard_intact_margin": mean_margin(selected_intact, "new_margin"),
        "repair_loss": float(repair_loss.detach()),
        "hard_intact_loss": float(hard_intact_loss.detach()),
        "occurrence_loss": float(occurrence_loss.detach()),
        "protection_loss": float(protection_loss.detach()),
        "severity_loss": 0.0,
        "severity_delta_exact_zero": True,
    }
    return loss, diagnostics


def hr_validation_checks(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    minimum_damaged_gain_pp: float = 0.30,
    minimum_gate_miss_gain_pp: float = 0.50,
    minimum_affected_recall_gain_pp: float = 0.50,
    maximum_guard_drop_pp: float = 0.20,
    minimum_nonnegative_events: int = 10,
) -> tuple[Dict[str, bool], Dict[str, object]]:
    checks, diagnostics = o1_validation_checks(
        candidate,
        baseline,
        minimum_damaged_gain_pp=minimum_damaged_gain_pp,
        maximum_guard_drop_pp=maximum_guard_drop_pp,
        minimum_nonnegative_events=minimum_nonnegative_events,
    )
    gate_miss_gain = -float(diagnostics["damaged_gate_miss_delta_pp"])
    affected_recall_gain = float(diagnostics["affected_recall_delta_pp"])
    checks["damaged_gate_miss_improves_by_declared"] = (
        gate_miss_gain >= float(minimum_gate_miss_gain_pp)
    )
    checks["affected_recall_improves_by_declared"] = (
        affected_recall_gain >= float(minimum_affected_recall_gain_pp)
    )
    checks["event_mean_nondecreasing"] = float(
        candidate["event_mean_official_style_4class_miou"]
    ) >= float(baseline["event_mean_official_style_4class_miou"])
    diagnostics["damaged_gate_miss_gain_pp"] = gate_miss_gain
    diagnostics["affected_recall_gain_pp"] = affected_recall_gain
    return checks, diagnostics


def severity_validation_checks(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    maximum_guard_drop_pp: float = 0.20,
    minimum_nonnegative_events: int = 10,
) -> tuple[Dict[str, bool], Dict[str, object]]:
    """Admission gate for HR1-S, which cannot alter Occurrence by design."""

    event_base = baseline["event_miou_by_name"]
    event_candidate = candidate["event_miou_by_name"]
    common = sorted(set(event_base) & set(event_candidate))
    event_delta_pp = {
        event: 100.0 * (
            float(event_candidate[event]) - float(event_base[event])
        )
        for event in common
    }
    nonnegative = sum(value >= -1e-12 for value in event_delta_pp.values())
    severity_mean_base = 0.5 * (
        float(baseline["damaged_iou"]) + float(baseline["destroyed_iou"])
    )
    severity_mean_candidate = 0.5 * (
        float(candidate["damaged_iou"]) + float(candidate["destroyed_iou"])
    )
    severity_gain_pp = 100.0 * (
        severity_mean_candidate - severity_mean_base
    )
    intact_drop_pp = 100.0 * (
        float(baseline["intact_iou"]) - float(candidate["intact_iou"])
    )
    affected_recall_drop_pp = 100.0 * (
        float(baseline["affected_recall"])
        - float(candidate["affected_recall"])
    )
    diagnostics: Dict[str, object] = {
        # The trainer uses this common key only to rank admitted checkpoints.
        "damaged_gain_pp": severity_gain_pp,
        "severity_mean_iou_gain_pp": severity_gain_pp,
        "intact_drop_pp": intact_drop_pp,
        "affected_recall_drop_pp": affected_recall_drop_pp,
        "nonnegative_events": nonnegative,
        "comparable_events": len(common),
        "event_delta_pp": event_delta_pp,
    }
    checks = {
        "all_14_events_present": len(common) == 14,
        "valid_miou_nondecreasing": float(candidate["valid_miou"])
        >= float(baseline["valid_miou"]),
        "severity_mean_iou_nondecreasing": severity_gain_pp >= 0.0,
        "damage_macro_f1_nondecreasing": float(candidate["damage_macro_f1"])
        >= float(baseline["damage_macro_f1"]),
        "intact_iou_drop_within_limit": intact_drop_pp
        <= float(maximum_guard_drop_pp),
        "affected_recall_drop_within_limit": affected_recall_drop_pp
        <= float(maximum_guard_drop_pp),
        "event_worst_nondecreasing": float(
            candidate["event_worst_official_style_4class_miou"]
        )
        >= float(baseline["event_worst_official_style_4class_miou"]),
        "nonnegative_events_at_least_declared": nonnegative
        >= int(minimum_nonnegative_events),
    }
    return checks, diagnostics


__all__ = [
    "hierarchical_residual_loss",
    "hr3o_region_occurrence_loss",
    "hr_validation_checks",
    "severity_validation_checks",
]
