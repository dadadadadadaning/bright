# -*- coding: utf-8 -*-
"""Training utilities for H14-UWDC."""
from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def error_conditioned_occurrence_bce(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
    *,
    missed_damaged_weight: float = 4.0,
    missed_destroyed_weight: float = 2.0,
    intact_false_alarm_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """One occurrence objective concentrated on locked-A2 correction errors.

    This replaces, rather than augments, the generic hierarchical NLL.  The
    severity log-odds are frozen and therefore need no supervised term here.
    """

    occurrence = outputs["h10_affected_logits"].float().squeeze(1)
    base = outputs["h10_base_affected_logits"].detach().float().squeeze(1)
    valid = (labels != ignore_index) & (labels >= 1) & (labels <= 3)
    if not valid.any():
        zero = occurrence.sum() * 0.0
        return {
            "extra_supervised_loss": zero,
            "h14_error_conditioned_occurrence_bce": zero,
            "h14_missed_damaged_pixels": zero.detach(),
            "h14_missed_destroyed_pixels": zero.detach(),
            "h14_intact_false_alarm_pixels": zero.detach(),
        }

    target = ((labels == 2) | (labels == 3)).float()
    base_affected = base >= 0.0
    missed_damaged = valid & (labels == 2) & (~base_affected)
    missed_destroyed = valid & (labels == 3) & (~base_affected)
    intact_false_alarm = valid & (labels == 1) & base_affected
    weight = torch.ones_like(occurrence)
    weight = torch.where(
        missed_damaged,
        weight.new_full((), float(missed_damaged_weight)),
        weight,
    )
    weight = torch.where(
        missed_destroyed,
        weight.new_full((), float(missed_destroyed_weight)),
        weight,
    )
    weight = torch.where(
        intact_false_alarm,
        weight.new_full((), float(intact_false_alarm_weight)),
        weight,
    )
    loss_map = F.binary_cross_entropy_with_logits(
        occurrence, target, reduction="none"
    )
    loss = (loss_map[valid] * weight[valid]).sum() / weight[valid].sum().clamp_min(1.0)
    return {
        "extra_supervised_loss": loss,
        "h14_error_conditioned_occurrence_bce": loss,
        "h14_missed_damaged_pixels": missed_damaged.sum().detach().float(),
        "h14_missed_destroyed_pixels": missed_destroyed.sum().detach().float(),
        "h14_intact_false_alarm_pixels": intact_false_alarm.sum().detach().float(),
    }


__all__ = ["error_conditioned_occurrence_bce"]
