"""Minimal ordinal supervision for Intact/Damaged/Destroyed."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_building_mask(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return (labels != ignore_index) & (labels >= 1) & (labels <= 3)


def masked_severity_cross_entropy(
    severity_logits: torch.Tensor,
    damage_labels: torch.Tensor,
    *,
    ignore_index: int = 255,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Three-class CE evaluated only on labelled building pixels."""
    if severity_logits.ndim != 4 or severity_logits.shape[1] != 3:
        raise ValueError("severity_logits must be [B,3,H,W]")
    mask = _valid_building_mask(damage_labels, ignore_index)
    if not mask.any():
        return severity_logits.sum() * 0.0
    logits = severity_logits.permute(0, 2, 3, 1)[mask].float()
    targets = damage_labels[mask].long() - 1
    return F.cross_entropy(logits, targets, weight=class_weight)


def expected_severity_distance(
    severity_logits: torch.Tensor,
    damage_labels: torch.Tensor,
    *,
    ignore_index: int = 255,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 distance between expected severity and the ordered target 0/1/2."""
    if severity_logits.ndim != 4 or severity_logits.shape[1] != 3:
        raise ValueError("severity_logits must be [B,3,H,W]")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")
    mask = _valid_building_mask(damage_labels, ignore_index)
    probability = severity_logits.float().softmax(1)
    ranks = probability.new_tensor([0.0, 1.0, 2.0]).view(1, 3, 1, 1)
    expected = (probability * ranks).sum(1)
    target = damage_labels.float() - 1.0
    distance = (expected - target).abs()
    selected = distance[mask]
    if reduction == "none":
        output = distance.new_zeros(distance.shape)
        output[mask] = selected
        return output
    if selected.numel() == 0:
        return severity_logits.sum() * 0.0
    return selected.mean() if reduction == "mean" else selected.sum()
