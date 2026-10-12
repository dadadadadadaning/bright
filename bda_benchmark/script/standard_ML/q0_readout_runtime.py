"""Numerically explicit runtime helpers for the Q0 SAR readout."""
from __future__ import annotations

import re
import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


IGNORE_INDEX = 255
EVENT_RE = re.compile(r"_\d+$")


def event_name(sample: str) -> str:
    return EVENT_RE.sub("", str(sample))


def supervised_mask(
    damage: torch.Tensor,
    building_probability: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    building = (damage >= 1) & (damage <= 3)
    if building_probability is None:
        return building
    probability = building_probability
    if probability.ndim == 4:
        probability = probability[:, 0]
    if probability.shape != damage.shape:
        probability = F.interpolate(
            probability.unsqueeze(1).float(),
            damage.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return building & (probability >= float(threshold))


def masked_three_class_ce(
    logits: torch.Tensor,
    damage: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    if logits.shape[-2:] != damage.shape[-2:]:
        logits = F.interpolate(logits, damage.shape[-2:], mode="bilinear", align_corners=False)
    eligible = mask & (damage >= 1) & (damage <= 3)
    count = int(eligible.sum())
    if count == 0:
        # Preserve a valid graph without inventing supervision.
        return logits.sum() * 0.0, 0
    target = damage[eligible] - 1
    selected = logits.permute(0, 2, 3, 1)[eligible]
    loss = F.cross_entropy(selected.float(), target.long(), weight=class_weights)
    if not torch.isfinite(loss):
        raise FloatingPointError("Q0 masked CE became non-finite")
    return loss, count


def effective_number_weights(
    counts: Sequence[int | float], beta: float = 0.9999, maximum_ratio: float = 5.0
) -> torch.Tensor:
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.numel() != 3 or torch.any(values <= 0):
        raise ValueError(f"Q0 requires three positive train-only counts, got {list(counts)}")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0,1)")
    if beta == 0.0:
        weights = torch.ones_like(values)
    else:
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), values))
    if maximum_ratio < 1.0:
        raise ValueError("maximum_ratio must be at least 1")
    # A hard clamp can collapse several rare classes to exactly the same
    # weight.  Compress the dynamic range monotonically instead, preserving
    # the ordering implied by train-only counts while bounding max/min.
    raw_ratio = float(weights.max() / weights.min())
    if raw_ratio > maximum_ratio and maximum_ratio > 1.0:
        exponent = math.log(float(maximum_ratio)) / math.log(raw_ratio)
        weights = weights.pow(exponent)
    elif maximum_ratio == 1.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.mean()
    return weights.float()


def stage4_cell_counts(
    stage4: torch.Tensor, damage: torch.Tensor, minimum_purity: float = 0.75
) -> torch.Tensor:
    safe = damage.clamp(0, 3)
    valid = (damage != IGNORE_INDEX).float().unsqueeze(1)
    one_hot = F.one_hot(safe, num_classes=4).permute(0, 3, 1, 2).float() * valid
    size = stage4.shape[-2:]
    mass = F.interpolate(valid, size=size, mode="area")
    fractions = F.interpolate(one_hot, size=size, mode="area") / mass.clamp_min(1e-6)
    purity, labels = fractions.max(1)
    eligible = (purity >= minimum_purity) & (mass[:, 0] >= 0.5)
    return torch.stack([((labels == index) & eligible).sum() for index in (1, 2, 3)]).long()


def confusion_update(
    confusion: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> None:
    eligible = mask & (target >= 1) & (target <= 3)
    if not torch.any(eligible):
        return
    encoded = (target[eligible] - 1) * 3 + prediction[eligible]
    confusion += torch.bincount(encoded.cpu(), minlength=9).reshape(3, 3)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    matrix = confusion.double()
    tp = matrix.diag()
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    recall = tp / support.clamp_min(1)
    precision = tp / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = tp.sum() / matrix.sum().clamp_min(1)
    return {
        "confusion": matrix.long().tolist(),
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "intact_recall": float(recall[0]),
        "damaged_recall": float(recall[1]),
        "destroyed_recall": float(recall[2]),
        "intact_f1": float(f1[0]),
        "damaged_f1": float(f1[1]),
        "destroyed_f1": float(f1[2]),
        "support": support.long().tolist(),
    }


class Q0MetricAccumulator:
    def __init__(self) -> None:
        self.conditional = torch.zeros(3, 3, dtype=torch.long)
        self.event_confusions: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(3, 3, dtype=torch.long)
        )
        self.class_total = torch.zeros(3, dtype=torch.long)
        self.class_correct_covered = torch.zeros(3, dtype=torch.long)
        self.covered = 0
        self.building = 0

    def update(
        self,
        logits: torch.Tensor,
        damage: torch.Tensor,
        mask: torch.Tensor,
        names: Sequence[str],
    ) -> None:
        prediction = logits.argmax(1)
        confusion_update(self.conditional, prediction, damage, mask)
        building = (damage >= 1) & (damage <= 3)
        self.covered += int((mask & building).sum())
        self.building += int(building.sum())
        for class_index in (1, 2, 3):
            truth = damage == class_index
            self.class_total[class_index - 1] += int(truth.sum())
            self.class_correct_covered[class_index - 1] += int(
                (truth & mask & (prediction == class_index - 1)).sum()
            )
        for batch_index, name in enumerate(names):
            confusion_update(
                self.event_confusions[event_name(name)],
                prediction[batch_index : batch_index + 1],
                damage[batch_index : batch_index + 1],
                mask[batch_index : batch_index + 1],
            )

    def compute(self) -> dict[str, object]:
        conditional = metrics_from_confusion(self.conditional)
        event_metrics = {
            event: metrics_from_confusion(confusion)
            for event, confusion in sorted(self.event_confusions.items())
        }
        event_macro_f1 = float(
            np.mean([value["macro_f1"] for value in event_metrics.values()])
        ) if event_metrics else 0.0
        total = self.class_total.clamp_min(1).double()
        end_to_end = self.class_correct_covered.double() / total
        return {
            "conditional": conditional,
            "building_coverage": self.covered / max(self.building, 1),
            "end_to_end_intact_recall": float(end_to_end[0]),
            "end_to_end_damaged_recall": float(end_to_end[1]),
            "end_to_end_destroyed_recall": float(end_to_end[2]),
            "event_macro_f1": event_macro_f1,
            "event_metrics": event_metrics,
            "processed_building_pixels": self.building,
        }


class SourceSARAuxAccumulator:
    """Building-only audit of the original 4-class SAR auxiliary prediction.

    Prediction column zero is retained, so a Damaged->Background failure is
    not hidden by renormalising the remaining three logits.
    """

    def __init__(self) -> None:
        self.confusion = torch.zeros(3, 4, dtype=torch.long)

    def update(self, logits: torch.Tensor, damage: torch.Tensor) -> None:
        prediction = logits.argmax(1)
        building = (damage >= 1) & (damage <= 3)
        if not torch.any(building):
            return
        encoded = (damage[building] - 1) * 4 + prediction[building]
        self.confusion += torch.bincount(encoded.cpu(), minlength=12).reshape(3, 4)

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.double()
        support = matrix.sum(1).clamp_min(1)
        tp = torch.stack([matrix[index, index + 1] for index in range(3)])
        predicted = matrix[:, 1:].sum(0).clamp_min(1)
        recall = tp / support
        precision = tp / predicted
        f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-12)
        return {
            "confusion_gt3_by_prediction4": matrix.long().tolist(),
            "intact_recall": float(recall[0]),
            "damaged_recall": float(recall[1]),
            "destroyed_recall": float(recall[2]),
            "intact_f1": float(f1[0]),
            "damaged_f1": float(f1[1]),
            "destroyed_f1": float(f1[2]),
            "damaged_to_background": float(matrix[1, 0] / support[1]),
            "destroyed_to_background": float(matrix[2, 0] / support[2]),
        }


def selection_key(metrics: Mapping[str, object]) -> tuple[float, float, float, float]:
    conditional = metrics["conditional"]
    assert isinstance(conditional, Mapping)
    return (
        float(conditional["macro_f1"]),
        float(metrics["event_macro_f1"]),
        float(metrics["end_to_end_damaged_recall"]),
        float(conditional["damaged_recall"]),
    )


__all__ = [
    "Q0MetricAccumulator", "SourceSARAuxAccumulator", "effective_number_weights", "event_name",
    "masked_three_class_ce", "metrics_from_confusion", "selection_key",
    "stage4_cell_counts", "supervised_mask",
]
