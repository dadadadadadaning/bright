# -*- coding: utf-8 -*-
"""Pure utilities for the validation-only M0 occurrence-boundary audit."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import metrics_from_hist


M0_THRESHOLDS_PP = {
    "damaged_iou": 0.30,
    "affected_recall": 0.50,
    "damaged_gate_miss": -0.50,
    "valid_miou": 0.0,
    "event_mean": 0.0,
    "event_worst": 0.0,
}


def normalize_betas(values: Iterable[float]) -> tuple[float, ...]:
    """Return sorted unique offsets and require an exact zero baseline."""
    betas = tuple(sorted({round(float(value), 8) for value in values}))
    if not betas:
        raise ValueError("at least one beta is required")
    if 0.0 not in betas:
        raise ValueError("M0 beta sweep must contain 0.0 as the P3 baseline")
    return betas


def apply_occurrence_offset(
    damage_logits: torch.Tensor,
    localization_logits: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift only Intact-vs-Affected inside frozen P3 building support.

    The same offset is added to Damaged and Destroyed, so their odds are
    preserved exactly.  A hard P3 localisation support prevents a threshold
    sweep from manufacturing buildings in the background.  No ground truth is
    consumed by this operation.
    """
    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ValueError("damage_logits must have shape [B,4,H,W]")
    if localization_logits.ndim != 4 or localization_logits.shape[1] != 2:
        raise ValueError("localization_logits must have shape [B,2,H,W]")
    if damage_logits.shape[0] != localization_logits.shape[0] or (
        damage_logits.shape[-2:] != localization_logits.shape[-2:]
    ):
        raise ValueError("damage/localisation logits must share batch and spatial shape")
    support = (localization_logits.argmax(1, keepdim=True) == 1).to(
        damage_logits.dtype
    )
    residual = support * float(beta)
    return (
        apply_hierarchical_residuals(
            damage_logits,
            occurrence_residual=residual,
        ),
        support,
    )


def new_occurrence_counters() -> MutableMapping[str, float]:
    return defaultdict(float)


def update_occurrence_counters(
    counters: MutableMapping[str, float],
    occurrence_prediction: torch.Tensor,
    severity_prediction: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> None:
    valid = labels != int(ignore_index)
    intact = valid & (labels == 1)
    damaged = valid & (labels == 2)
    destroyed = valid & (labels == 3)
    affected = damaged | destroyed
    counters["affected_tp"] += float((affected & occurrence_prediction).sum())
    counters["affected_fp"] += float((intact & occurrence_prediction).sum())
    counters["affected_fn"] += float((affected & ~occurrence_prediction).sum())
    counters["intact_total"] += float(intact.sum())
    counters["intact_to_affected"] += float((intact & occurrence_prediction).sum())
    counters["damaged_total"] += float(damaged.sum())
    counters["damaged_gate_miss"] += float((damaged & ~occurrence_prediction).sum())
    counters["damaged_severity_to_destroyed"] += float(
        (damaged & occurrence_prediction & severity_prediction).sum()
    )
    counters["destroyed_total"] += float(destroyed.sum())
    counters["destroyed_gate_miss"] += float(
        (destroyed & ~occurrence_prediction).sum()
    )


def finish_metrics(
    loc_hist: torch.Tensor,
    damage_hist: torch.Tensor,
    counters: Mapping[str, float],
    event_hist: Mapping[str, torch.Tensor],
    processed_images: int,
) -> Dict[str, float]:
    metrics = metrics_from_hist(loc_hist, damage_hist)
    tp = float(counters.get("affected_tp", 0.0))
    fp = float(counters.get("affected_fp", 0.0))
    fn = float(counters.get("affected_fn", 0.0))
    metrics["affected_iou"] = tp / max(tp + fp + fn, 1.0)
    metrics["affected_f1"] = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    metrics["affected_recall"] = tp / max(tp + fn, 1.0)
    metrics["intact_to_affected"] = float(
        counters.get("intact_to_affected", 0.0)
    ) / max(float(counters.get("intact_total", 0.0)), 1.0)
    metrics["damaged_gate_miss"] = float(
        counters.get("damaged_gate_miss", 0.0)
    ) / max(float(counters.get("damaged_total", 0.0)), 1.0)
    metrics["damaged_severity_to_destroyed"] = float(
        counters.get("damaged_severity_to_destroyed", 0.0)
    ) / max(float(counters.get("damaged_total", 0.0)), 1.0)
    metrics["destroyed_gate_miss"] = float(
        counters.get("destroyed_gate_miss", 0.0)
    ) / max(float(counters.get("destroyed_total", 0.0)), 1.0)
    metrics["processed_images"] = int(processed_images)

    event_values = []
    for histogram in event_hist.values():
        intersection = torch.diag(histogram)
        union = histogram.sum(1) + histogram.sum(0) - intersection
        event_values.append(float((intersection / union.clamp_min(1.0)).mean()))
    metrics["event_count"] = len(event_values)
    metrics["event_mean_official_style_4class_miou"] = (
        sum(event_values) / len(event_values) if event_values else 0.0
    )
    metrics["event_worst_official_style_4class_miou"] = (
        min(event_values) if event_values else 0.0
    )
    return metrics


def deltas_pp(
    observed: Mapping[str, float], baseline: Mapping[str, float]
) -> Dict[str, float]:
    keys = {
        "valid_miou": "valid_miou",
        "miou4": "miou4",
        "damaged_iou": "damaged_iou",
        "destroyed_iou": "destroyed_iou",
        "affected_recall": "affected_recall",
        "damaged_gate_miss": "damaged_gate_miss",
        "intact_to_affected": "intact_to_affected",
        "event_mean": "event_mean_official_style_4class_miou",
        "event_worst": "event_worst_official_style_4class_miou",
    }
    return {
        name: 100.0 * (float(observed[key]) - float(baseline[key]))
        for name, key in keys.items()
    }


def global_boundary_gate(
    delta: Mapping[str, float], nonnegative_events: int
) -> Dict[str, object]:
    checks = {
        "damaged_iou": float(delta["damaged_iou"]) >= 0.30,
        "affected_recall": float(delta["affected_recall"]) >= 0.50,
        "damaged_gate_miss": float(delta["damaged_gate_miss"]) <= -0.50,
        "valid_miou_nondecreasing": float(delta["valid_miou"]) >= 0.0,
        "event_mean_nondecreasing": float(delta["event_mean"]) >= 0.0,
        "event_worst_nondecreasing": float(delta["event_worst"]) >= 0.0,
        "nonnegative_events": int(nonnegative_events) >= 10,
    }
    return {"passed": all(checks.values()), "checks": checks}


def choose_global_candidate(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Choose a nonzero candidate without hiding event robustness failures."""
    nonzero = [record for record in records if float(record["beta"]) != 0.0]
    if not nonzero:
        raise ValueError("M0 requires at least one nonzero beta")
    passed = [record for record in nonzero if bool(record["gate"]["passed"])]  # type: ignore[index]
    pool = passed if passed else nonzero
    return max(
        pool,
        key=lambda record: (
            float(record["metrics"]["event_worst_official_style_4class_miou"]),  # type: ignore[index]
            float(record["metrics"]["event_mean_official_style_4class_miou"]),  # type: ignore[index]
            float(record["metrics"]["damaged_iou"]),  # type: ignore[index]
            float(record["metrics"]["valid_miou"]),  # type: ignore[index]
        ),
    )


__all__ = (
    "M0_THRESHOLDS_PP",
    "apply_occurrence_offset",
    "choose_global_candidate",
    "deltas_pp",
    "finish_metrics",
    "global_boundary_gate",
    "new_occurrence_counters",
    "normalize_betas",
    "update_occurrence_counters",
)
