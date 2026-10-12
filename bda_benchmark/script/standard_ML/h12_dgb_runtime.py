# -*- coding: utf-8 -*-
"""Loss, selection and parameter-free evidence scores for H12-DGB."""
from __future__ import annotations

import re
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.h10_h3d_runtime import (
    evaluate_h10,
    hierarchical_damage_nll as _hierarchical_damage_nll,
    metrics_line,
)

_LOGIT_LIMIT = 20.0


def _finite_decision_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        logits.float(), nan=0.0, posinf=_LOGIT_LIMIT, neginf=-_LOGIT_LIMIT
    ).clamp(-_LOGIT_LIMIT, _LOGIT_LIMIT)


def hierarchical_damage_nll(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    """H10 NLL with finite occurrence/severity logits under AMP."""

    safe = {
        **outputs,
        "h10_affected_logits": _finite_decision_logits(outputs["h10_affected_logits"]),
        "h10_severity_logits": _finite_decision_logits(outputs["h10_severity_logits"]),
    }
    return _hierarchical_damage_nll(safe, labels, ignore_index)


def fn_aware_hierarchical_nll(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
    *,
    fn_weight: float = 2.0,
    fp_weight: float = 1.0,
    focal_gamma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Replace the Affected term with an FN-aware asymmetric focal loss.

    Severity and localisation objectives stay exactly the H10/H6-A2 terms.
    Background pixels still do not supervise either damage decision.
    """

    occurrence = _finite_decision_logits(outputs["h10_affected_logits"]).squeeze(1)
    severity = _finite_decision_logits(outputs["h10_severity_logits"]).squeeze(1)
    valid = (labels != ignore_index) & (labels >= 1) & (labels <= 3)
    if not valid.any():
        zero = occurrence.sum() * 0.0
        return {
            "extra_supervised_loss": zero,
            "h10_hierarchical_nll": zero,
            "h10_occurrence_nll": zero,
            "h10_conditional_severity_nll": zero,
            "h12_fn_aware_occurrence_nll": zero,
            "h10_supervised_building_pixels": zero.detach(),
        }

    occurrence_target = ((labels == 2) | (labels == 3)).float()
    probability = occurrence.sigmoid()
    correct_probability = torch.where(
        occurrence_target > 0.5, probability, 1.0 - probability
    )
    class_weight = torch.where(
        occurrence_target > 0.5,
        occurrence.new_full((), float(fn_weight)),
        occurrence.new_full((), float(fp_weight)),
    )
    occurrence_map = class_weight * (1.0 - correct_probability).clamp_min(0.0).pow(
        float(focal_gamma)
    ) * F.binary_cross_entropy_with_logits(
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
        severity_map = severity.new_zeros(severity.shape)
        severity_loss = severity.sum() * 0.0

    total = occurrence_map[valid].sum()
    if affected.any():
        total = total + severity_map[affected].sum()
    total = total / valid.sum().clamp_min(1)
    return {
        "extra_supervised_loss": total,
        "h10_hierarchical_nll": total,
        "h10_occurrence_nll": occurrence_loss,
        "h12_fn_aware_occurrence_nll": occurrence_loss,
        "h10_conditional_severity_nll": severity_loss,
        "h10_supervised_building_pixels": valid.sum().detach().float(),
    }


def event_protocol_key(metrics: Mapping[str, float]) -> tuple[float, float, float, float]:
    """Checkpoint order: event worst, event mean, Damaged IoU, valid mIoU."""

    return (
        float(metrics["event_worst_official_style_4class_miou"]),
        float(metrics["event_mean_official_style_4class_miou"]),
        float(metrics["damaged_iou"]),
        float(metrics["valid_miou"]),
    )


def update_h12_selection_records(
    records: MutableMapping[str, Dict[str, object]],
    metrics: Mapping[str, float],
    step: int,
) -> Sequence[str]:
    key = event_protocol_key(metrics)
    previous = records.get("valid", {}).get("score_tuple")
    improved = []
    if previous is None or key > tuple(previous):
        records["valid"] = {
            "score": key[0],
            "score_tuple": list(key),
            "step": int(step),
            "metrics": dict(metrics),
        }
        improved.append("valid")
    secondary = {
        "event": float(metrics["event_mean_official_style_4class_miou"]),
        "balance": float(metrics["damage_balance_score"]),
        "damaged": float(metrics["damaged_iou"]),
    }
    for name, score in secondary.items():
        previous_score = float(records.get(name, {}).get("score", float("-inf")))
        if score > previous_score:
            records[name] = {
                "score": score,
                "step": int(step),
                "metrics": dict(metrics),
            }
            improved.append(name)
    return improved


def binary_auprc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Average precision without sklearn."""

    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels, dtype=np.uint8).reshape(-1)
    finite = np.isfinite(score)
    score, label = score[finite], label[finite]
    positive = int((label == 1).sum())
    if positive == 0 or positive == label.size:
        return None
    order = np.argsort(-score, kind="mergesort")
    ranked = label[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / positive)


def parameter_free_difference_scores(
    rgb: torch.Tensor, sar: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Scores oriented so larger means more likely affected/damaged."""

    if rgb.shape != sar.shape or rgb.ndim != 4:
        raise ValueError("difference scores require equal [B,C,H,W] tensors")
    rgb_unit = F.normalize(rgb.float(), dim=1, eps=1e-6)
    sar_unit = F.normalize(sar.float(), dim=1, eps=1e-6)
    difference = (rgb.float() - sar.float()).abs()
    abs_diff_mean = difference.mean(1, keepdim=True)
    cosine_distance = 1.0 - (rgb_unit * sar_unit).sum(1, keepdim=True).clamp(-1.0, 1.0)
    local_mean = F.avg_pool2d(difference, kernel_size=3, stride=1, padding=1)
    local_second = F.avg_pool2d(difference.square(), kernel_size=3, stride=1, padding=1)
    local_std = (local_second - local_mean.square()).clamp_min(0.0).sqrt().mean(
        1, keepdim=True
    )
    product_energy = (rgb_unit * sar_unit).abs().mean(1, keepdim=True)
    return {
        "abs_diff_mean": abs_diff_mean,
        "cosine_distance": cosine_distance,
        "local_std": local_std,
        "product_energy": product_energy,
    }


def parameter_free_unimodal_scores(feature: torch.Tensor) -> Dict[str, torch.Tensor]:
    """SAR-only scores used for SARMAE block probes."""

    if feature.ndim != 4:
        raise ValueError("unimodal scores require [B,C,H,W] tensors")
    channel_abs = feature.float().abs().mean(1, keepdim=True)
    energy = feature.float().pow(2).mean(1, keepdim=True).sqrt()
    local_mean = F.avg_pool2d(channel_abs, kernel_size=3, stride=1, padding=1)
    local_second = F.avg_pool2d(channel_abs.square(), kernel_size=3, stride=1, padding=1)
    local_std = (local_second - local_mean.square()).clamp_min(0.0).sqrt()
    return {"energy": energy, "local_std": local_std}


class EventBalancedSampler(torch.utils.data.Sampler[int]):
    """Draw one event uniformly, then one image from that event."""

    def __init__(self, dataset, generator: torch.Generator | None = None) -> None:
        names = getattr(dataset, "data_list", None)
        if not names:
            raise ValueError("EventBalancedSampler requires dataset.data_list")
        groups: Dict[str, list[int]] = {}
        for index, name in enumerate(names):
            event = re.sub(r"_\d+$", "", str(name))
            groups.setdefault(event, []).append(index)
        self.groups = groups
        self.events = tuple(sorted(groups))
        self.num_samples = len(names)
        self.generator = generator

    def __iter__(self):
        for _ in range(self.num_samples):
            event_index = int(
                torch.randint(
                    len(self.events), (1,), generator=self.generator
                ).item()
            )
            event = self.events[event_index]
            members = self.groups[event]
            sample_index = int(
                torch.randint(len(members), (1,), generator=self.generator).item()
            )
            yield members[sample_index]

    def __len__(self) -> int:
        return self.num_samples


__all__ = (
    "evaluate_h10",
    "metrics_line",
    "hierarchical_damage_nll",
    "fn_aware_hierarchical_nll",
    "event_protocol_key",
    "update_h12_selection_records",
    "binary_auprc",
    "parameter_free_difference_scores",
    "parameter_free_unimodal_scores",
    "EventBalancedSampler",
)
