# -*- coding: utf-8 -*-
"""Pure metrics for no-training Occurrence/Severity audits."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from script.standard_ML.rm_region_retrieval_runtime import binary_auroc


def binary_metrics(score: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float | int]:
    score = score.float().flatten().cpu()
    target = target.bool().flatten().cpu()
    if score.numel() != target.numel() or score.numel() == 0:
        raise ValueError("binary metric inputs must be aligned and non-empty")
    prediction = score >= float(threshold)
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auroc": binary_auroc(score, target),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(target.numel(), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def event_binary_metrics(
    score: torch.Tensor, target: torch.Tensor, events: Sequence[str], threshold: float = 0.5
) -> Dict[str, object]:
    values = {}
    for event in sorted(set(map(str, events))):
        mask = torch.tensor([str(value) == event for value in events], dtype=torch.bool)
        local_target = target[mask]
        if int(local_target.sum()) == 0 or int((~local_target).sum()) == 0:
            continue
        values[event] = binary_metrics(score[mask], local_target, threshold)
    return {
        "comparable_events": len(values),
        "event_macro_auroc": sum(row["auroc"] for row in values.values()) / max(len(values), 1),
        "event_macro_f1": sum(row["f1"] for row in values.values()) / max(len(values), 1),
        "per_event": values,
    }

