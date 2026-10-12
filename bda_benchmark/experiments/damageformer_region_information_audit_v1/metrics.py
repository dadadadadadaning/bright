from __future__ import annotations

from typing import Dict

import torch


def confusion_metrics(hist: torch.Tensor) -> Dict[str, object]:
    hist = hist.double()
    tp = hist.diag()
    recall = tp / hist.sum(1).clamp_min(1)
    precision = tp / hist.sum(0).clamp_min(1)
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-12)
    iou = tp / (hist.sum(1) + hist.sum(0) - tp).clamp_min(1)
    return {
        "accuracy": float(tp.sum() / hist.sum().clamp_min(1)),
        "macro_f1": float(f1.mean()),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "iou": iou.tolist(),
        "confusion": hist.long().tolist(),
    }


def add_hist(hist: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, weight=None) -> None:
    target = target.long().reshape(-1)
    prediction = prediction.long().reshape(-1)
    if weight is None:
        weight = torch.ones_like(target, dtype=torch.long)
    code = target * hist.shape[1] + prediction
    hist += torch.bincount(code.cpu(), weights=weight.cpu().double(), minlength=hist.numel()).reshape_as(hist)

