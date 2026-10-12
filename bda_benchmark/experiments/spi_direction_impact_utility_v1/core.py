from __future__ import annotations

import math

import torch


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


def direction_name(code: int) -> str:
    before, after = divmod(int(code), 4)
    return f"{CLASS_NAMES[before]}_to_{CLASS_NAMES[after]}"


def batched_metrics(histograms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return foreground mIoU and Damaged F1 for one or more 4x4 histograms."""
    hist = histograms.double()
    if hist.shape[-2:] != (4, 4):
        raise ValueError("histograms must end in [4,4]")
    ious = []
    for class_id in (1, 2, 3):
        tp = hist[..., class_id, class_id]
        union = (
            hist[..., class_id, :].sum(-1)
            + hist[..., :, class_id].sum(-1)
            - tp
        ).clamp_min(1.0)
        ious.append(tp / union)
    miou = torch.stack(ious, -1).mean(-1)
    damaged_tp = hist[..., 2, 2]
    damaged_denominator = (
        hist[..., 2, :].sum(-1) + hist[..., :, 2].sum(-1)
    ).clamp_min(1.0)
    damaged_f1 = 2.0 * damaged_tp / damaged_denominator
    return miou, damaged_f1


def exact_keep_utilities(
    fixed_histogram: torch.Tensor,
    delta_histograms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact one-cluster value of keeping S2 instead of reverting that cluster."""
    fixed = fixed_histogram.double()
    delta = delta_histograms.double()
    if fixed.shape != (4, 4) or delta.ndim != 3 or delta.shape[1:] != (4, 4):
        raise ValueError("expected fixed [4,4] and delta [N,4,4]")
    fixed_miou, fixed_damaged = batched_metrics(fixed)
    reverted_miou, reverted_damaged = batched_metrics(fixed[None] - delta)
    return (
        100.0 * (fixed_miou - reverted_miou),
        100.0 * (fixed_damaged - reverted_damaged),
    )


def weighted_binary_auroc(
    labels: torch.Tensor,
    scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    """Tie-aware weighted AUROC; weights apply to positive and negative samples."""
    labels = labels.bool().flatten().cpu()
    scores = scores.double().flatten().cpu()
    weights = (
        torch.ones_like(scores) if weights is None
        else weights.double().flatten().cpu().clamp_min(0)
    )
    if not (len(labels) == len(scores) == len(weights)) or not len(labels):
        raise ValueError("labels, scores and weights must be equal non-empty vectors")
    positive_weight = weights[labels].sum()
    negative_weight = weights[~labels].sum()
    if float(positive_weight) <= 0 or float(negative_weight) <= 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    labels, scores, weights = labels[order], scores[order], weights[order]
    concordant = torch.tensor(0.0, dtype=torch.float64)
    negative_before = torch.tensor(0.0, dtype=torch.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and bool(scores[end] == scores[start]):
            end += 1
        block_positive = weights[start:end][labels[start:end]].sum()
        block_negative = weights[start:end][~labels[start:end]].sum()
        concordant += block_positive * (negative_before + 0.5 * block_negative)
        negative_before += block_negative
        start = end
    return float(concordant / (positive_weight * negative_weight))


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool().flatten().cpu()
    scores = scores.double().flatten().cpu()
    positives = int(labels.sum())
    if not positives or positives == len(labels):
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].double()
    precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1, dtype=torch.double)
    return float(precision[ranked.bool()].mean())


def sign_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    decisive = (left != 0) & (right != 0)
    if not bool(decisive.any()):
        return float("nan")
    return float((torch.sign(left[decisive]) == torch.sign(right[decisive])).float().mean())


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    if not len(values):
        return {}
    return {
        name: float(torch.quantile(values, q))
        for name, q in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90))
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if abs(denominator) > 1e-12 else None


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


__all__ = [
    "average_precision", "batched_metrics", "direction_name",
    "exact_keep_utilities", "finite", "quantiles", "safe_ratio",
    "sign_agreement", "weighted_binary_auroc",
]

