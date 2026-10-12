from __future__ import annotations

import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram


def miou_gradient(histogram: torch.Tensor) -> torch.Tensor:
    """Gradient of foreground dataset-mIoU with respect to a 4x4 confusion matrix.

    Rows are truth and columns are prediction. Background is excluded from the
    mean but foreground false positives in the background row still matter.
    """
    hist = histogram.double()
    if hist.shape != (4, 4):
        raise ValueError("histogram must be [4,4]")
    gradient = torch.zeros_like(hist)
    for class_id in (1, 2, 3):
        tp = hist[class_id, class_id]
        union = hist[class_id].sum() + hist[:, class_id].sum() - tp
        union = union.clamp_min(1.0)
        # dIoU/dh[r,c]: diagonal TP contributes +1 to numerator and +1 to
        # row/column but is subtracted once from union. Off-diagonal elements
        # in the class row/column only increase the union.
        gradient[class_id, class_id] += (union - tp) / union.square()
        for column in range(4):
            if column != class_id:
                gradient[class_id, column] += -tp / union.square()
        for row in range(4):
            if row != class_id:
                gradient[row, class_id] += -tp / union.square()
    return gradient / 3.0


def metric_utilities(delta_hist: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """First-order percentage-point value of keeping each S2 cluster."""
    if delta_hist.ndim != 3 or delta_hist.shape[-2:] != (4, 4):
        raise ValueError("delta_hist must be [N,4,4]")
    return 100.0 * (delta_hist.double() * gradient.double()[None]).sum((1, 2))


def selective_histogram(
    fixed_histograms: torch.Tensor,
    delta_hist: torch.Tensor,
    accept: torch.Tensor,
) -> torch.Tensor:
    """Start from fixed S2 and revert every rejected eligible cluster to P3."""
    fixed = fixed_histograms.sum(0).double()
    reject = ~accept.bool()
    return fixed - (
        delta_hist[reject].sum(0).double()
        if bool(reject.any()) else torch.zeros_like(fixed)
    )


def exact_marginal_utilities(cache: dict) -> torch.Tensor:
    """Exact one-cluster-at-a-time mIoU marginal, used only as a diagnostic."""
    fixed = cache["fixed_s2_histograms"].sum(0).double()
    baseline = metrics_from_histogram(fixed)["valid_miou"]
    rows = []
    for delta in cache["delta_hist"]:
        reverted = metrics_from_histogram(fixed - delta.double())["valid_miou"]
        rows.append(100.0 * (baseline - reverted))
    return torch.tensor(rows, dtype=torch.float64)


def metric_labels(utility: torch.Tensor, epsilon_pp: float = 0.0) -> torch.Tensor:
    labels = torch.full(utility.shape, -1, dtype=torch.long)
    labels[utility > epsilon_pp] = 1
    labels[utility < -epsilon_pp] = 0
    return labels


def per_image_selective_histograms(cache: dict, accept: torch.Tensor) -> torch.Tensor:
    rows = []
    image_index = cache["image_index"].long()
    for index in range(len(cache["image_names"])):
        current = image_index == index
        reject = current & ~accept.bool()
        histogram = cache["fixed_s2_histograms"][index].double()
        if bool(reject.any()):
            histogram = histogram - cache["delta_hist"][reject].sum(0).double()
        rows.append(histogram)
    return torch.stack(rows)


__all__ = [
    "exact_marginal_utilities", "metric_labels", "metric_utilities",
    "miou_gradient", "per_image_selective_histograms", "selective_histogram",
]
