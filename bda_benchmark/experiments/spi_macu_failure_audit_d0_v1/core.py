from __future__ import annotations

import torch


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


def class_f1_gradient(histogram: torch.Tensor, class_index: int) -> torch.Tensor:
    """Analytic gradient of one class F1 for a true-row/predicted-column histogram."""

    histogram = histogram.double()
    if histogram.ndim != 2 or histogram.shape[0] != histogram.shape[1]:
        raise ValueError("histogram must be a square matrix")
    if not 0 <= int(class_index) < histogram.shape[0]:
        raise ValueError("class_index is out of range")
    k = int(class_index)
    true_positive = histogram[k, k]
    denominator = histogram[k, :].sum() + histogram[:, k].sum()
    if float(denominator) <= 0:
        return torch.zeros_like(histogram)
    gradient = torch.zeros_like(histogram)
    for truth in range(histogram.shape[0]):
        for prediction in range(histogram.shape[1]):
            d_tp = float(truth == k and prediction == k)
            d_denominator = float(truth == k) + float(prediction == k)
            gradient[truth, prediction] = (
                2.0 * d_tp * denominator
                - 2.0 * true_positive * d_denominator
            ) / denominator.square()
    return gradient


def linear_utility(delta_histograms: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    if delta_histograms.ndim != 3 or delta_histograms.shape[1:] != gradient.shape:
        raise ValueError("delta histogram/gradient shape mismatch")
    return 100.0 * (delta_histograms.double() * gradient.double()).sum((1, 2))


def direction_name(code: int) -> str:
    before, after = divmod(int(code), 4)
    return f"{CLASS_NAMES[before]}_to_{CLASS_NAMES[after]}"


def area_masks(areas: torch.Tensor) -> dict[str, torch.Tensor]:
    areas = areas.long()
    return {
        "33_64": (areas >= 33) & (areas <= 64),
        "65_128": (areas >= 65) & (areas <= 128),
        "129_256": (areas >= 129) & (areas <= 256),
        "257_512": (areas >= 257) & (areas <= 512),
        "gt_512": areas > 512,
    }


def mechanism_counts(
    accept: torch.Tensor,
    metric_utility: torch.Tensor,
    damaged_utility: torch.Tensor,
) -> dict[str, int]:
    accept = accept.bool()
    return {
        "accepted_metric_harm": int((accept & (metric_utility < 0)).sum()),
        "rejected_metric_benefit": int((~accept & (metric_utility > 0)).sum()),
        "accepted_damaged_harm": int((accept & (damaged_utility < 0)).sum()),
        "rejected_damaged_benefit": int((~accept & (damaged_utility > 0)).sum()),
    }


__all__ = [
    "CLASS_NAMES", "area_masks", "class_f1_gradient", "direction_name",
    "linear_utility", "mechanism_counts",
]
