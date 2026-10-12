from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    metrics_from_histogram,
)


BASE_VARIANTS = (
    "c0_raw_p3",
    "c1_local_window_a6",
    "c2_loccc_a6",
)
RGID_VARIANT = "c3_rgid_g3_a6"


def local_window_gated_majority(
    base_prediction: torch.Tensor,
    probabilities: torch.Tensor,
    localization_mask: torch.Tensor,
    *,
    kernel_size: int = 5,
    minimum_purity: float = 0.70,
    minimum_margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Generic local-mode control restricted to the same predicted building mask.

    It uses no connected components. The purity and probability-margin rules
    match A6, so C1-vs-C2 isolates local smoothing from whole-building context.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if probabilities.shape != (4, *base_prediction.shape):
        raise ValueError("probabilities must be [4,H,W] and match prediction")
    if localization_mask.shape != base_prediction.shape:
        raise ValueError("localization mask has a different shape")
    if not 0.0 <= minimum_purity <= 1.0 or minimum_margin < 0.0:
        raise ValueError("invalid local gate thresholds")

    mask = localization_mask.bool()
    padding = kernel_size // 2
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        dtype=torch.float32,
        device=base_prediction.device,
    )
    mask_f = mask.float()[None, None]
    support = F.conv2d(mask_f, kernel, padding=padding)[0, 0].clamp_min(1.0)

    counts = []
    means = []
    for class_id in (1, 2, 3):
        class_pixels = ((base_prediction == class_id) & mask).float()[None, None]
        counts.append(F.conv2d(class_pixels, kernel, padding=padding)[0, 0])
        weighted_probability = (probabilities[class_id] * mask.float())[None, None]
        means.append(
            F.conv2d(weighted_probability, kernel, padding=padding)[0, 0] / support
        )
    count_tensor = torch.stack(counts)
    mean_tensor = torch.stack(means)
    winning_index = count_tensor.argmax(dim=0)
    winning_label = winning_index + 1
    winning_count = count_tensor.gather(0, winning_index[None]).squeeze(0)
    purity = winning_count / support
    selected_probability = mean_tensor.gather(0, winning_index[None]).squeeze(0)
    alternatives = mean_tensor.clone()
    alternatives.scatter_(0, winning_index[None], float("-inf"))
    margin = selected_probability - alternatives.max(dim=0).values
    accepted = mask & (purity >= minimum_purity) & (margin >= minimum_margin)

    output = base_prediction.clone()
    output[accepted] = winning_label[accepted]
    changed = accepted & (output != base_prediction)
    return output, {
        "localization_pixels": int(mask.sum().item()),
        "accepted_pixels": int(accepted.sum().item()),
        "changed_pixels": int(changed.sum().item()),
        "kernel_size": int(kernel_size),
        "minimum_purity": float(minimum_purity),
        "minimum_margin": float(minimum_margin),
    }


def instance_labels_to_regions(
    instance_labels: np.ndarray | torch.Tensor,
    *,
    minimum_area: int = 32,
) -> list[torch.Tensor]:
    labels = torch.as_tensor(instance_labels).long().cpu()
    if labels.ndim != 2:
        raise ValueError("instance_labels must be [H,W]")
    regions = []
    for instance_id in torch.unique(labels).tolist():
        if int(instance_id) <= 0:
            continue
        mask = labels == int(instance_id)
        if int(mask.sum().item()) >= minimum_area:
            regions.append(mask)
    return regions


def event_macro_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    variants: Sequence[str],
    baseline: str,
    comparison_pairs: Sequence[tuple[str, str]],
    iterations: int = 10_000,
    seed: int = 3407,
) -> dict[str, object]:
    """Resample events and average event-level mIoU deltas equally."""

    by_event: dict[str, dict[str, torch.Tensor]] = defaultdict(
        lambda: {
            variant: torch.zeros((4, 4), dtype=torch.float64)
            for variant in variants
        }
    )
    for row in rows:
        event = str(row["event"])
        histograms = row["histograms"]
        for variant in variants:
            by_event[event][variant] += torch.as_tensor(
                histograms[variant], dtype=torch.float64
            )
    events = sorted(by_event)
    if not events:
        raise ValueError("cannot bootstrap an empty event table")
    event_miou = {
        event: {
            variant: float(metrics_from_histogram(by_event[event][variant])["valid_miou"])
            for variant in variants
        }
        for event in events
    }
    pairs = [(variant, baseline) for variant in variants if variant != baseline]
    pairs.extend(comparison_pairs)
    pairs = list(dict.fromkeys(pairs))
    rng = np.random.default_rng(seed)
    samples = {pair: [] for pair in pairs}
    for _ in range(iterations):
        selected = rng.integers(0, len(events), size=len(events))
        for left, right in pairs:
            delta = np.mean(
                [
                    event_miou[events[int(index)]][left]
                    - event_miou[events[int(index)]][right]
                    for index in selected
                ]
            )
            samples[(left, right)].append(100.0 * float(delta))
    output = {
        "iterations": int(iterations),
        "seed": int(seed),
        "unit": "percentage_points",
        "estimand": "equal-weight macro mean of event-level mIoU deltas",
        "event_count": len(events),
        "comparisons": {},
    }
    for (left, right), values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        output["comparisons"][f"{left}_minus_{right}"] = {
            "left": left,
            "right": right,
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }
    return output

