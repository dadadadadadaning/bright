from __future__ import annotations

import math
from typing import Mapping

import torch


SHAPE_NAMES = ("log_area", "bbox_fill", "log_aspect", "boundary_fraction")
SEMANTIC_NAMES = (
    "entropy_mean", "entropy_std", "entropy_q25", "entropy_q75",
    "margin_mean", "margin_std", "margin_q25", "margin_q75",
    "base_background_fraction", "base_intact_fraction",
    "base_damaged_fraction", "base_destroyed_fraction",
)
S2_NAMES = (
    "changed_fraction", "probability_l1_mean", "probability_l1_std",
    "probability_l1_q75", "region_purity", "region_margin",
) + tuple(f"direction_{before}_to_{after}" for before in range(4) for after in range(4))


def _moments(values: torch.Tensor) -> torch.Tensor:
    values = values.float().flatten()
    if not values.numel():
        return torch.zeros(4)
    return torch.stack(
        (
            values.mean(),
            values.std(unbiased=False),
            torch.quantile(values, 0.25),
            torch.quantile(values, 0.75),
        )
    )


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    interior = mask.clone()
    interior[1:] &= mask[:-1]
    interior[:-1] &= mask[1:]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def summarize_evidence(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return mean/std/q75 for every [C,H,W] evidence channel."""

    if values.ndim != 3 or mask.shape != values.shape[-2:]:
        raise ValueError("evidence and building mask do not align")
    rows = []
    for channel in values.float():
        selected = channel[mask]
        rows.extend(
            (
                selected.mean(),
                selected.std(unbiased=False),
                torch.quantile(selected, 0.75),
            )
        )
    return torch.stack(rows) if rows else torch.empty(0)


def evidence_feature_names(channel_names) -> tuple[str, ...]:
    return tuple(
        f"{name}_{statistic}"
        for name in channel_names
        for statistic in ("mean", "std", "q75")
    )


def building_feature_groups(
    *,
    mask: torch.Tensor,
    base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    record: Mapping[str, float | int | bool],
    evidence: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if mask.ndim != 2 or not bool(mask.any()):
        raise ValueError("building mask must be non-empty [H,W]")
    height, width = mask.shape
    yy, xx = torch.nonzero(mask, as_tuple=True)
    area = int(mask.sum())
    bbox_area = int((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1))
    bbox_height = float(yy.max() - yy.min() + 1)
    bbox_width = float(xx.max() - xx.min() + 1)
    shape = torch.tensor(
        [
            math.log1p(area) / math.log1p(height * width),
            area / max(bbox_area, 1),
            math.log((bbox_width + 1e-6) / (bbox_height + 1e-6)),
            float(_boundary(mask).sum()) / max(area, 1),
        ],
        dtype=torch.float32,
    )

    probabilities = base_probabilities.float().clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(0) / torch.log(torch.tensor(4.0))
    top2 = probabilities.topk(2, dim=0).values
    margin = top2[0] - top2[1]
    entropy_stats = _moments(entropy[mask])
    margin_stats = _moments(margin[mask])
    base_fractions = torch.bincount(base_prediction[mask].long(), minlength=4).float() / area
    semantic = torch.cat((entropy_stats, margin_stats, base_fractions))

    changed = mask & (base_prediction != candidate_prediction)
    changed_count = int(changed.sum())
    l1 = (candidate_probabilities.float() - base_probabilities.float()).abs().sum(0)
    l1_values = l1[mask]
    transition = 4 * base_prediction[changed].long() + candidate_prediction[changed].long()
    transition_fraction = torch.bincount(transition, minlength=16).float() / max(changed_count, 1)
    s2 = torch.cat(
        (
            torch.tensor(
                [
                    changed_count / area,
                    float(l1_values.mean()),
                    float(l1_values.std(unbiased=False)),
                    float(torch.quantile(l1_values, 0.75)),
                    float(record["purity"]),
                    float(record["margin"]),
                ]
            ),
            transition_fraction,
        )
    ).float()
    output = {"shape": shape, "semantic": semantic, "s2": s2}
    for name, values in evidence.items():
        output[name] = summarize_evidence(values, mask)
    return output


def utility_counts(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, int | float]:
    valid = mask & (target >= 0) & (target <= 3)
    changed = valid & (base_prediction != candidate_prediction)
    corrected = changed & (candidate_prediction == target) & (base_prediction != target)
    harmed = changed & (base_prediction == target) & (candidate_prediction != target)
    neutral = changed & ~corrected & ~harmed
    changed_count = int(changed.sum())
    net = int(corrected.sum()) - int(harmed.sum())
    return {
        "changed": changed_count,
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "neutral": int(neutral.sum()),
        "net": net,
        "utility_ratio": net / max(changed_count, 1),
    }


def oracle_predictions(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
    region_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct-count building oracle and P3/S2 restricted pixel oracle."""

    building = base_prediction.clone()
    pixel = base_prediction.clone()
    valid = (target >= 0) & (target <= 3)
    changed = valid & (base_prediction != candidate_prediction)
    corrected = changed & (candidate_prediction == target) & (base_prediction != target)
    pixel[corrected] = candidate_prediction[corrected]
    for label in range(1, int(region_labels.max()) + 1):
        mask = region_labels == label
        counts = utility_counts(base_prediction, candidate_prediction, target, mask)
        if int(counts["net"]) > 0:
            building[mask] = candidate_prediction[mask]
    return building, pixel
