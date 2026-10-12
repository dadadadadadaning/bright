from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import valid_miou


COMMON_FEATURES = (
    "p3_p0_mean", "p3_p1_mean", "p3_p2_mean", "p3_p3_mean",
    "p3_p0_std", "p3_p1_std", "p3_p2_std", "p3_p3_std",
    "p3_affected_mean", "p3_affected_std", "p3_entropy_mean", "p3_entropy_std",
    "p3_occ_margin_mean", "p3_occ_margin_std",
    "base_frac0", "base_frac1", "base_frac2", "base_frac3",
    "log_pixel_count", "region_area_fraction", "building_probability",
    "alignment_entropy", "expected_offset", "rgb_edge_mean", "rgb_boundary_edge",
    "direct_p0", "direct_p1", "direct_p2", "direct_margin", "direct_affected",
)

UNION_BASIC_FEATURES = COMMON_FEATURES + (
    "source_direct", "source_memory", "proposal_affected",
    "direct_memory_same_prediction", "direct_memory_same_occurrence",
)

MEMORY_FEATURES = UNION_BASIC_FEATURES + (
    "memory_p0", "memory_p1", "memory_p2", "memory_margin", "memory_agreement",
    "memory_max_similarity", "memory_affected", "memory_event_count",
    "memory_event_entropy", "direct_memory_l1", "direct_memory_affected_delta",
)

VARIANT_FEATURES = {
    "m2_direct": COMMON_FEATURES + ("proposal_affected",),
    "m2_union_basic": UNION_BASIC_FEATURES,
    "m3_class_memory": MEMORY_FEATURES,
}


def feature_names(variant: str) -> tuple[str, ...]:
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"unknown variant {variant!r}; choose {sorted(VARIANT_FEATURES)}")
    return VARIANT_FEATURES[variant]


def proposal_filter(rows: Sequence[Mapping[str, Any]], variant: str) -> list[Mapping[str, Any]]:
    if variant == "m2_direct":
        return [row for row in rows if row["source"] == "direct"]
    feature_names(variant)
    return list(rows)


def matrix_from_rows(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> torch.Tensor:
    return torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in names] for row in rows],
        dtype=torch.float32,
    )


class LogisticVerifier(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(dimension), 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value.float()).squeeze(1)


def standardize(
    value: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mean is None:
        mean = value.mean(0)
    if std is None:
        std = value.std(0, unbiased=False).clamp_min(1e-5)
    return (value - mean) / std, mean, std


def utility_weights(utility: torch.Tensor, maximum: float = 5.0) -> torch.Tensor:
    nonzero = utility.abs()[utility != 0]
    scale = nonzero.median() if nonzero.numel() else utility.new_tensor(1.0)
    return (utility.abs() / scale.clamp_min(1e-12)).clamp(0.10, float(maximum))


def fit_logistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    utility: torch.Tensor,
    *,
    steps: int = 1500,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    seed: int = 3407,
) -> LogisticVerifier:
    torch.manual_seed(int(seed))
    model = LogisticVerifier(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    weights = utility_weights(utility)
    positives = labels.sum().clamp_min(1.0)
    negatives = (1.0 - labels).sum().clamp_min(1.0)
    balance = torch.where(labels > 0.5, negatives / positives, labels.new_tensor(1.0))
    for _ in range(int(steps)):
        logits = model(features)
        loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights * balance).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def select_proposals(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    threshold: float,
    conflict_margin: float = 0.10,
) -> list[int]:
    """Select at most one proposal per region; uncertain conflicts fall back to P3."""
    grouped: dict[str, list[tuple[int, Mapping[str, Any], float]]] = defaultdict(list)
    for index, (row, score) in enumerate(zip(rows, scores)):
        if float(score) >= float(threshold):
            grouped[str(row["region_key"])].append((index, row, float(score)))
    selected: list[int] = []
    for alternatives in grouped.values():
        alternatives.sort(key=lambda item: (-item[2], str(item[1]["source"])))
        if len(alternatives) == 1:
            selected.append(alternatives[0][0]); continue
        first, second = alternatives[0], alternatives[1]
        same_direction = bool(first[1]["target_affected"]) == bool(second[1]["target_affected"])
        if same_direction or first[2] - second[2] >= float(conflict_margin):
            selected.append(first[0])
    return selected


def apply_selected(base_hist: torch.Tensor, rows: Sequence[Mapping[str, Any]], selected: Iterable[int]) -> torch.Tensor:
    output = base_hist.double().clone()
    for index in selected:
        output += torch.as_tensor(rows[int(index)]["delta"], dtype=torch.float64)
    return output


def threshold_search(
    base_hist: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    maximum_acceptance: float = 0.70,
    conflict_margin: float = 0.10,
) -> dict[str, float]:
    base = valid_miou(base_hist)
    best = {"threshold": 0.95, "gain_pp": -1e9, "acceptance_rate": 0.0}
    for integer in range(5, 96):
        threshold = integer / 100.0
        selected = select_proposals(rows, scores, threshold=threshold, conflict_margin=conflict_margin)
        rate = len(selected) / max(len({str(row["region_key"]) for row in rows}), 1)
        if rate > float(maximum_acceptance):
            continue
        gain = 100.0 * (valid_miou(apply_selected(base_hist, rows, selected)) - base)
        if gain > best["gain_pp"] + 1e-12 or (
            abs(gain - best["gain_pp"]) <= 1e-12 and rate < best["acceptance_rate"]
        ):
            best = {"threshold": threshold, "gain_pp": gain, "acceptance_rate": rate}
    return best


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else value.item()
    return value


def entropy_from_counts(values: Sequence[str]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    probability = torch.tensor(list(counts.values()), dtype=torch.float32)
    probability /= probability.sum()
    entropy = float((-(probability * probability.clamp_min(1e-8).log()).sum() / math.log(max(len(counts), 2))).item())
    return float(len(counts)), entropy

