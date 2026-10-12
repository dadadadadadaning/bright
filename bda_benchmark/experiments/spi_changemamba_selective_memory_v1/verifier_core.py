from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import balanced_event_folds, threshold_candidates, valid_miou


DIRECT_FEATURES = (
    "p3_p0", "p3_p1", "p3_p2", "p3_p3", "p3_entropy",
    "p3_occurrence_margin", "p3_fragmentation", "p3_uncertainty",
    "region_area", "building_probability", "direct_p0", "direct_p1", "direct_p2",
    "direct_margin", "direct_affected", "proposal_affected",
)
MEMORY_FEATURES = DIRECT_FEATURES + (
    "source_direct", "source_memory", "memory_p0", "memory_p1", "memory_p2",
    "memory_margin", "memory_agreement", "memory_relative_gap", "memory_density",
    "direct_memory_l1", "direct_memory_affected_delta", "same_prediction",
)
VARIANT_FEATURES = {"v1_direct": DIRECT_FEATURES, "v2_memory": MEMORY_FEATURES}


class LogisticVerifier(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(dimension), 1)

    def forward(self, value):
        return self.linear(value.float()).squeeze(1)


def filter_rows(rows: Sequence[Mapping[str, Any]], variant: str):
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"unknown verifier variant {variant}")
    if variant == "v1_direct":
        return [row for row in rows if row["source"] == "direct"]
    return list(rows)


def matrix(rows, names):
    return torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in names] for row in rows],
        dtype=torch.float32,
    )


def standardize(value, mean=None, std=None):
    mean = value.mean(0) if mean is None else mean
    std = value.std(0, unbiased=False).clamp_min(1e-5) if std is None else std
    return (value - mean) / std, mean, std


def fit(features, labels, utility, steps=1500, lr=0.03, seed=3407):
    torch.manual_seed(int(seed))
    model = LogisticVerifier(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    nonzero = utility.abs()[utility != 0]
    scale = nonzero.median() if nonzero.numel() else utility.new_tensor(1.0)
    weights = (utility.abs() / scale.clamp_min(1e-12)).clamp(0.10, 5.0)
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    balance = torch.where(labels > 0.5, negative / positive, labels.new_tensor(1.0))
    for _ in range(int(steps)):
        logits = model(features)
        loss = (
            F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            * weights * balance
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def select(rows, scores, threshold, conflict_margin=0.10):
    grouped = defaultdict(list)
    for index, (row, score) in enumerate(zip(rows, scores)):
        if float(score) >= float(threshold):
            grouped[str(row["region_key"])].append((index, row, float(score)))
    selected = []
    for alternatives in grouped.values():
        alternatives.sort(key=lambda item: (-item[2], str(item[1]["source"])))
        if len(alternatives) == 1:
            selected.append(alternatives[0][0])
            continue
        first, second = alternatives[0], alternatives[1]
        same_direction = bool(first[1]["target_affected"]) == bool(second[1]["target_affected"])
        if same_direction or first[2] - second[2] >= float(conflict_margin):
            selected.append(first[0])
    return selected


def apply(base_hist, rows, selected):
    output = base_hist.double().clone()
    for index in selected:
        output += torch.as_tensor(rows[index]["delta"], dtype=torch.float64)
    return output


def search_threshold(cache, rows, scores, maximum_acceptance=0.70, conflict_margin=0.10):
    base = valid_miou(cache["base_hist"])
    region_count = len({str(row["region_key"]) for row in rows})
    best = None
    for threshold in threshold_candidates():
        chosen = select(rows, scores, threshold, conflict_margin)
        acceptance = len(chosen) / max(region_count, 1)
        if acceptance > float(maximum_acceptance):
            continue
        final = apply(cache["base_hist"], rows, chosen)
        gain = 100.0 * (valid_miou(final) - base)
        event_delta = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
        for index in chosen:
            event_delta[str(rows[index]["event"])] += rows[index]["delta"].double()
        event_gains = [
            100.0 * (valid_miou(hist.double() + event_delta[str(event)]) - valid_miou(hist))
            for event, hist in cache["event_base_hist"].items()
        ]
        event_macro = sum(event_gains) / max(len(event_gains), 1)
        nonnegative = sum(value >= 0.0 for value in event_gains)
        objective = gain + 0.25 * event_macro
        row = {
            "threshold": threshold, "gain_pp": gain, "event_macro_gain_pp": event_macro,
            "nonnegative_events": nonnegative, "acceptance_rate": acceptance,
            "objective": objective,
        }
        if best is None or (row["objective"], -acceptance) > (best["objective"], -best["acceptance_rate"]):
            best = row
    return best


__all__ = [
    "VARIANT_FEATURES", "LogisticVerifier", "apply", "balanced_event_folds", "filter_rows",
    "fit", "matrix", "search_threshold", "select", "standardize",
]

