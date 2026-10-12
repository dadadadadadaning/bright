from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_macu_failure_audit_d0_v1.core import (
    class_f1_gradient,
    linear_utility,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    metric_utilities,
    miou_gradient,
    per_image_selective_histograms,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    metrics_from_histogram,
)


OUTCOMES = ("corrected", "harmed", "neutral")
VARIANTS = (
    "b1_shared_outcome",
    "b2_direction_outcome",
    "b3_expected_confusion",
    "b4_confident_expected_confusion",
)
REJECTION_GRID = (0.0, .01, .02, .05, .10)


def outcome_counts(cache: dict) -> torch.Tensor:
    rows = [cache["utility"][name].double() for name in OUTCOMES]
    counts = torch.stack(rows, 1)
    changed = cache["utility"]["changed"].double()
    if bool((counts < 0).any()) or not torch.allclose(counts.sum(1), changed):
        raise RuntimeError("corrected/harmed/neutral do not partition changed pixels")
    return counts


def outcome_proportions(cache: dict) -> torch.Tensor:
    counts = outcome_counts(cache)
    return counts / counts.sum(1, keepdim=True).clamp_min(1.0)


class SharedOutcomeModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, 3)

    def forward(self, features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        del direction
        return self.head(features)


class DirectionConditionedOutcomeModel(nn.Module):
    """Shared affine predictor plus shrunk direction-specific feature slopes."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.shared = nn.Linear(input_dim, 3)
        self.direction_weight = nn.Parameter(torch.zeros(16, input_dim, 3))
        self.direction_bias = nn.Parameter(torch.zeros(16, 3))

    def forward(self, features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        direction = direction.long()
        if bool(((direction < 0) | (direction >= 16)).any()):
            raise ValueError("direction code must be in [0,15]")
        interaction = torch.einsum(
            "nd,ndk->nk", features, self.direction_weight[direction]
        )
        return self.shared(features) + interaction + self.direction_bias[direction]

    def interaction_penalty(self) -> torch.Tensor:
        return self.direction_weight.square().mean() + self.direction_bias.square().mean()


def cluster_events(cache: dict) -> list[str]:
    return [
        cache["image_events"][int(index)]
        for index in cache["image_index"].long().tolist()
    ]


def event_balanced_cluster_weights(cache: dict, selected: torch.Tensor) -> torch.Tensor:
    events = [event for event, keep in zip(cluster_events(cache), selected.tolist()) if keep]
    counts = Counter(events)
    weights = torch.tensor([1.0 / counts[event] for event in events], dtype=torch.float32)
    return weights / weights.mean().clamp_min(1e-8)


def train_outcome_model(
    features: torch.Tensor,
    directions: torch.Tensor,
    cache: dict,
    selected: torch.Tensor,
    *, conditioned: bool,
    steps: int,
    batch_size: int,
    learning_rate: float,
    direction_shrinkage: float,
    seed: int,
) -> nn.Module:
    ids = torch.nonzero(selected, as_tuple=False).flatten()
    if not len(ids):
        raise RuntimeError("outcome fold has no training clusters")
    targets = outcome_proportions(cache).float()
    weights = event_balanced_cluster_weights(cache, selected)
    torch.manual_seed(seed)
    model: nn.Module = (
        DirectionConditionedOutcomeModel(features.shape[1])
        if conditioned else SharedOutcomeModel(features.shape[1])
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    size = min(int(batch_size), len(ids))
    for step in range(1, int(steps) + 1):
        local = torch.randint(len(ids), (size,), generator=generator)
        batch_ids = ids[local]
        log_probability = F.log_softmax(
            model(features[batch_ids], directions[batch_ids]), dim=1
        )
        # Each cluster has equal primary weight. Area is deliberately excluded:
        # impact enters only through expected counts during decision making.
        loss = (-(targets[batch_ids] * log_probability).sum(1) * weights[local]).mean()
        if conditioned:
            loss = loss + float(direction_shrinkage) * model.interaction_penalty()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite outcome loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"non-finite outcome gradient at step {step}")
        optimizer.step()
    return model


@torch.inference_mode()
def predict_outcomes(
    model: nn.Module, features: torch.Tensor, directions: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    return model(features, directions).softmax(1)


def neutral_truth_prior(
    cache: dict, selected: torch.Tensor, smoothing: float = 1.0,
) -> torch.Tensor:
    """Train-only P(true class | direction, neutral), with forbidden rows masked."""

    directions = cache["direction_code"].long()
    delta = cache["delta_hist"].double()
    prior = torch.zeros(16, 4, dtype=torch.float64)
    for code in range(16):
        before, after = divmod(code, 4)
        if before == after:
            continue
        allowed = torch.ones(4, dtype=torch.bool)
        allowed[before] = False
        allowed[after] = False
        rows = selected & (directions == code)
        # For a homogeneous before->after cluster, positive mass in column
        # `after` equals the number of pixels for each true-class row.
        values = delta[rows, :, after].sum(0).clamp_min(0)
        values[~allowed] = 0
        values[allowed] += float(smoothing)
        prior[code] = values / values.sum().clamp_min(1.0)
    return prior


def expected_confusion_delta(
    probabilities: torch.Tensor,
    direction_code: torch.Tensor,
    cluster_area: torch.Tensor,
    neutral_prior: torch.Tensor,
) -> torch.Tensor:
    """Convert local outcomes into expected candidate-minus-P3 confusion deltas."""

    n = len(probabilities)
    result = torch.zeros(n, 4, 4, dtype=torch.float64)
    expected = probabilities.double() * cluster_area.double()[:, None]
    direction_code = direction_code.long()
    for code in torch.unique(direction_code).tolist():
        before, after = divmod(int(code), 4)
        if before == after:
            raise ValueError("unchanged transition cannot define a modification cluster")
        ids = torch.nonzero(direction_code == code, as_tuple=False).flatten()
        corrected, harmed, neutral = expected[ids].unbind(1)
        result[ids, after, after] += corrected
        result[ids, after, before] -= corrected
        result[ids, before, after] += harmed
        result[ids, before, before] -= harmed
        neutral_rows = neutral[:, None] * neutral_prior[code][None]
        result[ids, :, after] += neutral_rows
        result[ids, :, before] -= neutral_rows
    return result


def simple_outcome_risk(probabilities: torch.Tensor) -> torch.Tensor:
    return probabilities[:, 1] - probabilities[:, 0]


def expected_confusion_risk(
    expected_delta: torch.Tensor,
    reference_histogram: torch.Tensor,
) -> torch.Tensor:
    metric = metric_utilities(expected_delta, miou_gradient(reference_histogram))
    damaged = linear_utility(expected_delta, class_f1_gradient(reference_histogram, 2))
    return torch.maximum(-metric, -damaged).float()


def outcome_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(1)
    return (1.0 - entropy / torch.log(torch.tensor(3.0))).clamp(0, 1)


def choose_policy(
    cache: dict,
    risk: torch.Tensor,
    cluster_mask: torch.Tensor,
    image_mask: torch.Tensor,
    *, maximum_reject_fraction: float,
) -> tuple[dict, dict]:
    fixed = metrics_from_histogram(cache["fixed_s2_histograms"][image_mask].sum(0))
    fallback = {
        "risk_threshold": None, "target_reject_fraction": 0.0,
        "actual_reject_fraction": 0.0, "fallback_fixed_s2": True,
        "miou": fixed["valid_miou"], "damaged_f1": fixed["damaged_f1"],
    }
    local = risk[cluster_mask]
    candidates = []
    diagnostics = {"candidate_total": 0, "damaged_unsafe": 0,
                   "miou_not_above_fixed": 0, "useful_nontrivial": 0}
    for fraction in REJECTION_GRID[1:]:
        diagnostics["candidate_total"] += 1
        threshold = float(torch.quantile(local.float(), 1.0 - fraction))
        local_accept = local < threshold
        rejected = float((~local_accept).float().mean())
        if rejected > maximum_reject_fraction + 1e-8:
            continue
        accept = torch.ones(len(risk), dtype=torch.bool)
        accept[cluster_mask] = local_accept
        selected = per_image_selective_histograms(cache, accept)
        report = metrics_from_histogram(selected[image_mask].sum(0))
        if report["damaged_f1"] < fixed["damaged_f1"]:
            diagnostics["damaged_unsafe"] += 1
            continue
        if report["valid_miou"] <= fixed["valid_miou"]:
            diagnostics["miou_not_above_fixed"] += 1
        else:
            diagnostics["useful_nontrivial"] += 1
        candidates.append({
            "risk_threshold": threshold, "target_reject_fraction": fraction,
            "actual_reject_fraction": rejected, "fallback_fixed_s2": False,
            "miou": report["valid_miou"], "damaged_f1": report["damaged_f1"],
        })
    policy = max(
        [fallback, *candidates],
        key=lambda row: (row["miou"], row["damaged_f1"], -row["actual_reject_fraction"]),
    )
    diagnostics["selected_policy"] = policy
    return policy, diagnostics


def apply_policy(risk: torch.Tensor, policy: dict) -> torch.Tensor:
    if policy["fallback_fixed_s2"]:
        return torch.ones(len(risk), dtype=torch.bool)
    return risk < float(policy["risk_threshold"])


__all__ = [
    "DirectionConditionedOutcomeModel", "OUTCOMES", "REJECTION_GRID",
    "SharedOutcomeModel", "VARIANTS", "apply_policy", "choose_policy",
    "event_balanced_cluster_weights", "expected_confusion_delta",
    "expected_confusion_risk", "neutral_truth_prior", "outcome_confidence",
    "outcome_counts", "outcome_proportions", "predict_outcomes",
    "simple_outcome_risk", "train_outcome_model",
]
