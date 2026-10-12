# -*- coding: utf-8 -*-
"""Pure utilities for proposal-conditional IM1 reliability auditing.

IM1 never changes P3 predictions.  It asks whether train-only calibrated
signals can rank useful P3-Intact -> Memory-Damaged rescues above harmful
Intact modifications.  All feature normalisation and logistic parameters are
fit on train-memory leave-one-event-out proposals only.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from script.standard_ML.rm_region_retrieval_runtime import binary_auroc


BASE_SIGNAL_NAMES = (
    "p3_uncertainty",
    "p3_affected_probability",
    "retrieval_damaged_score",
    "retrieval_margin",
)

MATCHING_SIGNAL_NAMES = (
    "cosine_distance_q90",
    "cosine_distance_std",
    "peak_gain_q90",
    "one_minus_peak_q90",
    "one_minus_peak_std",
    "margin_mean",
    "entropy_mean",
)

COMBINED_MATCHING_SIGNAL_NAMES = (
    "cosine_distance_q90",
    "cosine_distance_std",
    "peak_gain_q90",
    "one_minus_peak_q90",
    "margin_mean",
    "entropy_mean",
)


def _column_indices(available: Sequence[str], requested: Sequence[str]) -> list[int]:
    positions = {str(name): index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in positions]
    if missing:
        raise KeyError(f"matching cache is missing signal columns: {missing}")
    return [positions[name] for name in requested]


def proposal_signal_table(
    cache: Mapping[str, object],
    retrieval_prediction: torch.Tensor,
    retrieval_scores: torch.Tensor,
) -> Dict[str, object]:
    """Build fixed IM1-A/B/C features on auditable I->D proposals."""

    required = (
        "labels", "base_predictions", "base_probabilities", "events",
        "colocated_stats", "matching_stats", "metadata",
    )
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"IM1 cache is missing: {missing}")

    truth = cache["labels"].long().flatten().cpu()
    base = cache["base_predictions"].long().flatten().cpu()
    base_probability = cache["base_probabilities"].float().cpu()
    retrieval_prediction = retrieval_prediction.long().flatten().cpu()
    retrieval_scores = retrieval_scores.float().cpu()
    count = truth.numel()
    if base_probability.shape != (count, 3):
        raise ValueError("IM1 expects P3 conditional probabilities [N,3]")
    if retrieval_scores.shape != (count, 3):
        raise ValueError("IM1 expects retrieval class scores [N,3]")
    if retrieval_prediction.numel() != count or len(cache["events"]) != count:
        raise ValueError("IM1 cache/retrieval counts do not match")

    proposed = (base == 0) & (retrieval_prediction == 1)
    auditable = proposed & ((truth == 0) | (truth == 1))
    indices = torch.nonzero(auditable, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("IM1 found no auditable P3-Intact -> Memory-Damaged proposals")
    target = truth[indices] == 1
    if int(target.sum()) == 0 or int((~target).sum()) == 0:
        raise RuntimeError("IM1 requires both rescue and harm proposals")

    p3 = base_probability[indices]
    p3_sorted = p3.sort(dim=1, descending=True).values
    damaged_score = retrieval_scores[indices, 1]
    retrieval_margin = damaged_score - torch.maximum(
        retrieval_scores[indices, 0], retrieval_scores[indices, 2]
    )
    base_signals = torch.stack(
        (
            1.0 - (p3_sorted[:, 0] - p3_sorted[:, 1]),
            p3[:, 1:3].sum(dim=1),
            damaged_score,
            retrieval_margin,
        ),
        dim=1,
    )

    metadata = cache["metadata"]
    colocated_names = list(map(str, metadata.get("colocated_stat_names", [])))
    matching_names = list(map(str, metadata.get("matching_stat_names", [])))
    colocated = cache["colocated_stats"].float().cpu()[indices]
    matching = cache["matching_stats"].float().cpu()[indices]
    matching_columns = {
        "cosine_distance_q90": colocated[:, _column_indices(
            colocated_names, ("cosine_distance_q90",)
        )[0]],
        "cosine_distance_std": colocated[:, _column_indices(
            colocated_names, ("cosine_distance_std",)
        )[0]],
        "peak_gain_q90": matching[:, _column_indices(
            matching_names, ("peak_gain_q90",)
        )[0]],
        "one_minus_peak_q90": matching[:, _column_indices(
            matching_names, ("one_minus_peak_q90",)
        )[0]],
        "one_minus_peak_std": matching[:, _column_indices(
            matching_names, ("one_minus_peak_std",)
        )[0]],
        "margin_mean": matching[:, _column_indices(
            matching_names, ("margin_mean",)
        )[0]],
        "entropy_mean": matching[:, _column_indices(
            matching_names, ("entropy_mean",)
        )[0]],
    }
    matching_signals = torch.stack(
        [matching_columns[name] for name in MATCHING_SIGNAL_NAMES], dim=1
    )
    combined_matching = torch.stack(
        [matching_columns[name] for name in COMBINED_MATCHING_SIGNAL_NAMES], dim=1
    )
    groups = {
        "im1_a_p3_memory": {
            "names": list(BASE_SIGNAL_NAMES),
            "features": base_signals,
        },
        "im1_b_matching": {
            "names": list(MATCHING_SIGNAL_NAMES),
            "features": matching_signals,
        },
        "im1_c_combined": {
            "names": list(BASE_SIGNAL_NAMES + COMBINED_MATCHING_SIGNAL_NAMES),
            "features": torch.cat((base_signals, combined_matching), dim=1),
        },
    }
    return {
        "indices": indices,
        "target": target,
        "events": [str(cache["events"][int(index)]) for index in indices],
        "groups": groups,
        "proposed_total": int(proposed.sum()),
        "auditable_proposals": int(indices.numel()),
        "rescue_proposals": int(target.sum()),
        "harm_proposals": int((~target).sum()),
        "destroyed_proposals_excluded": int((proposed & (truth == 2)).sum()),
    }


def _weighted_moments(
    features: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = weights.float().flatten().clamp_min(0.0)
    denominator = weights.sum().clamp_min(1e-8)
    mean = (features.float() * weights[:, None]).sum(dim=0) / denominator
    variance = (
        (features.float() - mean).square() * weights[:, None]
    ).sum(dim=0) / denominator
    return mean, variance.clamp_min(1e-8).sqrt()


def fit_logistic_calibrator(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    l2: float = 1e-2,
    max_iter: int = 200,
    balance_classes: bool = True,
) -> Dict[str, object]:
    """Fit a deterministic tiny logistic model without sklearn."""

    x = features.detach().float().cpu()
    y = target.detach().float().flatten().cpu()
    if x.ndim != 2 or x.shape[0] != y.numel() or x.shape[0] == 0:
        raise ValueError("logistic features/targets must be non-empty and aligned")
    if int((y > 0.5).sum()) == 0 or int((y <= 0.5).sum()) == 0:
        raise ValueError("logistic calibration requires both classes")
    weights = (
        torch.ones_like(y)
        if sample_weight is None
        else sample_weight.detach().float().flatten().cpu().clamp_min(0.0)
    )
    if weights.numel() != y.numel() or float(weights.sum()) <= 0.0:
        raise ValueError("invalid logistic sample weights")

    mean, std = _weighted_moments(x, weights)
    z = (x - mean) / std
    objective_weights = weights.clone()
    if balance_classes:
        positive_mass = weights[y > 0.5].sum().clamp_min(1e-8)
        negative_mass = weights[y <= 0.5].sum().clamp_min(1e-8)
        objective_weights[y > 0.5] *= 0.5 / positive_mass
        objective_weights[y <= 0.5] *= 0.5 / negative_mass
    objective_weights /= objective_weights.sum().clamp_min(1e-8)

    coefficient = torch.zeros(z.shape[1], requires_grad=True)
    intercept = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (coefficient, intercept),
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = z @ coefficient + intercept
        data_loss = (
            F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            * objective_weights
        ).sum()
        loss = data_loss + 0.5 * float(l2) * coefficient.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return {
            "mean": mean,
            "std": std,
            "coefficient": coefficient.detach(),
            "intercept": intercept.detach(),
            "l2": float(l2),
            "max_iter": int(max_iter),
            "balance_classes": bool(balance_classes),
        }


def cross_fitted_logistic_scores(
    features: torch.Tensor,
    target: torch.Tensor,
    events: Sequence[str],
    *,
    sample_weight: torch.Tensor | None = None,
    l2: float = 1e-2,
    max_iter: int = 200,
    balance_classes: bool = False,
) -> Dict[str, object]:
    """Return leave-one-event-out scores with no in-fold fitted predictions."""

    x = features.detach().float().cpu()
    y = target.detach().bool().flatten().cpu()
    if x.ndim != 2 or x.shape[0] != y.numel() or len(events) != y.numel():
        raise ValueError("cross-fitting inputs are not aligned")
    weights = (
        None
        if sample_weight is None
        else sample_weight.detach().float().flatten().cpu()
    )
    if weights is not None and weights.numel() != y.numel():
        raise ValueError("cross-fitting weights are not aligned")

    event_values = list(map(str, events))
    scores = torch.full((y.numel(),), float("nan"), dtype=torch.float32)
    folds: list[Dict[str, object]] = []
    for held_out in sorted(set(event_values)):
        validation_mask = torch.tensor(
            [event == held_out for event in event_values], dtype=torch.bool
        )
        training_mask = ~validation_mask
        if int(validation_mask.sum()) == 0:
            continue
        training_target = y[training_mask]
        if int(training_target.sum()) == 0 or int((~training_target).sum()) == 0:
            raise RuntimeError(
                f"IM1-D fold {held_out!r} lacks both rescue and harm training cases"
            )
        model = fit_logistic_calibrator(
            x[training_mask], training_target,
            sample_weight=(weights[training_mask] if weights is not None else None),
            l2=l2,
            max_iter=max_iter,
            balance_classes=balance_classes,
        )
        scores[validation_mask] = apply_logistic_calibrator(
            x[validation_mask], model
        )
        folds.append({
            "held_out_event": held_out,
            "train_count": int(training_mask.sum()),
            "held_out_count": int(validation_mask.sum()),
        })
    if not torch.isfinite(scores).all():
        raise RuntimeError("IM1-D cross-fitting left non-finite proposal scores")
    return {"scores": scores, "folds": folds}


def apply_logistic_calibrator(
    features: torch.Tensor, model: Mapping[str, object]
) -> torch.Tensor:
    x = features.detach().float().cpu()
    mean = torch.as_tensor(model["mean"], dtype=torch.float32)
    std = torch.as_tensor(model["std"], dtype=torch.float32).clamp_min(1e-8)
    coefficient = torch.as_tensor(model["coefficient"], dtype=torch.float32)
    intercept = torch.as_tensor(model["intercept"], dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != coefficient.numel():
        raise ValueError("calibrator feature dimension mismatch")
    return torch.sigmoid(((x - mean) / std) @ coefficient + intercept)


def event_macro_auroc(
    scores: torch.Tensor, target: torch.Tensor, events: Sequence[str]
) -> Dict[str, object]:
    scores = scores.float().flatten().cpu()
    target = target.bool().flatten().cpu()
    if scores.numel() != target.numel() or len(events) != target.numel():
        raise ValueError("event AUROC inputs are not aligned")
    values: Dict[str, float] = {}
    for event in sorted(set(map(str, events))):
        mask = torch.tensor([str(item) == event for item in events], dtype=torch.bool)
        if int(target[mask].sum()) == 0 or int((~target[mask]).sum()) == 0:
            continue
        value = binary_auroc(scores[mask], target[mask])
        if math.isfinite(value):
            values[event] = float(value)
    return {
        "event_macro_auroc": (
            sum(values.values()) / len(values) if values else float("nan")
        ),
        "comparable_events": len(values),
        "event_auroc": values,
    }


def risk_coverage_table(
    scores: torch.Tensor,
    target: torch.Tensor,
    events: Sequence[str],
    *,
    coverages: Sequence[float],
) -> list[Dict[str, object]]:
    scores = scores.float().flatten().cpu()
    target = target.bool().flatten().cpu()
    if scores.numel() != target.numel() or len(events) != target.numel():
        raise ValueError("risk-coverage inputs are not aligned")
    order = torch.argsort(scores, descending=True, stable=True)
    all_events = sorted(set(map(str, events)))
    rows: list[Dict[str, object]] = []
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError("coverage must be in (0,1]")
        take = max(1, min(order.numel(), int(math.ceil(float(coverage) * order.numel()))))
        chosen = order[:take]
        row = _summarise_selection(chosen, target, events, all_events)
        row.update({
            "coverage": float(coverage),
            "score_threshold": float(scores[chosen[-1]]),
        })
        rows.append(row)
    return rows


def _summarise_selection(
    chosen: torch.Tensor,
    target: torch.Tensor,
    events: Sequence[str],
    all_events: Sequence[str],
) -> Dict[str, object]:
    rescued = int(target[chosen].sum()) if chosen.numel() else 0
    harmed = int((~target[chosen]).sum()) if chosen.numel() else 0
    event_net = {str(event): 0 for event in all_events}
    for local_index in chosen.tolist():
        event = str(events[int(local_index)])
        event_net[event] += 1 if bool(target[int(local_index)]) else -1
    return {
        "selected": int(chosen.numel()),
        "rescued": rescued,
        "harmed": harmed,
        "net_correction": rescued - harmed,
        "precision": rescued / max(int(chosen.numel()), 1),
        "nonnegative_events": sum(value >= 0 for value in event_net.values()),
        "event_net_correction": event_net,
    }


def operating_point_at_threshold(
    scores: torch.Tensor,
    target: torch.Tensor,
    events: Sequence[str],
    *,
    threshold: float,
) -> Dict[str, object]:
    """Apply a train-selected probability threshold without re-ranking val."""

    scores = scores.float().flatten().cpu()
    target = target.bool().flatten().cpu()
    if scores.numel() != target.numel() or len(events) != target.numel():
        raise ValueError("threshold operating-point inputs are not aligned")
    chosen = torch.nonzero(scores >= float(threshold), as_tuple=False).flatten()
    row = _summarise_selection(
        chosen, target, events, sorted(set(map(str, events)))
    )
    row.update({
        "coverage": int(chosen.numel()) / max(int(scores.numel()), 1),
        "score_threshold": float(threshold),
    })
    return row


def select_train_operating_point(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    admissible = [
        row for row in rows
        if float(row["precision"]) >= 0.55
        and int(row["net_correction"]) > 0
        and int(row["rescued"]) >= 10
        and int(row["nonnegative_events"]) >= 10
    ]
    return (
        max(admissible, key=lambda row: (int(row["net_correction"]), -float(row["coverage"])))
        if admissible else None
    )


def serialise_calibrator(model: Mapping[str, object]) -> Dict[str, object]:
    return {
        "mean": torch.as_tensor(model["mean"]).tolist(),
        "std": torch.as_tensor(model["std"]).tolist(),
        "coefficient": torch.as_tensor(model["coefficient"]).tolist(),
        "intercept": float(torch.as_tensor(model["intercept"])),
        "l2": float(model["l2"]),
        "max_iter": int(model["max_iter"]),
        "balance_classes": bool(model.get("balance_classes", True)),
    }
