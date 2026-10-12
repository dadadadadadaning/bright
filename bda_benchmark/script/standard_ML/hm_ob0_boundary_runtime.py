# -*- coding: utf-8 -*-
"""Pure utilities for HM-OB0 boundary-conditional Occurrence auditing.

OB0 never changes P3.  It restricts evaluation to building regions that the
frozen P3 predicts as Intact and asks whether the frozen old-P3 Mean key adds
cross-event Affected evidence beyond P3's own probability.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Dict

import torch

from script.standard_ML.hm_occurrence_runtime import (
    binary_centre_scores,
    event_class_capped_indices,
    fit_binary_centres,
    occurrence_auc_report,
)
from script.standard_ML.im1_matching_reliability_runtime import (
    apply_logistic_calibrator,
    fit_logistic_calibrator,
)


MODEL_NAMES = ("ob0_a_p3", "ob0_b_old_mean", "ob0_c_combined")


def boundary_candidate_table(cache: Mapping[str, object]) -> Dict[str, object]:
    """Return the diagnostic P3-Intact candidate pool from a P0 cache."""

    required = (
        "labels",
        "base_predictions",
        "base_probabilities",
        "events",
        "key_variants",
    )
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"OB0 P0 cache is missing: {missing}")

    truth = torch.as_tensor(cache["labels"]).long().flatten().cpu()
    base = torch.as_tensor(cache["base_predictions"]).long().flatten().cpu()
    probability = torch.as_tensor(cache["base_probabilities"]).float().cpu()
    mean = torch.as_tensor(cache["key_variants"]["mean"]).float().cpu()
    events = list(map(str, cache["events"]))
    count = truth.numel()
    if probability.shape != (count, 3):
        raise ValueError("OB0 expects conditional P3 probabilities [N,3]")
    if base.numel() != count or mean.shape[0] != count or len(events) != count:
        raise ValueError("OB0 P0 rows are not aligned")
    if not set(torch.unique(truth).tolist()).issubset({0, 1, 2}):
        raise ValueError("OB0 expects Intact/Damaged/Destroyed labels 0/1/2")

    candidate_mask = base == 0
    indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("OB0 found no P3-Intact candidate regions")
    candidate_truth = truth[indices]
    occurrence_target = candidate_truth > 0
    if int(occurrence_target.sum()) == 0 or int((~occurrence_target).sum()) == 0:
        raise RuntimeError("OB0 requires both missed Affected and true Intact candidates")

    return {
        "indices": indices,
        "truth": candidate_truth,
        "occurrence_target": occurrence_target,
        "events": [events[int(index)] for index in indices],
        "p3_affected_probability": probability[indices, 1:3].sum(dim=1),
        "mean_features": mean,
        "all_occurrence_labels": truth > 0,
        "all_events": events,
        "candidate_count": int(indices.numel()),
        "missed_damaged": int((candidate_truth == 1).sum()),
        "missed_destroyed": int((candidate_truth == 2).sum()),
        "true_intact": int((candidate_truth == 0).sum()),
    }


def event_codes(
    train_names: Sequence[str], query_names: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    names = sorted(set(map(str, train_names)))
    if not set(map(str, query_names)).issubset(names):
        raise RuntimeError("OB0 Validation contains an event absent from Train")
    mapping = {name: index for index, name in enumerate(names)}
    return (
        torch.tensor([mapping[str(name)] for name in train_names], dtype=torch.long),
        torch.tensor([mapping[str(name)] for name in query_names], dtype=torch.long),
        mapping,
    )


def strict_event_crossfit_scores(
    table: Mapping[str, object],
    event_tensor: torch.Tensor,
    *,
    cap_per_event_class: int,
    seed: int,
    logistic_l2: float,
    logistic_max_iter: int,
) -> Dict[str, object]:
    """Produce Train scores with every query event excluded from all fitting.

    For a held-out event, binary centres and the tiny Combined calibrator are
    both fitted without any row from that event.  This is stricter than first
    building a global score table and cross-fitting only the calibrator.
    """

    features = torch.as_tensor(table["mean_features"]).float().cpu()
    all_labels = torch.as_tensor(table["all_occurrence_labels"]).long().cpu()
    candidate_indices = torch.as_tensor(table["indices"]).long().cpu()
    candidate_target = torch.as_tensor(table["occurrence_target"]).bool().cpu()
    p3_score = torch.as_tensor(table["p3_affected_probability"]).float().cpu()
    candidate_events = event_tensor[candidate_indices]
    if features.shape[0] != all_labels.numel() or all_labels.numel() != event_tensor.numel():
        raise ValueError("OB0 cross-fit full rows are not aligned")

    mean_score = torch.full_like(p3_score, float("nan"))
    combined_score = torch.full_like(p3_score, float("nan"))
    folds = []
    for held_event in sorted(int(value) for value in torch.unique(candidate_events).tolist()):
        query_mask = candidate_events == held_event
        training_mask = candidate_events != held_event
        if int(query_mask.sum()) == 0:
            continue
        train_target = candidate_target[training_mask]
        if int(train_target.sum()) == 0 or int((~train_target).sum()) == 0:
            raise RuntimeError(f"OB0 held event {held_event} leaves one boundary class empty")

        fit_indices = event_class_capped_indices(
            all_labels,
            event_tensor,
            cap_per_event_class=cap_per_event_class,
            seed=seed,
            excluded_event=held_event,
        )
        centres = fit_binary_centres(features, all_labels, fit_indices)
        train_mean = binary_centre_scores(
            features[candidate_indices[training_mask]], centres
        )
        query_mean = binary_centre_scores(
            features[candidate_indices[query_mask]], centres
        )
        calibrator = fit_logistic_calibrator(
            torch.stack((p3_score[training_mask], train_mean), dim=1),
            train_target,
            l2=logistic_l2,
            max_iter=logistic_max_iter,
            balance_classes=False,
        )
        mean_score[query_mask] = query_mean
        combined_score[query_mask] = apply_logistic_calibrator(
            torch.stack((p3_score[query_mask], query_mean), dim=1), calibrator
        )
        folds.append({
            "held_out_event": int(held_event),
            "fit_centre_rows": int(fit_indices.numel()),
            "calibrator_train_candidates": int(training_mask.sum()),
            "held_out_candidates": int(query_mask.sum()),
        })

    if not torch.isfinite(mean_score).all() or not torch.isfinite(combined_score).all():
        raise RuntimeError("OB0 strict cross-fitting left non-finite scores")
    return {
        "scores": {
            "ob0_a_p3": p3_score,
            "ob0_b_old_mean": mean_score,
            "ob0_c_combined": combined_score,
        },
        "folds": folds,
    }


def fit_final_models(
    table: Mapping[str, object],
    event_tensor: torch.Tensor,
    train_oof_mean_score: torch.Tensor,
    *,
    cap_per_event_class: int,
    seed: int,
    logistic_l2: float,
    logistic_max_iter: int,
) -> Dict[str, object]:
    """Fit frozen Train-only centres and the final two-signal calibrator."""

    labels = torch.as_tensor(table["all_occurrence_labels"]).long().cpu()
    features = torch.as_tensor(table["mean_features"]).float().cpu()
    fit_indices = event_class_capped_indices(
        labels,
        event_tensor,
        cap_per_event_class=cap_per_event_class,
        seed=seed,
    )
    centres = fit_binary_centres(features, labels, fit_indices)
    calibrator = fit_logistic_calibrator(
        torch.stack((
            torch.as_tensor(table["p3_affected_probability"]).float().cpu(),
            train_oof_mean_score.float().cpu(),
        ), dim=1),
        torch.as_tensor(table["occurrence_target"]).bool().cpu(),
        l2=logistic_l2,
        max_iter=logistic_max_iter,
        balance_classes=False,
    )
    return {
        "centres": centres,
        "calibrator": calibrator,
        "fit_centre_rows": int(fit_indices.numel()),
    }


def apply_final_models(
    table: Mapping[str, object], final_models: Mapping[str, object]
) -> Dict[str, torch.Tensor]:
    indices = torch.as_tensor(table["indices"]).long().cpu()
    p3_score = torch.as_tensor(table["p3_affected_probability"]).float().cpu()
    mean_score = binary_centre_scores(
        torch.as_tensor(table["mean_features"]).float().cpu()[indices],
        torch.as_tensor(final_models["centres"]).float().cpu(),
    )
    combined_score = apply_logistic_calibrator(
        torch.stack((p3_score, mean_score), dim=1), final_models["calibrator"]
    )
    return {
        "ob0_a_p3": p3_score,
        "ob0_b_old_mean": mean_score,
        "ob0_c_combined": combined_score,
    }


def _selection_report(
    chosen: torch.Tensor,
    truth: torch.Tensor,
    events: Sequence[str],
) -> Dict[str, object]:
    truth = truth.long().flatten().cpu()
    chosen = chosen.long().flatten().cpu()
    all_events = sorted(set(map(str, events)))
    event_net = {event: 0 for event in all_events}
    for index in chosen.tolist():
        event = str(events[int(index)])
        event_net[event] += 1 if int(truth[int(index)]) > 0 else -1
    damaged = int((truth[chosen] == 1).sum()) if chosen.numel() else 0
    destroyed = int((truth[chosen] == 2).sum()) if chosen.numel() else 0
    harmed = int((truth[chosen] == 0).sum()) if chosen.numel() else 0
    rescued = damaged + destroyed
    return {
        "selected": int(chosen.numel()),
        "rescued": rescued,
        "harmed": harmed,
        "net_correction": rescued - harmed,
        "precision": rescued / max(int(chosen.numel()), 1),
        "damaged_rescued": damaged,
        "destroyed_rescued": destroyed,
        "intact_harmed": harmed,
        "nonnegative_events": sum(value >= 0 for value in event_net.values()),
        "positive_events": sum(value > 0 for value in event_net.values()),
        "event_net_correction": event_net,
    }


def risk_coverage_table(
    scores: torch.Tensor,
    truth: torch.Tensor,
    events: Sequence[str],
    *,
    coverages: Sequence[float],
) -> list[Dict[str, object]]:
    scores = scores.float().flatten().cpu()
    truth = truth.long().flatten().cpu()
    if scores.numel() != truth.numel() or len(events) != truth.numel():
        raise ValueError("OB0 risk-coverage rows are not aligned")
    order = torch.argsort(scores, descending=True, stable=True)
    rows = []
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError("OB0 coverage must be in (0,1]")
        take = max(1, min(order.numel(), int(math.ceil(float(coverage) * order.numel()))))
        chosen = order[:take]
        row = _selection_report(chosen, truth, events)
        row.update({
            "coverage": float(coverage),
            "score_threshold": float(scores[chosen[-1]]),
        })
        rows.append(row)
    return rows


def select_train_operating_point(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_precision: float,
    minimum_rescued: int,
    minimum_nonnegative_events: int,
    minimum_positive_events: int,
) -> Mapping[str, object] | None:
    admissible = [
        row for row in rows
        if float(row["precision"]) >= float(minimum_precision)
        and int(row["net_correction"]) > 0
        and int(row["rescued"]) >= int(minimum_rescued)
        and int(row["nonnegative_events"]) >= int(minimum_nonnegative_events)
        and int(row["positive_events"]) >= int(minimum_positive_events)
    ]
    return (
        max(
            admissible,
            key=lambda row: (
                int(row["net_correction"]),
                float(row["precision"]),
                -float(row["coverage"]),
            ),
        )
        if admissible else None
    )


def locked_point_at_threshold(
    scores: torch.Tensor,
    truth: torch.Tensor,
    events: Sequence[str],
    *,
    threshold: float,
) -> Dict[str, object]:
    scores = scores.float().flatten().cpu()
    chosen = torch.nonzero(scores >= float(threshold), as_tuple=False).flatten()
    row = _selection_report(chosen, truth, events)
    row.update({
        "coverage": int(chosen.numel()) / max(int(scores.numel()), 1),
        "score_threshold": float(threshold),
    })
    return row


def operating_point_passes(
    row: Mapping[str, object] | None,
    *,
    minimum_precision: float,
    minimum_rescued: int,
    minimum_nonnegative_events: int,
    minimum_positive_events: int,
) -> bool:
    return bool(
        row is not None
        and float(row["precision"]) >= float(minimum_precision)
        and int(row["net_correction"]) > 0
        and int(row["rescued"]) >= int(minimum_rescued)
        and int(row["nonnegative_events"]) >= int(minimum_nonnegative_events)
        and int(row["positive_events"]) >= int(minimum_positive_events)
    )


def model_report(
    scores: torch.Tensor,
    table: Mapping[str, object],
    *,
    coverages: Sequence[float],
) -> Dict[str, object]:
    target = torch.as_tensor(table["occurrence_target"]).bool().cpu()
    truth = torch.as_tensor(table["truth"]).long().cpu()
    events = list(map(str, table["events"]))
    return {
        "auc": occurrence_auc_report(scores, target.long(), events),
        "risk_coverage": risk_coverage_table(
            scores, truth, events, coverages=coverages
        ),
    }


__all__ = [
    "MODEL_NAMES",
    "apply_final_models",
    "boundary_candidate_table",
    "event_codes",
    "fit_final_models",
    "locked_point_at_threshold",
    "model_report",
    "operating_point_passes",
    "risk_coverage_table",
    "select_train_operating_point",
    "strict_event_crossfit_scores",
]
