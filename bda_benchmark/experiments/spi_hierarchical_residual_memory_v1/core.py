from __future__ import annotations

import math
import torch

from bda_benchmark.experiments.spi_complete_building_memory_v1.core import binary_auroc
from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import (
    FactorizedCaseMetric, Retrieval, correction_direction, retrieval_weights,
    row_events,
)


TARGET_NAMES = ("p3_source", "gt_class", "affected", "p3_error", "error_family", "full_direction")


def granularity_targets(cache: dict) -> dict[str, torch.Tensor]:
    source = cache["p3_predictions"].long(); target = cache["labels"].long()
    error = source != target
    family = torch.zeros_like(source)
    family[(source <= 1) & (target >= 2)] = 1       # missed affected
    family[(source >= 2) & (target <= 1)] = 2       # false affected
    family[(source >= 2) & (target >= 2) & error] = 3  # severity confusion
    family[error & (family == 0)] = 4               # background/intact or other
    return {
        "p3_source": source,
        "gt_class": target,
        "affected": (target >= 2).long(),
        "p3_error": error.long(),
        "error_family": family,
        "full_direction": correction_direction(cache),
    }


def build_hierarchical_pools(cache: dict, minimum_purity: float = .60):
    target = granularity_targets(cache); events = row_events(cache)
    valid = cache["label_purity"].float() >= minimum_purity
    pools: dict[tuple, list[int]] = {}
    for i in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        source = int(target["p3_source"][i]); error = int(target["p3_error"][i]); family = int(target["error_family"][i])
        pools.setdefault(("source", source), []).append(i)
        pools.setdefault(("state", source, error), []).append(i)
        pools.setdefault(("family", source, family), []).append(i)
    return {"targets": target, "events": events, "valid": valid, "pools": pools}


def _draw_pair(kind: str, state: dict, generator: torch.Generator, maximum_attempts: int = 1000):
    target, events, pools = state["targets"], state["events"], state["pools"]
    if kind in {"error_positive", "error_hard_negative"}: anchors = torch.nonzero(state["valid"] & target["p3_error"].bool(), as_tuple=False).flatten()
    elif kind == "correct_positive": anchors = torch.nonzero(state["valid"] & ~target["p3_error"].bool(), as_tuple=False).flatten()
    else: anchors = torch.nonzero(state["valid"], as_tuple=False).flatten()
    for _ in range(maximum_attempts):
        a = int(anchors[torch.randint(len(anchors), (1,), generator=generator)])
        source = int(target["p3_source"][a]); family = int(target["error_family"][a]); error = int(target["p3_error"][a])
        if kind == "error_positive": candidates, label = pools.get(("family", source, family), []), 1.
        elif kind == "error_hard_negative": candidates, label = pools.get(("state", source, 1), []), 0.
        elif kind == "correct_positive": candidates, label = pools.get(("state", source, 0), []), 1.
        else: candidates, label = pools.get(("state", source, 1 - error), []), 0.
        if not candidates: continue
        b = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
        if a == b or events[a] == events[b]: continue
        if kind == "error_hard_negative" and int(target["error_family"][b]) == family: continue
        return a, b, label
    # Some P3 sources have only one error family.  Preserve a hard, same-source
    # cross-event negative by contrasting an error with a correct case.
    if kind == "error_hard_negative":
        return _draw_pair("state_negative", state, generator, maximum_attempts)
    raise RuntimeError(f"cannot sample cross-event pair for {kind}")


def sample_hierarchical_pairs(cache: dict, count: int, generator: torch.Generator, state=None):
    state = build_hierarchical_pools(cache) if state is None else state
    quotas = {
        "error_positive": int(round(count * .40)),
        "error_hard_negative": int(round(count * .30)),
        "correct_positive": int(round(count * .20)),
    }
    quotas["state_negative"] = count - sum(quotas.values())
    left, right, labels, kinds = [], [], [], []
    for kind, quota in quotas.items():
        for _ in range(quota):
            a, b, label = _draw_pair(kind, state, generator)
            left.append(a); right.append(b); labels.append(label); kinds.append(kind)
    order = torch.randperm(count, generator=generator)
    return (torch.tensor(left)[order], torch.tensor(right)[order],
            torch.tensor(labels)[order], [kinds[i] for i in order.tolist()])


def weighted_neighbour_distribution(retrieval: Retrieval, memory_target: torch.Tensor, classes: int):
    weight, _ = retrieval_weights(retrieval)
    safe = retrieval.indices.clamp_min(0)
    one_hot = torch.nn.functional.one_hot(memory_target.long(), classes).float()
    return (one_hot[safe] * weight[..., None]).sum(1)


def categorical_retrieval_report(retrieval: Retrieval, query_target: torch.Tensor,
                                 memory_target: torch.Tensor, classes: int,
                                 query_events: list[str], mask: torch.Tensor | None = None):
    probability = weighted_neighbour_distribution(retrieval, memory_target, classes)
    valid = retrieval.coverage.clone()
    if mask is not None: valid &= mask.bool()
    safe = retrieval.indices.clamp_min(0); neighbour_valid = retrieval.indices >= 0
    purity = (((memory_target[safe] == query_target[:, None]) & neighbour_valid).sum(1).float()
              / neighbour_valid.sum(1).clamp_min(1))
    aucs = [binary_auroc((query_target[valid] == c).long(), probability[valid, c]) for c in range(classes)]
    aucs_valid = [v for v in aucs if v == v]
    event_aucs = {}
    for event in sorted(set(query_events)):
        event_mask = valid & torch.tensor([x == event for x in query_events])
        values = [binary_auroc((query_target[event_mask] == c).long(), probability[event_mask, c]) for c in range(classes)]
        values = [v for v in values if v == v]
        if values: event_aucs[event] = sum(values) / len(values)
    return {
        "queries": int(valid.sum()), "coverage": float(valid.float().mean()),
        "neighbour_purity": float(purity[valid].mean()) if valid.any() else 0.,
        "macro_ovr_auroc": sum(aucs_valid) / max(len(aucs_valid), 1),
        "event_macro_auroc": sum(event_aucs.values()) / max(len(event_aucs), 1),
        "correct_events": sum(v > .5 for v in event_aucs.values()),
        "evaluable_events": len(event_aucs), "events": event_aucs,
    }


def mean_retrieval_similarity(retrieval: Retrieval):
    weight, _ = retrieval_weights(retrieval)
    valid = retrieval.indices >= 0
    return (retrieval.scores.masked_fill(~valid, 0) * weight).sum(1)


def binary_event_report(target: torch.Tensor, score: torch.Tensor, events: list[str], coverage: torch.Tensor):
    auc = binary_auroc(target[coverage], score[coverage]); event_auc = {}
    for event in sorted(set(events)):
        mask = coverage & torch.tensor([x == event for x in events])
        value = binary_auroc(target[mask], score[mask])
        if value == value: event_auc[event] = value
    return {
        "auroc": auc, "event_macro_auroc": sum(event_auc.values()) / max(len(event_auc), 1),
        "correct_events": sum(v > .5 for v in event_auc.values()), "evaluable_events": len(event_auc),
        "coverage": float(coverage.float().mean()), "events": event_auc,
    }


def dual_bank_error_score(correct: Retrieval, residual: Retrieval, temperature: float = .10):
    difference = mean_retrieval_similarity(residual) - mean_retrieval_similarity(correct)
    score = torch.sigmoid(difference / max(temperature, 1e-6))
    return score, difference


__all__ = ["FactorizedCaseMetric", "TARGET_NAMES", "granularity_targets",
           "build_hierarchical_pools", "sample_hierarchical_pairs",
           "categorical_retrieval_report", "binary_event_report", "dual_bank_error_score"]
