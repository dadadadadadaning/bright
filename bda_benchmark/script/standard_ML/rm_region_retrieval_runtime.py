# -*- coding: utf-8 -*-
"""Pure utilities for the RM0--RM4 train-only region-memory audit.

The module deliberately contains no learnable classifier.  It makes the
retrieval hypothesis falsifiable before a Transformer or retrieval residual is
allowed to affect P3 predictions.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


NUM_CLASSES = 3  # 0=Intact, 1=Damaged, 2=Destroyed


def event_from_name(name: str) -> str:
    """Remove the terminal tile index while preserving the disaster event."""

    stem = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"_\d+$", "", stem)


def validate_disjoint_ids(memory_ids: Sequence[str], query_ids: Sequence[str]) -> None:
    overlap = sorted(set(map(str, memory_ids)).intersection(map(str, query_ids)))
    if overlap:
        raise RuntimeError(
            "train-memory/query image overlap detected; first overlaps="
            + ", ".join(overlap[:5])
        )


def balanced_indices(
    labels: torch.Tensor, capacity_per_class: int, seed: int
) -> torch.Tensor:
    """Select an equal, deterministic number of items from every class."""

    labels = labels.long().cpu()
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty vector")
    counts = [int((labels == cls).sum()) for cls in range(NUM_CLASSES)]
    if min(counts) == 0:
        raise RuntimeError(f"cannot build class-balanced memory: counts={counts}")
    target = min(int(capacity_per_class), min(counts))
    if target <= 0:
        raise ValueError("capacity_per_class must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = []
    for cls in range(NUM_CLASSES):
        indices = torch.nonzero(labels == cls, as_tuple=False).flatten()
        order = torch.randperm(indices.numel(), generator=generator)[:target]
        selected.append(indices[order])
    return torch.cat(selected, dim=0)


def class_event_balanced_indices(
    labels: torch.Tensor,
    events: Sequence[str],
    capacity_per_class: int,
    capacity_per_event_class: int,
    seed: int,
) -> torch.Tensor:
    """Build a deterministic class-balanced memory with an event cap.

    Selection is round-robin over events inside every class.  Consequently an
    event with many eligible regions cannot fill the class quota before rare
    events have contributed their available examples.  The method never
    fabricates examples: absent class/event cells remain absent and are exposed
    by the cache manifest.
    """

    labels = labels.long().cpu()
    events = list(map(str, events))
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty vector")
    if len(events) != labels.numel():
        raise ValueError("event count must equal label count")
    if int(capacity_per_class) <= 0 or int(capacity_per_event_class) <= 0:
        raise ValueError("memory capacities must be positive")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for cls in range(NUM_CLASSES):
        class_mask = labels == cls
        class_events = sorted({events[i] for i in torch.nonzero(class_mask).flatten()})
        if not class_events:
            raise RuntimeError(f"class {cls} has no memory candidates")
        queues: Dict[str, torch.Tensor] = {}
        for event in class_events:
            indices = torch.tensor(
                [i for i, item in enumerate(events) if item == event and bool(class_mask[i])],
                dtype=torch.long,
            )
            order = torch.randperm(indices.numel(), generator=generator)
            queues[event] = indices[order[: int(capacity_per_event_class)]]

        # Round-robin rather than concatenation prevents early events from
        # monopolising the global per-class capacity.
        class_selected: list[torch.Tensor] = []
        depth = 0
        while len(class_selected) < int(capacity_per_class):
            added = False
            for event in class_events:
                queue = queues[event]
                if depth < queue.numel():
                    class_selected.append(queue[depth])
                    added = True
                    if len(class_selected) == int(capacity_per_class):
                        break
            if not added:
                break
            depth += 1
        if not class_selected:
            raise RuntimeError(f"class {cls} has no selected memory entries")
        selected.append(torch.stack(class_selected))
    target = min(int(capacity_per_class), min(item.numel() for item in selected))
    if target <= 0:
        raise RuntimeError("class-event balancing produced an empty class")
    return torch.cat([item[:target] for item in selected], dim=0)


def class_event_count_table(
    labels: torch.Tensor, events: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    labels = labels.long().cpu()
    if labels.numel() != len(events):
        raise ValueError("event count must equal label count")
    names = ("intact", "damaged", "destroyed")
    return {
        names[cls]: {
            event: int(
                sum(
                    int(labels[index]) == cls and str(item) == event
                    for index, item in enumerate(events)
                )
            )
            for event in sorted(set(map(str, events)))
        }
        for cls in range(NUM_CLASSES)
    }


def scene_relative_raw_stats(
    region_stats: torch.Tensor, scene_stats: torch.Tensor
) -> torch.Tensor:
    """Remove tile-level SAR location/scale from region mean/std.

    Output dimensions are the region mean offset in scene-standard-deviation
    units and region_std / scene_std.  ``raw_stat_normalizer`` applies the
    logarithm to the positive ratio later.  These retain local contrast while
    suppressing absolute acquisition brightness and scale.
    """

    if region_stats.ndim != 2 or region_stats.shape[1] != 2:
        raise ValueError("region_stats must be [N,2]")
    if scene_stats.shape != region_stats.shape:
        raise ValueError("scene_stats must match region_stats")
    scene_std = scene_stats[:, 1].float().clamp_min(1e-6)
    relative_mean = (region_stats[:, 0].float() - scene_stats[:, 0].float()) / scene_std
    relative_std_ratio = region_stats[:, 1].float().clamp_min(1e-6) / scene_std
    return torch.stack((relative_mean, relative_std_ratio), dim=1)


def raw_stat_normalizer(raw_stats: torch.Tensor) -> Dict[str, float]:
    if raw_stats.ndim != 2 or raw_stats.shape[1] != 2:
        raise ValueError("raw_stats must be [N,2] containing mean/std")
    transformed = torch.stack(
        (raw_stats[:, 0].float(), raw_stats[:, 1].float().clamp_min(1e-6).log()),
        dim=1,
    )
    mean = transformed.mean(dim=0)
    std = transformed.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "sar_mean_center": float(mean[0]),
        "sar_mean_scale": float(std[0]),
        "sar_logstd_center": float(mean[1]),
        "sar_logstd_scale": float(std[1]),
    }


def build_combined_keys(
    visual: torch.Tensor,
    raw_stats: torch.Tensor,
    normalizer: Mapping[str, float],
    raw_weight: float,
) -> torch.Tensor:
    visual = F.normalize(visual.float(), dim=1)
    if float(raw_weight) == 0.0:
        return visual
    z_mean = (raw_stats[:, 0].float() - float(normalizer["sar_mean_center"])) / float(
        normalizer["sar_mean_scale"]
    )
    z_std = (
        raw_stats[:, 1].float().clamp_min(1e-6).log()
        - float(normalizer["sar_logstd_center"])
    ) / float(normalizer["sar_logstd_scale"])
    raw = torch.stack((z_mean, z_std), dim=1)
    return F.normalize(torch.cat((visual, float(raw_weight) * raw), dim=1), dim=1)


def build_raw_only_keys(
    raw_stats: torch.Tensor,
    normalizer: Mapping[str, float],
) -> torch.Tensor:
    """Build a separate scene-relative SAR key instead of hiding it in concat.

    RM7 deliberately retrieves semantic and SAR evidence independently.  This
    prevents two low-dimensional acquisition statistics from silently rotating
    the much larger P3 semantic key and makes cross-modal agreement auditable.
    """

    if raw_stats.ndim != 2 or raw_stats.shape[1] != 2:
        raise ValueError("raw_stats must be [N,2]")
    z_mean = (
        raw_stats[:, 0].float() - float(normalizer["sar_mean_center"])
    ) / float(normalizer["sar_mean_scale"])
    z_std = (
        raw_stats[:, 1].float().clamp_min(1e-6).log()
        - float(normalizer["sar_logstd_center"])
    ) / float(normalizer["sar_logstd_scale"])
    return F.normalize(torch.stack((z_mean, z_std), dim=1), dim=1)


def dual_key_reject_signals(
    base_probabilities: torch.Tensor,
    visual_retrieval: Mapping[str, torch.Tensor],
    sar_retrieval: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return label-free signals for a conservative Intact->Damaged rejector."""

    base = base_probabilities.float().cpu()
    visual_scores = visual_retrieval["scores"].float().cpu()
    sar_scores = sar_retrieval["scores"].float().cpu()
    if base.ndim != 2 or base.shape[1] != NUM_CLASSES:
        raise ValueError("base_probabilities must be [N,3]")
    if visual_scores.shape != base.shape or sar_scores.shape != base.shape:
        raise ValueError("dual-key retrieval scores must match base probabilities")
    eps = 1e-8
    visual_llr = (
        visual_scores[:, 1].clamp_min(eps).log()
        - visual_scores[:, 0].clamp_min(eps).log()
    )
    sar_llr = (
        sar_scores[:, 1].clamp_min(eps).log()
        - sar_scores[:, 0].clamp_min(eps).log()
    )
    sorted_base = base.sort(dim=1, descending=True).values
    p3_uncertainty = 1.0 - (sorted_base[:, 0] - sorted_base[:, 1])
    visual_best = visual_retrieval["best_similarity_by_class"].float().cpu()
    sar_best = sar_retrieval["best_similarity_by_class"].float().cpu()
    damaged_support = torch.minimum(visual_best[:, 1], sar_best[:, 1])
    visual_events = visual_retrieval["unique_neighbor_events_by_class"].float().cpu()
    sar_events = sar_retrieval["unique_neighbor_events_by_class"].float().cpu()
    damaged_event_consensus = torch.minimum(visual_events[:, 1], sar_events[:, 1])
    return {
        "visual_damaged_vs_intact_llr": visual_llr,
        "sar_damaged_vs_intact_llr": sar_llr,
        "p3_uncertainty": p3_uncertainty,
        "damaged_support": damaged_support,
        "damaged_event_consensus": damaged_event_consensus,
        "agreement_score": torch.minimum(visual_llr, sar_llr),
    }


def apply_dual_key_reject_rule(
    base_prediction: torch.Tensor,
    signals: Mapping[str, torch.Tensor],
    rule: Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    """Accept Memory only when every independently declared safety gate passes.

    A missing/admissibly empty rule rejects every proposal and exactly preserves
    P3.  The function never uses labels and is therefore safe on validation.
    """

    base = base_prediction.long().flatten().cpu()
    if not bool(rule.get("enabled", False)):
        return {
            "prediction": base.clone(),
            "accepted": torch.zeros_like(base, dtype=torch.bool),
        }
    required = {
        "visual_llr_threshold",
        "sar_llr_threshold",
        "p3_uncertainty_threshold",
        "damaged_support_threshold",
        "minimum_damaged_events",
    }
    missing = sorted(required.difference(rule))
    if missing:
        raise KeyError(f"dual-key reject rule is missing: {missing}")
    accepted = (
        (base == 0)
        & (signals["visual_damaged_vs_intact_llr"] >= float(rule["visual_llr_threshold"]))
        & (signals["sar_damaged_vs_intact_llr"] >= float(rule["sar_llr_threshold"]))
        & (signals["p3_uncertainty"] >= float(rule["p3_uncertainty_threshold"]))
        & (signals["damaged_support"] >= float(rule["damaged_support_threshold"]))
        & (signals["damaged_event_consensus"] >= float(rule["minimum_damaged_events"]))
    )
    prediction = base.clone()
    prediction[accepted] = 1
    return {"prediction": prediction, "accepted": accepted}


@torch.no_grad()
def class_balanced_knn(
    query_keys: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    topk_per_class: int = 8,
    beta: float = 5.0,
    query_events: Sequence[str] | None = None,
    memory_events: Sequence[str] | None = None,
    exclude_same_event: bool = False,
    max_neighbors_per_event: int = 0,
    chunk_size: int = 512,
) -> Dict[str, torch.Tensor]:
    """Score every class with the same number of neighbours.

    Equal per-class top-k prevents the majority Intact class from winning only
    because it occupies more entries in the memory bank.
    """

    q = F.normalize(query_keys.float(), dim=1)
    m = F.normalize(memory_keys.float(), dim=1)
    labels = memory_labels.long().cpu()
    if m.shape[0] != labels.numel():
        raise ValueError("memory key/label count mismatch")
    if (exclude_same_event or int(max_neighbors_per_event) > 0) and (
        query_events is None or memory_events is None
    ):
        raise ValueError("event names are required for event-aware retrieval")
    if int(max_neighbors_per_event) < 0:
        raise ValueError("max_neighbors_per_event cannot be negative")
    class_indices = [torch.nonzero(labels == c).flatten() for c in range(NUM_CLASSES)]
    if any(index.numel() == 0 for index in class_indices):
        raise RuntimeError("every class must have at least one memory item")
    memory_event_list = list(map(str, memory_events or []))
    predictions, scores, best_similarities, unique_event_counts = [], [], [], []
    for start in range(0, q.shape[0], int(chunk_size)):
        end = min(q.shape[0], start + int(chunk_size))
        similarity = q[start:end] @ m.t()
        chunk_scores = []
        chunk_best = []
        chunk_unique_events = []
        for cls, indices_cpu in enumerate(class_indices):
            indices = indices_cpu.to(similarity.device)
            class_similarity = similarity[:, indices]
            if exclude_same_event:
                memory_cls_events = [memory_event_list[int(i)] for i in indices_cpu]
                for row, event in enumerate(query_events[start:end]):
                    forbidden = torch.tensor(
                        [candidate == str(event) for candidate in memory_cls_events],
                        device=similarity.device,
                        dtype=torch.bool,
                    )
                    class_similarity[row, forbidden] = -torch.inf
            available = torch.isfinite(class_similarity).sum(dim=1)
            if int(available.min()) == 0:
                raise RuntimeError(
                    f"same-event exclusion leaves class {cls} without neighbours"
                )
            k = min(int(topk_per_class), class_similarity.shape[1])
            if int(max_neighbors_per_event) > 0:
                memory_cls_events = [memory_event_list[int(i)] for i in indices_cpu]
                event_candidates = []
                event_candidate_codes = []
                for event_code, event in enumerate(sorted(set(memory_cls_events))):
                    event_mask = torch.tensor(
                        [candidate == event for candidate in memory_cls_events],
                        device=similarity.device,
                        dtype=torch.bool,
                    )
                    event_similarity = class_similarity[:, event_mask]
                    take = min(int(max_neighbors_per_event), event_similarity.shape[1])
                    values = event_similarity.topk(take, dim=1).values
                    event_candidates.append(values)
                    event_candidate_codes.append(
                        torch.full_like(values, event_code, dtype=torch.long)
                    )
                diverse = torch.cat(event_candidates, dim=1)
                diverse_codes = torch.cat(event_candidate_codes, dim=1)
                k = min(k, diverse.shape[1])
                top_result = diverse.topk(k, dim=1)
                top = top_result.values
                selected_event_codes = diverse_codes.gather(1, top_result.indices)
            else:
                top_result = class_similarity.topk(k, dim=1)
                top = top_result.values
                if memory_event_list:
                    memory_cls_events = [memory_event_list[int(i)] for i in indices_cpu]
                    event_names = sorted(set(memory_cls_events))
                    event_to_code = {event: code for code, event in enumerate(event_names)}
                    class_event_codes = torch.tensor(
                        [event_to_code[event] for event in memory_cls_events],
                        device=similarity.device,
                        dtype=torch.long,
                    )
                    selected_event_codes = class_event_codes[top_result.indices]
                else:
                    selected_event_codes = torch.zeros_like(top_result.indices)
            valid = torch.isfinite(top)
            affinity = torch.exp(-float(beta) * (1.0 - top.clamp(-1.0, 1.0)))
            affinity = affinity * valid
            score = affinity.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            chunk_scores.append(score)
            chunk_best.append(top[:, 0])
            chunk_unique_events.append(
                torch.tensor(
                    [
                        selected_event_codes[row, valid[row]].unique().numel()
                        for row in range(selected_event_codes.shape[0])
                    ],
                    device=similarity.device,
                    dtype=torch.float32,
                )
            )
        score_matrix = torch.stack(chunk_scores, dim=1)
        predictions.append(score_matrix.argmax(dim=1).cpu())
        scores.append(score_matrix.cpu())
        best_similarities.append(torch.stack(chunk_best, dim=1).cpu())
        unique_event_counts.append(torch.stack(chunk_unique_events, dim=1).cpu())
    return {
        "prediction": torch.cat(predictions),
        "scores": torch.cat(scores),
        "best_similarity_by_class": torch.cat(best_similarities),
        "unique_neighbor_events_by_class": torch.cat(unique_event_counts),
        "max_neighbors_per_event": int(max_neighbors_per_event),
    }


def prior_corrected_retrieval(
    scores: torch.Tensor,
    class_counts: Sequence[int | float],
    tau: float,
    *,
    smoothing: float = 1.0,
) -> Dict[str, torch.Tensor | float | list[float]]:
    """Restore the train-region class prior after balanced retrieval.

    ``class_balanced_knn`` returns positive affinity evidence, not logits.
    Consequently the Bayes-style correction has to be applied in log-evidence
    space.  At ``tau=0`` this is an exact decision guard because logarithm is
    monotonic.  Counts must be measured before memory balancing and must never
    come from validation or test queries.
    """

    evidence = scores.detach().float().cpu()
    if evidence.ndim != 2 or evidence.shape[1] != NUM_CLASSES:
        raise ValueError("scores must have shape [N,3]")
    counts = torch.as_tensor(class_counts, dtype=torch.float64).flatten()
    if counts.numel() != NUM_CLASSES or bool((counts < 0).any()):
        raise ValueError("class_counts must contain three non-negative values")
    if float(counts.sum()) <= 0.0:
        raise ValueError("class_counts must have positive total mass")
    if float(smoothing) < 0.0:
        raise ValueError("smoothing cannot be negative")
    priors = (counts + float(smoothing)) / (
        counts.sum() + NUM_CLASSES * float(smoothing)
    )
    corrected = evidence.clamp_min(1e-12).log() + float(tau) * priors.log().float()
    return {
        "prediction": corrected.argmax(dim=1),
        "scores": corrected,
        "priors": [float(value) for value in priors],
        "tau": float(tau),
        "smoothing": float(smoothing),
    }


def p3_anchored_memory_fusion(
    base_probabilities: torch.Tensor,
    memory_scores: torch.Tensor,
    alpha: float,
) -> Dict[str, torch.Tensor | float]:
    """Use retrieval as product-of-experts evidence while preserving P3.

    P3 already contains the natural task prior, so the balanced-memory score
    must not become a second standalone classifier.  Positive ``alpha`` adds
    only the relative evidence carried by the memory affinity.  ``alpha=0`` is
    an exact P3 guard, including probabilities up to floating-point roundoff.
    """

    base = base_probabilities.detach().float().cpu()
    memory = memory_scores.detach().float().cpu()
    if base.ndim != 2 or base.shape[1] != NUM_CLASSES:
        raise ValueError("base_probabilities must have shape [N,3]")
    if memory.shape != base.shape:
        raise ValueError("memory_scores must match base_probabilities")
    if bool((base < 0).any()) or bool((memory < 0).any()):
        raise ValueError("P3 probabilities and memory affinities cannot be negative")
    if float(alpha) < 0.0:
        raise ValueError("alpha cannot be negative")
    base = base / base.sum(dim=1, keepdim=True).clamp_min(1e-12)
    logits = base.clamp_min(1e-12).log()
    if float(alpha) > 0.0:
        logits = logits + float(alpha) * memory.clamp_min(1e-12).log()
    probabilities = logits.softmax(dim=1)
    return {
        "prediction": probabilities.argmax(dim=1),
        "probabilities": probabilities,
        "logits": logits,
        "alpha": float(alpha),
    }


def class_event_stratum_weights(
    labels: torch.Tensor,
    events: Sequence[str],
    counts_before_balance: Mapping[str, Mapping[str, int]],
    counts_after_balance: Mapping[str, Mapping[str, int]],
) -> torch.Tensor:
    """Inverse-stratum weights for train-only LOEO prior calibration.

    A class/event-balanced memory is a stratified sample of all eligible train
    regions.  Each retained item represents ``N_before/N_after`` regions from
    its class-event stratum.  These weights restore the natural eligible-train
    distribution without rebuilding an unbalanced cache.
    """

    labels = labels.long().flatten().cpu()
    if labels.numel() != len(events):
        raise ValueError("event count must match label count")
    names = ("intact", "damaged", "destroyed")
    weights = torch.empty(labels.numel(), dtype=torch.float64)
    for index, (label, event) in enumerate(zip(labels.tolist(), map(str, events))):
        if label < 0 or label >= NUM_CLASSES:
            raise ValueError(f"invalid class label: {label}")
        name = names[label]
        before = int(counts_before_balance.get(name, {}).get(event, 0))
        after = int(counts_after_balance.get(name, {}).get(event, 0))
        if before <= 0 or after <= 0:
            raise RuntimeError(
                f"unrepresented train stratum: class={name}, event={event}, "
                f"before={before}, after={after}"
            )
        weights[index] = before / after
    return weights


def weighted_complement_audit(
    truth: torch.Tensor,
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    weights: torch.Tensor,
    events: Sequence[str],
) -> Dict[str, object]:
    """Weighted rescue/harm audit used only for train LOEO tau selection."""

    truth = truth.long().flatten().cpu()
    base = base_prediction.long().flatten().cpu()
    candidate = candidate_prediction.long().flatten().cpu()
    weights = weights.double().flatten().cpu()
    if not (truth.numel() == base.numel() == candidate.numel() == weights.numel()):
        raise ValueError("weighted audit tensors must have equal length")
    if len(events) != truth.numel():
        raise ValueError("event count must match weighted audit tensors")
    base_correct = base == truth
    candidate_correct = candidate == truth
    rescue = (~base_correct) & candidate_correct
    harm = base_correct & (~candidate_correct)

    def summarize(mask: torch.Tensor) -> Dict[str, float]:
        rescued = float(weights[rescue & mask].sum())
        harmed = float(weights[harm & mask].sum())
        return {"rescued": rescued, "harmed": harmed, "net": rescued - harmed}

    per_class = {
        name: summarize(truth == cls)
        for cls, name in enumerate(("intact", "damaged", "destroyed"))
    }
    event_net = {}
    for event in sorted(set(map(str, events))):
        mask = torch.tensor([str(item) == event for item in events], dtype=torch.bool)
        event_net[event] = summarize(mask)["net"]
    overall = summarize(torch.ones_like(truth, dtype=torch.bool))
    return {
        "overall": overall,
        "per_class": per_class,
        "event_net": event_net,
        "nonnegative_events": sum(value >= 0.0 for value in event_net.values()),
        "represented_train_mass": float(weights.sum()),
    }


def confusion_matrix(
    truth: torch.Tensor, prediction: torch.Tensor, classes: int = NUM_CLASSES
) -> torch.Tensor:
    truth, prediction = truth.long().cpu(), prediction.long().cpu()
    valid = (truth >= 0) & (truth < classes) & (prediction >= 0) & (prediction < classes)
    flat = truth[valid] * classes + prediction[valid]
    return torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def metrics_from_predictions(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    events: Sequence[str],
) -> Dict[str, object]:
    matrix = confusion_matrix(truth, prediction)
    tp = matrix.diag().float()
    precision = tp / matrix.sum(dim=0).float().clamp_min(1)
    recall = tp / matrix.sum(dim=1).float().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    event_metrics: Dict[str, Dict[str, float]] = {}
    for event in sorted(set(map(str, events))):
        indices = torch.tensor([item == event for item in events], dtype=torch.bool)
        event_matrix = confusion_matrix(truth[indices], prediction[indices])
        etp = event_matrix.diag().float()
        ep = etp / event_matrix.sum(dim=0).float().clamp_min(1)
        er = etp / event_matrix.sum(dim=1).float().clamp_min(1)
        ef1 = 2 * ep * er / (ep + er).clamp_min(1e-12)
        event_metrics[event] = {
            "macro_f1": float(ef1.mean()),
            "accuracy": float(etp.sum() / event_matrix.sum().clamp_min(1)),
        }
    return {
        "confusion": matrix.tolist(),
        "accuracy": float(tp.sum() / matrix.sum().clamp_min(1)),
        "macro_f1": float(f1.mean()),
        "class_f1": [float(value) for value in f1],
        "class_recall": [float(value) for value in recall],
        "event_macro_f1": (
            sum(item["macro_f1"] for item in event_metrics.values())
            / max(len(event_metrics), 1)
        ),
        "events": event_metrics,
    }


def complement_audit(
    truth: torch.Tensor,
    base_prediction: torch.Tensor,
    retrieval_prediction: torch.Tensor,
) -> Dict[str, object]:
    truth = truth.long().cpu()
    base_prediction = base_prediction.long().cpu()
    retrieval_prediction = retrieval_prediction.long().cpu()
    base_correct = base_prediction == truth
    retrieval_correct = retrieval_prediction == truth
    rescue = (~base_correct) & retrieval_correct
    harm = base_correct & (~retrieval_correct)
    per_class = {}
    for cls, name in enumerate(("intact", "damaged", "destroyed")):
        mask = truth == cls
        corrected = int((rescue & mask).sum())
        harmed = int((harm & mask).sum())
        per_class[name] = {
            "p3_wrong_retrieval_right": corrected,
            "p3_right_retrieval_wrong": harmed,
            "net_complement": corrected - harmed,
        }
    corrected, harmed = int(rescue.sum()), int(harm.sum())
    return {
        "p3_wrong_retrieval_right": corrected,
        "p3_right_retrieval_wrong": harmed,
        "net_complement": corrected - harmed,
        "per_class": per_class,
    }


def gain_retention_audit(
    base_score: float,
    ordinary_score: float,
    candidate_score: float,
    *,
    minimum_retention: float = 0.80,
) -> Dict[str, object]:
    """Evaluate gain retention without emitting non-finite JSON values.

    A retention ratio is meaningful only when the ordinary retrieval setting
    improves on P3.  If ordinary retrieval is not better than P3, the candidate
    must instead beat both P3 and ordinary retrieval.  This prevents ``-inf``
    from being mistaken for a model metric and keeps the admission rule strict.
    """

    base = float(base_score)
    ordinary = float(ordinary_score)
    candidate = float(candidate_score)
    ordinary_gain = ordinary - base
    candidate_gain = candidate - base
    if ordinary_gain > 1e-12:
        retention = candidate_gain / ordinary_gain
        check_name = "retains_80pct_of_ordinary_gain"
        passed = candidate_gain > 0.0 and retention >= float(minimum_retention)
    else:
        retention = None
        check_name = "candidate_beats_p3_when_ordinary_has_no_gain"
        passed = candidate_gain > 0.0 and candidate >= ordinary
    return {
        "base_score": base,
        "ordinary_score": ordinary,
        "candidate_score": candidate,
        "ordinary_gain": ordinary_gain,
        "candidate_gain": candidate_gain,
        "gain_retention": retention,
        "check_name": check_name,
        "passed": bool(passed),
    }


def binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact rank AUROC with average ranks for ties."""

    values = scores.detach().float().flatten().cpu()
    labels = targets.detach().bool().flatten().cpu()
    if values.numel() != labels.numel() or values.numel() == 0:
        raise ValueError("scores and targets must be non-empty vectors of equal size")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    sorted_ranks = torch.arange(1, values.numel() + 1, dtype=torch.float64)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    cursor = 0
    for count in counts.tolist():
        end = cursor + int(count)
        sorted_ranks[cursor:end] = sorted_ranks[cursor:end].mean()
        cursor = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    rank_sum = float(ranks[labels].sum())
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def _rank01(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().flatten().cpu()
    if values.numel() <= 1:
        return torch.ones_like(values)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, values.numel())
    return ranks


def selective_damaged_rescue_audit(
    truth: torch.Tensor,
    base_prediction: torch.Tensor,
    base_probabilities: torch.Tensor,
    retrieval_prediction: torch.Tensor,
    retrieval_scores: torch.Tensor,
    events: Sequence[str],
    *,
    coverages: Sequence[float] = (0.01, 0.025, 0.05, 0.10, 0.20),
) -> Dict[str, object]:
    """Can confidence isolate useful Intact->Damaged retrieval proposals?

    The audit never changes the model.  It studies only proposals where P3
    predicts Intact and retrieval predicts Damaged.  True Damaged proposals are
    rescues; true Intact proposals are harms.  True Destroyed proposals are
    reported separately and excluded from the binary confidence AUROC because
    neither expert is correct for them.
    """

    truth = truth.long().flatten().cpu()
    base_prediction = base_prediction.long().flatten().cpu()
    retrieval_prediction = retrieval_prediction.long().flatten().cpu()
    base_probabilities = base_probabilities.float().cpu()
    retrieval_scores = retrieval_scores.float().cpu()
    count = truth.numel()
    if any(item.shape[0] != count for item in (
        base_prediction, base_probabilities, retrieval_prediction, retrieval_scores
    )):
        raise ValueError("RM4 tensors must share the same first dimension")
    if base_probabilities.shape[1] != NUM_CLASSES or retrieval_scores.shape[1] != NUM_CLASSES:
        raise ValueError("RM4 expects three conditional damage classes")
    if len(events) != count:
        raise ValueError("event count must match query count")

    proposed = (base_prediction == 0) & (retrieval_prediction == 1)
    auditable = proposed & ((truth == 0) | (truth == 1))
    destroyed_proposals = int((proposed & (truth == 2)).sum())
    indices = torch.nonzero(auditable, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("RM4 found no auditable Intact-to-Damaged proposals")
    target = truth[indices] == 1
    if int(target.sum()) == 0 or int((~target).sum()) == 0:
        raise RuntimeError("RM4 requires both rescue and harm proposals")

    selected_retrieval_scores = retrieval_scores[indices]
    damaged_score = selected_retrieval_scores[:, 1]
    competing_score = torch.maximum(
        selected_retrieval_scores[:, 0], selected_retrieval_scores[:, 2]
    )
    retrieval_margin = damaged_score - competing_score
    p3_affected_probability = base_probabilities[indices, 1:3].sum(dim=1)
    p3_sorted = base_probabilities[indices].sort(dim=1, descending=True).values
    p3_uncertainty = 1.0 - (p3_sorted[:, 0] - p3_sorted[:, 1])
    joint_rank_product = torch.sqrt(
        _rank01(retrieval_margin).clamp_min(0.0)
        * _rank01(p3_affected_probability).clamp_min(0.0)
    )
    signals = {
        "retrieval_damaged_score": damaged_score,
        "retrieval_margin": retrieval_margin,
        "p3_affected_probability": p3_affected_probability,
        "p3_uncertainty": p3_uncertainty,
        "joint_rank_product": joint_rank_product,
    }
    aucs = {name: binary_auroc(value, target) for name, value in signals.items()}
    finite_aucs = {name: value for name, value in aucs.items() if math.isfinite(value)}
    best_signal = max(finite_aucs, key=finite_aucs.get)

    candidate_events = [str(events[int(index)]) for index in indices]
    all_events = sorted(set(map(str, events)))
    operating_points = []
    order = torch.argsort(signals[best_signal], descending=True, stable=True)
    for coverage in coverages:
        fraction = float(coverage)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("RM4 coverages must be in (0, 1]")
        take = max(1, min(order.numel(), int(math.ceil(fraction * order.numel()))))
        chosen = order[:take]
        rescued = int(target[chosen].sum())
        harmed = int((~target[chosen]).sum())
        event_net = {event: 0 for event in all_events}
        for local_index in chosen.tolist():
            event = candidate_events[int(local_index)]
            event_net[event] += 1 if bool(target[int(local_index)]) else -1
        nonnegative_events = sum(value >= 0 for value in event_net.values())
        operating_points.append({
            "coverage": fraction,
            "selected": int(take),
            "rescued": rescued,
            "harmed": harmed,
            "net_correction": rescued - harmed,
            "precision": rescued / max(take, 1),
            "nonnegative_events": int(nonnegative_events),
            "event_net_correction": event_net,
        })
    admissible_points = [
        point for point in operating_points
        if point["net_correction"] > 0
        and point["precision"] >= 0.55
        and point["rescued"] >= 10
        and point["nonnegative_events"] >= 10
    ]
    checks = {
        "at_least_100_auditable_proposals": int(indices.numel()) >= 100,
        "best_confidence_auroc_at_least_0_65": finite_aucs[best_signal] >= 0.65,
        "predeclared_selective_rescue_point_exists": bool(admissible_points),
    }
    return {
        "proposal_contract": "P3=Intact and retrieval=Damaged; truth Intact/Damaged only",
        "proposed_total": int(proposed.sum()),
        "auditable_proposals": int(indices.numel()),
        "rescue_proposals": int(target.sum()),
        "harm_proposals": int((~target).sum()),
        "destroyed_proposals_excluded": destroyed_proposals,
        "rescue_prevalence": float(target.float().mean()),
        "signal_auroc": aucs,
        "best_signal": best_signal,
        "operating_points": operating_points,
        "best_admissible_point": (
            max(admissible_points, key=lambda item: item["net_correction"])
            if admissible_points else None
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def shuffled_labels(labels: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(labels.numel(), generator=generator)
    return labels.cpu()[order]
