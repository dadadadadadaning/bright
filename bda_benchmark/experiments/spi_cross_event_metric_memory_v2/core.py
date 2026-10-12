from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_pseudo_instance_hierarchical_memory_v1.core import (
    DEFAULT_GROUPS, balanced_memory_indices, feature_matrix, parent_ids, provenance,
)
from bda_benchmark.model.PseudoInstanceHierarchicalMemory import parent_mean_context


@dataclass
class PreparedCache:
    x: torch.Tensor
    soft_label: torch.Tensor
    hard_label: torch.Tensor
    purity: torch.Tensor
    foreground_fraction: torch.Tensor
    event: torch.Tensor
    image: torch.Tensor
    parent: torch.Tensor
    direction: torch.Tensor
    keep: torch.Tensor
    event_names: list[str]


def truth_distribution(direction_code: torch.Tensor, delta_hist: torch.Tensor):
    """Recover foreground soft labels and purity from a constant b->a cluster."""
    after = direction_code.long().remainder(4)
    row = torch.arange(len(after))
    counts4 = delta_hist.long()[row, :, after].clamp_min(0).float()
    total = counts4.sum(1)
    dominant4 = counts4.argmax(1)
    foreground = counts4[:, 1:4]
    foreground_total = foreground.sum(1)
    soft = foreground / foreground_total[:, None].clamp_min(1)
    hard = foreground.argmax(1)
    purity = counts4.max(1).values / total.clamp_min(1)
    foreground_fraction = foreground_total / total.clamp_min(1)
    keep = (total > 0) & (dominant4 > 0) & (foreground_total > 0)
    return soft, hard, purity, foreground_fraction, keep


def prepare_cache(cache: dict, groups=DEFAULT_GROUPS,
                  mean: torch.Tensor | None = None,
                  std: torch.Tensor | None = None):
    child = feature_matrix(cache, groups)
    parent = parent_ids(cache)
    parent_feature = parent_mean_context(child, parent)
    direction = cache["direction_code"].long()
    onehot = F.one_hot(direction.clamp(0, 15), 16).float()
    raw = torch.cat((child, parent_feature, child - parent_feature, onehot), 1)
    soft, hard, purity, foreground_fraction, keep = truth_distribution(
        direction, cache["delta_hist"]
    )
    if mean is None:
        mean = raw[keep].mean(0)
        std = raw[keep].std(0, unbiased=False).clamp_min(1e-4)
    x = ((raw[keep] - mean) / std).clamp(-8, 8)
    event, image = provenance(cache)
    image_rows = cache["image_index"].long()[keep]
    event_names = [cache["image_events"][int(i)] for i in image_rows]
    return PreparedCache(
        x=x, soft_label=soft[keep], hard_label=hard[keep], purity=purity[keep],
        foreground_fraction=foreground_fraction[keep], event=event[keep], image=image[keep],
        parent=parent[keep], direction=direction[keep], keep=keep,
        event_names=event_names,
    ), mean, std


class CrossEventMetricEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.classifier = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor):
        embedding = F.normalize(self.encoder(x), dim=1)
        return self.classifier(embedding), embedding


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = -(target * logits.log_softmax(1)).sum(1)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def cross_event_supervised_contrastive(
    embedding: torch.Tensor, label: torch.Tensor, event: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (label[:, None] == label[None, :]) & (event[:, None] != event[None, :])
    positive &= ~identity
    valid = positive.any(1)
    if not valid.any():
        return embedding.sum() * 0
    similarity = similarity.masked_fill(identity, -1e4)
    log_probability = similarity - torch.logsumexp(similarity, 1, keepdim=True)
    mean_positive = (log_probability * positive.float()).sum(1) / positive.sum(1).clamp_min(1)
    return -mean_positive[valid].mean()


def event_prototype_loss(embedding: torch.Tensor, label: torch.Tensor,
                         event: torch.Tensor) -> torch.Tensor:
    losses = []
    for semantic in range(3):
        prototypes = []
        for event_id in torch.unique(event[label == semantic]):
            mask = (label == semantic) & (event == event_id)
            if mask.any():
                prototypes.append(F.normalize(embedding[mask].mean(0), dim=0))
        if len(prototypes) >= 2:
            values = torch.stack(prototypes)
            centre = F.normalize(values.mean(0), dim=0).detach()
            losses.append((1 - values @ centre).mean())
    return torch.stack(losses).mean() if losses else embedding.sum() * 0


@torch.no_grad()
def class_similarity_scores(
    query: torch.Tensor, memory: torch.Tensor, memory_label: torch.Tensor,
    query_event: torch.Tensor, memory_event: torch.Tensor,
    topk: int = 4, exclude_same_event: bool = True,
    purity: torch.Tensor | None = None, minimum_purity: float = 0.0,
    chunk_size: int = 512,
):
    query = F.normalize(query.float(), dim=1)
    memory = F.normalize(memory.float(), dim=1)
    output, availability = [], []
    for start in range(0, len(query), chunk_size):
        q = query[start:start + chunk_size]
        qe = query_event[start:start + chunk_size]
        similarity = q @ memory.T
        class_rows, class_available = [], []
        for label in range(3):
            allowed = memory_label == label
            if purity is not None:
                allowed &= purity >= minimum_purity
            mask = allowed[None].expand(len(q), -1).clone()
            if exclude_same_event:
                mask &= memory_event[None] != qe[:, None]
            values = similarity.masked_fill(~mask, -1e4)
            count = min(topk, values.shape[1])
            selected = values.topk(count, 1).values
            valid = selected > -9e3
            score = (selected.masked_fill(~valid, 0).sum(1) /
                     valid.sum(1).clamp_min(1))
            score[~valid.any(1)] = -1e4
            class_rows.append(score)
            class_available.append(mask.sum(1))
        output.append(torch.stack(class_rows, 1))
        availability.append(torch.stack(class_available, 1))
    return torch.cat(output), torch.cat(availability)


def probability_from_scores(scores: torch.Tensor, temperature: float = 0.10):
    return (scores / temperature).softmax(1)


def selective_memory_probability(
    parametric_probability: torch.Tensor,
    memory_probability: torch.Tensor,
    availability: torch.Tensor,
    *,
    topk: int,
    minimum_confidence_advantage: float,
    minimum_memory_margin: float,
):
    """Use Memory only for a confident disagreement, otherwise keep the head."""

    if parametric_probability.shape != memory_probability.shape:
        raise ValueError("parametric and memory probabilities must match")
    if availability.shape != memory_probability.shape:
        raise ValueError("availability must have one value per query and class")
    memory_values, memory_labels = memory_probability.max(1)
    parametric_values, parametric_labels = parametric_probability.max(1)
    memory_top2 = memory_probability.topk(
        min(2, memory_probability.shape[1]), 1
    ).values
    memory_margin = (
        memory_top2[:, 0] - memory_top2[:, 1]
        if memory_top2.shape[1] == 2
        else memory_top2[:, 0]
    )
    row = torch.arange(len(memory_labels), device=memory_labels.device)
    enough_neighbours = availability[row, memory_labels] >= int(topk)
    intervene = (
        (memory_labels != parametric_labels)
        & enough_neighbours
        & ((memory_values - parametric_values) >= minimum_confidence_advantage)
        & (memory_margin >= minimum_memory_margin)
    )
    output = parametric_probability.clone()
    output[intervene] = memory_probability[intervene]
    return output, intervene, {
        "memory_margin": memory_margin,
        "confidence_advantage": memory_values - parametric_values,
        "enough_neighbours": enough_neighbours,
    }


def classification_metrics(target: torch.Tensor, probability: torch.Tensor) -> dict:
    prediction = probability.argmax(1)
    f1, recall, precision, support = [], [], [], []
    for label in range(3):
        truth = target == label; predicted = prediction == label
        tp = (truth & predicted).sum().float()
        fp = (~truth & predicted).sum().float()
        fn = (truth & ~predicted).sum().float()
        support.append(int(truth.sum()))
        recall.append(float(tp / (tp + fn).clamp_min(1)))
        precision.append(float(tp / (tp + fp).clamp_min(1)))
        f1.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    present = [value for value, count in zip(f1, support) if count > 0]
    return {
        "accuracy": float((prediction == target).float().mean()),
        "macro_f1": sum(f1) / 3, "present_class_macro_f1": sum(present) / max(len(present), 1),
        "class_f1": f1, "class_precision": precision, "class_recall": recall,
        "class_support": support,
    }


def event_metrics(data: PreparedCache, probability: torch.Tensor) -> dict:
    rows = {}
    for event in sorted(set(data.event_names)):
        mask = torch.tensor([name == event for name in data.event_names])
        rows[event] = classification_metrics(data.hard_label[mask], probability[mask])
    return {
        "event_macro_present_f1": sum(v["present_class_macro_f1"] for v in rows.values()) / max(len(rows), 1),
        "events": rows,
    }


def event_balanced_weights(data: PreparedCache) -> torch.Tensor:
    event_counts = Counter(int(value) for value in data.event)
    class_counts = torch.bincount(data.hard_label, minlength=3).float()
    class_weight = class_counts.sum() / class_counts.clamp_min(1)
    weight = torch.tensor([1.0 / event_counts[int(event)] for event in data.event])
    weight *= class_weight[data.hard_label]
    weight *= data.purity.clamp(0.25, 1.0)
    return weight / weight.sum()


__all__ = [
    "CrossEventMetricEncoder", "PreparedCache", "balanced_memory_indices",
    "class_similarity_scores", "classification_metrics", "cross_event_supervised_contrastive",
    "event_balanced_weights", "event_metrics", "event_prototype_loss", "prepare_cache",
    "probability_from_scores", "soft_cross_entropy", "truth_distribution",
]
