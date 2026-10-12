from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_complete_building_memory_v1.core import (
    binary_auroc, cross_event_topk, macro_f1, stable_string_hash,
)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = -(target * logits.log_softmax(1)).sum(1)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def cross_event_contrastive(embedding: torch.Tensor, label: torch.Tensor,
                            event: torch.Tensor, temperature: float = .10) -> torch.Tensor:
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (label[:, None] == label[None]) & (event[:, None] != event[None]) & ~identity
    valid = positive.any(1)
    if not valid.any():
        return embedding.sum() * 0
    similarity = similarity.masked_fill(identity, -1e4)
    log_probability = similarity - torch.logsumexp(similarity, 1, keepdim=True)
    value = (log_probability * positive).sum(1) / positive.sum(1).clamp_min(1)
    return -value[valid].mean()


def hard_negative_margin(embedding: torch.Tensor, label: torch.Tensor,
                         event: torch.Tensor, margin: float = .20) -> torch.Tensor:
    similarity = embedding @ embedding.T
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (label[:, None] == label[None]) & (event[:, None] != event[None]) & ~identity
    negative = (label[:, None] != label[None]) & ~identity
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embedding.sum() * 0
    positive_score = similarity.masked_fill(~positive, -1e4).max(1).values
    negative_score = similarity.masked_fill(~negative, -1e4).max(1).values
    return F.relu(margin + negative_score[valid] - positive_score[valid]).mean()


def event_prototype_alignment(embedding: torch.Tensor, label: torch.Tensor,
                              event: torch.Tensor) -> torch.Tensor:
    losses = []
    for class_id in torch.unique(label):
        prototypes = []
        for event_id in torch.unique(event[label == class_id]):
            mask = (label == class_id) & (event == event_id)
            prototypes.append(F.normalize(embedding[mask].mean(0), dim=0))
        if len(prototypes) >= 2:
            values = torch.stack(prototypes)
            centre = F.normalize(values.mean(0), dim=0).detach()
            losses.append((1 - values @ centre).mean())
    return torch.stack(losses).mean() if losses else embedding.sum() * 0


def balanced_weights(label: torch.Tensor, event: torch.Tensor,
                     purity: torch.Tensor | None = None) -> torch.Tensor:
    class_count = torch.bincount(label, minlength=4).float().clamp_min(1)
    event_count = Counter(int(value) for value in event)
    class_weight = class_count.reciprocal()[label]
    event_weight = torch.tensor(
        [1.0 / event_count[int(event_id)] for event_id in event],
        dtype=torch.float32,
        device=label.device,
    )
    weight = class_weight * event_weight
    if purity is not None:
        weight *= purity.float().clamp(.25, 1)
    return weight / weight.sum().clamp_min(1e-12)


def role_means(cache: dict) -> torch.Tensor:
    token = cache["role_tokens"].float()
    valid = cache["role_token_mask"].float()[..., None]
    return (token * valid).sum(2) / valid.sum(2).clamp_min(1)


class MetricTeacher(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, classes: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.encoder(value.float()), dim=1, eps=1e-8)
        return self.classifier(embedding), embedding


class PrototypeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, classes: int = 4) -> None:
        super().__init__()
        self.encoder = MetricTeacher(input_dim, hidden, classes).encoder
        self.prototypes = nn.Parameter(F.normalize(torch.randn(classes, hidden), dim=1))
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.encoder(value.float()), dim=1, eps=1e-8)
        logits = self.scale.clamp(1, 30) * embedding @ F.normalize(self.prototypes, dim=1).T
        return logits, embedding


class RoleSupportAdapter(nn.Module):
    """Support roles refine a frozen query embedding; no support label is input."""
    def __init__(self, hidden: int, role_dim: int, classifier: nn.Linear) -> None:
        super().__init__()
        self.query_role = nn.Linear(role_dim, hidden)
        self.support_role = nn.Linear(role_dim, hidden)
        heads = 4 if hidden % 4 == 0 else 1
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(hidden, classifier.out_features)
        self.classifier.load_state_dict(classifier.state_dict())

    def forward(self, base: torch.Tensor, query_roles: torch.Tensor,
                support_roles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_role(query_roles.float())
        support = self.support_role(support_roles.float().flatten(1, 2))
        attended, _ = self.attention(query, support, support, need_weights=False)
        candidate = F.normalize(base + self.norm(attended.mean(1)), dim=1)
        # Exact base identity at gate=0, while the gate still receives a gradient.
        refined = base + torch.tanh(self.gate) * (candidate - base)
        return self.classifier(refined), refined


class HardCaseRouter(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.float()).squeeze(1)


def classification_report(target: torch.Tensor, probability: torch.Tensor,
                          event_names: list[str]) -> dict:
    prediction = probability.argmax(1)
    class_f1, class_recall = [], []
    for label in range(4):
        truth = target == label; pred = prediction == label
        tp = (truth & pred).sum().float(); fp = (~truth & pred).sum().float(); fn = (truth & ~pred).sum().float()
        class_f1.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
        class_recall.append(float(tp / (tp + fn).clamp_min(1)))
    event_f1 = {}
    for event in sorted(set(event_names)):
        mask = torch.tensor([name == event for name in event_names])
        event_f1[event] = macro_f1(target[mask], prediction[mask])
    return {
        "macro_f1": sum(class_f1) / 4, "class_f1": class_f1,
        "class_recall": class_recall,
        "event_macro_f1": sum(event_f1.values()) / max(len(event_f1), 1),
        "events": event_f1,
    }


@torch.no_grad()
def retrieval_report(query_embedding: torch.Tensor, memory_embedding: torch.Tensor,
                     query_label: torch.Tensor, memory_label: torch.Tensor,
                     query_events: list[str], memory_events: list[str], *,
                     topk: int, maximum_per_event: int, device: str,
                     temperature: float = .10) -> tuple[dict, torch.Tensor, torch.Tensor]:
    found = cross_event_topk(
        query_embedding, memory_embedding, query_events, memory_events,
        topk=topk, maximum_per_event=maximum_per_event, device=device,
    )
    labels = memory_label[found.indices.clamp_min(0)]
    finite = torch.isfinite(found.scores)
    safe_scores = found.scores.masked_fill(~finite, -1e4)
    weights = torch.softmax(safe_scores / temperature, 1) * finite
    posterior = torch.zeros(len(query_embedding), 4)
    posterior.scatter_add_(1, labels, weights)
    posterior /= posterior.sum(1, keepdim=True).clamp_min(1e-8)
    posterior[~found.coverage] = .25
    purity = ((labels == query_label[:, None]) & finite).sum(1).float() / finite.sum(1).clamp_min(1)
    aucs = [binary_auroc(query_label == label, posterior[:, label]) for label in range(4)]
    valid_auc = [value for value in aucs if value == value]
    recall_at_k = [
        float(((labels == label) & finite).any(1)[query_label == label].float().mean())
        if bool((query_label == label).any()) else float("nan")
        for label in range(4)
    ]
    event_auc = {}
    for event in sorted(set(query_events)):
        mask = torch.tensor([name == event for name in query_events])
        values = [binary_auroc(query_label[mask] == label, posterior[mask, label]) for label in range(4)]
        values = [value for value in values if value == value]
        if values:
            event_auc[event] = sum(values) / len(values)
    report = {
        "macro_ovr_auroc": sum(valid_auc) / max(len(valid_auc), 1),
        "auroc_by_class": aucs, "recall_at_k_by_class": recall_at_k,
        "mean_neighbour_purity": float(purity[found.coverage].mean()) if found.coverage.any() else 0.0,
        "coverage": float(found.coverage.float().mean()),
        "event_macro_auroc": sum(event_auc.values()) / max(len(event_auc), 1),
        "correct_events": sum(value > .5 for value in event_auc.values()),
        "event_total": len(event_auc), "events": event_auc,
    }
    return report, posterior, found.indices


def event_ids(event_names: list[str]) -> torch.Tensor:
    return stable_string_hash(event_names)
