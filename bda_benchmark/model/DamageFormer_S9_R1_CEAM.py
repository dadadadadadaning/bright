# -*- coding: utf-8 -*-
"""S9-R1 Cross-Event Angular Metric for Change + Local building descriptors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-6)


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class CEAMConfig:
    raw_dim: int = 128
    embed_dim: int = 128
    hidden_dim: int = 256
    num_classes: int = 3
    angular_scale: float = 30.0
    angular_margin: float = 0.20
    contrastive_temperature: float = 0.07
    hard_negative_margin: float = 0.20

    def validate(self) -> None:
        if min(self.raw_dim, self.embed_dim, self.hidden_dim, self.num_classes) <= 0:
            raise ValueError("CEAM dimensions/classes must be positive")
        if self.angular_scale <= 0 or not 0.0 <= self.angular_margin < 1.0:
            raise ValueError("invalid angular scale/margin")
        if self.contrastive_temperature <= 0 or self.hard_negative_margin < 0:
            raise ValueError("invalid contrastive/hard-negative margin")


class ChangeLocalAngularMetric(nn.Module):
    """Damage-semantic embedding with an explicit normalized class geometry."""

    def __init__(self, config: CEAMConfig = CEAMConfig()):
        super().__init__()
        config.validate()
        self.config = config
        d, e, h = config.raw_dim, config.embed_dim, config.hidden_dim
        self.change_encoder = mlp(d, h, e)
        self.local_encoder = mlp(d, h, e)
        self.fusion = mlp(4 * e, h, e)
        self.class_centers = nn.Parameter(torch.empty(config.num_classes, e))
        nn.init.orthogonal_(self.class_centers)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        change_raw = batch.get("change_global")
        local_raw = batch.get("local_summary")
        if not isinstance(change_raw, torch.Tensor) or not isinstance(local_raw, torch.Tensor):
            raise KeyError("CEAM requires change_global and local_summary")
        expected = self.config.raw_dim
        if change_raw.ndim != 2 or change_raw.shape[1] != expected:
            raise ValueError(f"change_global must be [N,{expected}]")
        if local_raw.shape != change_raw.shape:
            raise ValueError("local_summary must match change_global shape")
        change = normalize(self.change_encoder(change_raw))
        local = normalize(self.local_encoder(local_raw))
        interaction = torch.cat(
            (change, local, torch.abs(change - local), change * local), dim=1
        )
        embedding = normalize(change + self.fusion(interaction))
        cosine = embedding @ normalize(self.class_centers).t()
        return {
            "embedding": embedding,
            "change": change,
            "local": local,
            "class_cosine": cosine,
            "inference_logits": self.config.angular_scale * cosine,
        }

    def arcface_logits(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if cosine.ndim != 2 or cosine.shape[1] != self.config.num_classes:
            raise ValueError("cosine must be [N,num_classes]")
        if labels.shape != cosine.shape[:1]:
            raise ValueError("labels must match cosine batch")
        cosine = cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        margin = self.config.angular_margin
        cos_m = torch.cos(cosine.new_tensor(margin))
        sin_m = torch.sin(cosine.new_tensor(margin))
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * cos_m - sine * sin_m
        threshold = torch.cos(cosine.new_tensor(torch.pi - margin))
        correction = torch.sin(cosine.new_tensor(torch.pi - margin)) * margin
        phi = torch.where(cosine > threshold, phi, cosine - correction)
        one_hot = F.one_hot(labels.long(), self.config.num_classes).to(cosine.dtype)
        return self.config.angular_scale * (one_hot * phi + (1.0 - one_hot) * cosine)

    def checkpoint_config(self) -> Dict[str, object]:
        return asdict(self.config)


def cross_event_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if embeddings.ndim != 2 or labels.ndim != 1 or events.shape != labels.shape:
        raise ValueError("invalid CEAM contrastive tensors")
    count = embeddings.shape[0]
    if count < 2:
        zero = embeddings.sum() * 0.0
        return zero, {"valid_anchor_ratio": 0.0, "positive_pairs": 0.0}
    z = normalize(embeddings)
    eye = torch.eye(count, device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & events[:, None].ne(events[None, :]) & ~eye
    valid_anchor = positive.any(dim=1)
    if not bool(valid_anchor.any()):
        zero = embeddings.sum() * 0.0
        return zero, {"valid_anchor_ratio": 0.0, "positive_pairs": 0.0}
    logits = z @ z.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = ~eye
    exp_logits = torch.exp(logits) * denominator_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive = (log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    loss = -mean_positive[valid_anchor].mean()
    return loss, {
        "valid_anchor_ratio": float(valid_anchor.float().mean().detach()),
        "positive_pairs": float(positive.sum().detach()),
    }


def adjacent_class_hard_negative_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hardest adjacent-class negative against a cross-event positive mean."""

    z = normalize(embeddings)
    similarity = z @ z.t()
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & events[:, None].ne(events[None, :]) & ~eye
    adjacent_negative = labels[:, None].sub(labels[None, :]).abs().eq(1) & ~eye
    valid = positive.any(dim=1) & adjacent_negative.any(dim=1)
    if not bool(valid.any()):
        zero = embeddings.sum() * 0.0
        return zero, {"valid_anchor_ratio": 0.0, "mean_positive": 0.0, "mean_hard_negative": 0.0}
    positive_sum = (similarity * positive).sum(dim=1)
    positive_mean = positive_sum / positive.sum(dim=1).clamp_min(1)
    negative_similarity = similarity.masked_fill(~adjacent_negative, -2.0)
    hard_negative = negative_similarity.max(dim=1).values
    losses = F.relu(margin + hard_negative - positive_mean)
    return losses[valid].mean(), {
        "valid_anchor_ratio": float(valid.float().mean().detach()),
        "mean_positive": float(positive_mean[valid].mean().detach()),
        "mean_hard_negative": float(hard_negative[valid].mean().detach()),
    }


def ceam_training_loss(
    model: ChangeLocalAngularMetric,
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    events: torch.Tensor,
    lambda_supcon: float = 0.20,
    lambda_hard: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    arc_logits = model.arcface_logits(outputs["class_cosine"], labels)
    angular = F.cross_entropy(arc_logits, labels.long())
    supcon, supcon_stats = cross_event_supervised_contrastive_loss(
        outputs["embedding"], labels, events, model.config.contrastive_temperature
    )
    hard, hard_stats = adjacent_class_hard_negative_loss(
        outputs["embedding"], labels, events, model.config.hard_negative_margin
    )
    total = angular + lambda_supcon * supcon + lambda_hard * hard
    prediction = outputs["inference_logits"].argmax(dim=1)
    return total, {
        "total": float(total.detach()),
        "angular": float(angular.detach()),
        "supcon": float(supcon.detach()),
        "hard_negative": float(hard.detach()),
        "batch_accuracy": float(prediction.eq(labels).float().mean().detach()),
        "supcon_valid_anchor_ratio": supcon_stats["valid_anchor_ratio"],
        "hard_valid_anchor_ratio": hard_stats["valid_anchor_ratio"],
        "mean_positive_similarity": hard_stats["mean_positive"],
        "mean_hard_negative_similarity": hard_stats["mean_hard_negative"],
    }


__all__ = [
    "CEAMConfig",
    "ChangeLocalAngularMetric",
    "adjacent_class_hard_negative_loss",
    "ceam_training_loss",
    "cross_event_supervised_contrastive_loss",
    "normalize",
]
