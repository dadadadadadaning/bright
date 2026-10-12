from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_complete_building_memory_v1.core import binary_auroc, macro_f1
from bda_benchmark.experiments.spi_hierarchical_residual_memory_v1.core import (
    dual_bank_error_score, granularity_targets, mean_retrieval_similarity,
)
from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import retrieval_weights


def uncertainty(probability: torch.Tensor) -> torch.Tensor:
    """Observable P3 uncertainty; high means P3 is less confident."""
    return 1.0 - probability.float().amax(1)


def source_quantile_thresholds(probability: torch.Tensor, source: torch.Tensor, quantile: float = .80):
    value = uncertainty(probability); thresholds = torch.zeros(4)
    for cls in range(4):
        rows = value[source.long() == cls]
        thresholds[cls] = torch.quantile(rows, quantile) if len(rows) else torch.inf
    return thresholds


def high_risk_mask(probability: torch.Tensor, source: torch.Tensor, thresholds: torch.Tensor):
    return uncertainty(probability) >= thresholds[source.long()]


def family_probability(retrieval, memory_family: torch.Tensor):
    weight, _ = retrieval_weights(retrieval)
    safe = retrieval.indices.clamp_min(0)
    one_hot = F.one_hot(memory_family[safe].long(), 5).float()
    return (one_hot * weight[..., None]).sum(1)


def memory_features(correct, residual, memory_family: torch.Tensor):
    family = family_probability(residual, memory_family)
    error_score, difference = dual_bank_error_score(correct, residual)
    return torch.cat((family, error_score[:, None], difference[:, None],
                      correct.coverage.float()[:, None], residual.coverage.float()[:, None]), 1)


class FamilyDiagnosticHead(nn.Module):
    """Small equal-capacity classifier. It never modifies P3 logits."""
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 5))

    def forward(self, x): return self.net(x.float())


def class_weights(target: torch.Tensor):
    counts = torch.bincount(target.long(), minlength=5).float().clamp_min(1)
    return (counts.sum() / counts) / (counts.sum() / counts).mean()


def train_head(feature: torch.Tensor, target: torch.Tensor, *, steps: int, batch_size: int,
               learning_rate: float, seed: int, device: torch.device):
    torch.manual_seed(seed); model = FamilyDiagnosticHead(feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weight = class_weights(target).to(device); generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        index = torch.randint(len(target), (min(batch_size, len(target)),), generator=generator)
        logits = model(feature[index].to(device)); loss = F.cross_entropy(logits, target[index].to(device), weight=weight)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
    return model.cpu().eval()


def family_report(target: torch.Tensor, probability: torch.Tensor, events: list[str]):
    prediction = probability.argmax(1); aucs = [binary_auroc((target == c).long(), probability[:, c]) for c in range(5)]
    valid_auc = [x for x in aucs if x == x]; event_f1 = {}
    for event in sorted(set(events)):
        mask = torch.tensor([x == event for x in events])
        event_f1[event] = macro_f1(target[mask], prediction[mask], classes=5)
    return {"macro_ovr_auroc": sum(valid_auc) / max(len(valid_auc), 1),
            "macro_f1": macro_f1(target, prediction, classes=5),
            "affected_miss_recall": float(((prediction == 1) & (target == 1)).sum() / (target == 1).sum().clamp_min(1)),
            "affected_false_alarm_recall": float(((prediction == 2) & (target == 2)).sum() / (target == 2).sum().clamp_min(1)),
            "severity_confusion_recall": float(((prediction == 3) & (target == 3)).sum() / (target == 3).sum().clamp_min(1)),
            "events": event_f1, "event_macro_f1": sum(event_f1.values()) / max(len(event_f1), 1)}

