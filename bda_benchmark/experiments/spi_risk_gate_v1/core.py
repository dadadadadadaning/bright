from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_core_ring_adapter_v1.model import adapter_inputs
from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import (
    RegionMaps,
    broadcast_table,
    scatter_region_mean,
)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.float().clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(1, keepdim=True) / torch.log(
        probabilities.new_tensor(float(probabilities.shape[1]))
    )


def fixed_uncertainty_gate(
    base_logits: torch.Tensor,
    support: torch.Tensor,
    threshold: float = 0.50,
    temperature: float = 0.10,
) -> torch.Tensor:
    entropy = normalized_entropy(F.softmax(base_logits.float(), dim=1))
    return support.float() * torch.sigmoid((entropy - threshold) / temperature)


def boundary_depth_weight(
    support: torch.Tensor,
    radius: int = 2,
    boundary_floor: float = 0.20,
) -> torch.Tensor:
    """Batched equivalent of the locked Selective Context interior weighting."""
    if radius < 0 or not 0.0 <= boundary_floor <= 1.0:
        raise ValueError("invalid boundary settings")
    binary = support.float()
    output = binary * boundary_floor
    if radius == 0:
        return binary
    current = binary
    for depth in range(1, radius + 1):
        current = (F.avg_pool2d(current, 3, stride=1, padding=1) == 1.0).float()
        value = boundary_floor + (1.0 - boundary_floor) * depth / radius
        output = torch.where(current.bool(), output.new_tensor(value), output)
    return output


def apply_gate(base_logits: torch.Tensor, candidate_logits: torch.Tensor, gate: torch.Tensor):
    if gate.shape != base_logits[:, :1].shape:
        raise ValueError("gate/logit shapes do not align")
    return base_logits + gate.clamp(0.0, 1.0) * (candidate_logits - base_logits)


def _region_scalar_tables(base_prediction, labels, region_count):
    one_hot = F.one_hot(base_prediction.long(), num_classes=4).permute(0, 3, 1, 2).float()
    class_table, counts = scatter_region_mean(one_hot, labels, region_count)
    foreground_distribution = class_table[:, 1:4]
    purity = foreground_distribution.max(dim=1).values
    area = counts
    if area.numel() > 1:
        normalizer = area[1:].max().clamp_min(1.0)
    else:
        normalizer = area.new_tensor(1.0)
    log_area = torch.log1p(area) / torch.log1p(normalizer)
    purity[0] = 0
    log_area[0] = 0
    return purity, log_area


def risk_features(
    base_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    feature: torch.Tensor,
    regions: RegionMaps,
    boundary_radius: int = 2,
    boundary_floor: float = 0.20,
):
    base_prob = F.softmax(base_logits.float(), dim=1)
    candidate_prob = F.softmax(candidate_logits.float(), dim=1)
    base_top2 = base_prob.topk(2, dim=1).values
    base_confidence = base_top2[:, :1]
    base_margin = base_top2[:, :1] - base_top2[:, 1:2]
    candidate_confidence = candidate_prob.max(dim=1, keepdim=True).values
    disagreement = (base_prob.argmax(1, keepdim=True) != candidate_prob.argmax(1, keepdim=True)).float()
    probability_shift = (candidate_prob - base_prob).abs().sum(1, keepdim=True)
    support = (regions.labels > 0).unsqueeze(1).float()
    depth = boundary_depth_weight(support, boundary_radius, boundary_floor)
    purity_table, area_table = _region_scalar_tables(base_prob.argmax(1), regions.labels, regions.region_count)
    purity = broadcast_table(purity_table[:, None], regions.labels)
    log_area = broadcast_table(area_table[:, None], regions.labels)
    _, contrast, _ = adapter_inputs("m4_core_ring_region", feature, regions)
    contrast_strength = torch.tanh(contrast.float().square().mean(1, keepdim=True).sqrt())
    inputs = torch.cat(
        [
            normalized_entropy(base_prob),
            base_confidence,
            base_margin,
            probability_shift,
            candidate_confidence - base_confidence,
            disagreement,
            purity,
            log_area,
            depth,
            contrast_strength,
        ],
        dim=1,
    )
    return inputs, support, depth


class RiskGate(nn.Module):
    def __init__(self, inputs: int = 10, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, inputs: torch.Tensor, support: torch.Tensor):
        return support.float() * torch.sigmoid(self.network(inputs.float()))


def selector_targets(base_logits, candidate_logits, target_low):
    base_prediction = base_logits.argmax(1)
    candidate_prediction = candidate_logits.argmax(1)
    valid = target_low != 255
    positive = valid & (candidate_prediction == target_low) & (base_prediction != target_low)
    negative = valid & (base_prediction == target_low) & (candidate_prediction != target_low)
    decisive = positive | negative
    target = positive.float().unsqueeze(1)
    return target, decisive.unsqueeze(1), positive, negative


def balanced_selector_bce(gate, target, decisive):
    positive = (decisive & (target > 0.5)).float()
    negative = (decisive & (target <= 0.5)).float()
    positive_loss = -(torch.log(gate.clamp_min(1e-6)) * positive).sum() / positive.sum().clamp_min(1.0)
    negative_loss = -(torch.log((1.0 - gate).clamp_min(1e-6)) * negative).sum() / negative.sum().clamp_min(1.0)
    return 0.5 * (positive_loss + negative_loss)


@dataclass
class ChangeCounts:
    changed: int = 0
    corrected: int = 0
    harmed: int = 0

    @property
    def net(self):
        return self.corrected - self.harmed

    def as_dict(self):
        return {
            "changed": self.changed,
            "corrected": self.corrected,
            "harmed": self.harmed,
            "net": self.net,
            "modification_precision": self.corrected / max(self.changed, 1),
            "harm_rate": self.harmed / max(self.changed, 1),
        }
