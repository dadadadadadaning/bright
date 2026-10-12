from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalSeverityResidual(nn.Module):
    """D2: P(B) times hierarchical I-vs-Affected and D-vs-X."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.occurrence = nn.Conv2d(channels, 1, 1)
        self.severity = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.occurrence.weight)
        nn.init.zeros_(self.occurrence.bias)
        nn.init.zeros_(self.severity.weight)
        nn.init.zeros_(self.severity.bias)
        self.transition = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        p3_logits: torch.Tensor,
        feature: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        occurrence_residual = F.interpolate(self.occurrence(feature), p3_logits.shape[-2:], mode="bilinear", align_corners=False)
        severity_residual = F.interpolate(self.severity(feature), p3_logits.shape[-2:], mode="bilinear", align_corners=False)
        base_occurrence = torch.logsumexp(p3_logits[:, 2:4], dim=1, keepdim=True) - p3_logits[:, 1:2]
        base_severity = p3_logits[:, 3:4] - p3_logits[:, 2:3]
        p_affected = torch.sigmoid(base_occurrence + occurrence_residual)
        p_destroyed_given_affected = torch.sigmoid(base_severity + severity_residual)
        conditional = torch.cat((
            1.0 - p_affected,
            p_affected * (1.0 - p_destroyed_given_affected),
            p_affected * p_destroyed_given_affected,
        ), dim=1).clamp_min(1e-7)
        p_building = F.interpolate(building_probability.detach(), p3_logits.shape[-2:], mode="bilinear", align_corners=False).clamp(1e-6, 1 - 1e-6)
        factorised = torch.cat((torch.log1p(-p_building), p_building.log() + conditional.log()), dim=1)
        gate = torch.tanh(self.transition)
        output = p3_logits + gate * (factorised - p3_logits)
        return output, {
            "hierarchy_transition": gate.reshape(1),
            "occurrence_residual_abs": occurrence_residual.detach().abs().mean().reshape(1),
            "severity_residual_abs": severity_residual.detach().abs().mean().reshape(1),
            "factorisation_delta_abs": (factorised - p3_logits).detach().abs().mean().reshape(1),
        }


def ordinal_expected_severity_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Squared error on expected I/D/X order, restricted to building pixels."""
    valid = (target != ignore_index) & (target > 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    probabilities = logits[:, 1:4].softmax(1)
    order = probabilities.new_tensor((0.0, 1.0, 2.0))[None, :, None, None]
    expected = (probabilities * order).sum(1)
    truth = (target.float() - 1.0).clamp(0.0, 2.0)
    return F.smooth_l1_loss(expected[valid], truth[valid])
