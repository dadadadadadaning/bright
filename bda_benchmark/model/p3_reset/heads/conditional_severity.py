from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalSeverityResidual(nn.Module):
    """D1: explicit P(B) x P(I/D/X|B), safely blended from P3."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(channels, 3, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.transition = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        p3_logits: torch.Tensor,
        feature: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = F.interpolate(self.classifier(feature), p3_logits.shape[-2:], mode="bilinear", align_corners=False)
        conditional = (p3_logits[:, 1:4] + residual).log_softmax(1)
        p_building = F.interpolate(building_probability.detach(), p3_logits.shape[-2:], mode="bilinear", align_corners=False).clamp(1e-6, 1 - 1e-6)
        factorised = torch.cat((torch.log1p(-p_building), p_building.log() + conditional), dim=1)
        gate = torch.tanh(self.transition)
        output = p3_logits + gate * (factorised - p3_logits)
        return output, {
            "severity_transition": gate.reshape(1),
            "severity_residual_abs": residual.detach().abs().mean().reshape(1),
            "factorisation_delta_abs": (factorised - p3_logits).detach().abs().mean().reshape(1),
        }
