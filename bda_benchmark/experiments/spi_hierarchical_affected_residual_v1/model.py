from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import equal_affected_residual


ABLATIONS = {
    "a2_ao": {"rgb": True, "sar": True, "relative": False,
              "expected_confusion": False, "confidence": False},
    "a3_rgb_structure": {"rgb": True, "sar": False, "relative": True,
                         "expected_confusion": False, "confidence": False},
    "a4_sar_texture": {"rgb": False, "sar": True, "relative": True,
                       "expected_confusion": False, "confidence": False},
    "a5_relative_rgbsar": {"rgb": True, "sar": True, "relative": True,
                           "expected_confusion": False, "confidence": False},
    "a6_expected_confusion": {"rgb": True, "sar": True, "relative": True,
                              "expected_confusion": True, "confidence": False},
    "a7_confidence": {"rgb": True, "sar": True, "relative": True,
                      "expected_confusion": True, "confidence": True},
}


def region_relative(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Subtract the mean of each detached pseudo-instance from its pixels."""

    if labels.ndim == 4:
        labels = labels[:, 0]
    if labels.shape != (features.shape[0], *features.shape[-2:]):
        raise ValueError("labels/features shape mismatch")
    output = torch.zeros_like(features)
    for batch in range(features.shape[0]):
        for label in torch.unique(labels[batch]).tolist():
            if int(label) <= 0:
                continue
            mask = labels[batch] == int(label)
            values = features[batch, :, mask]
            output[batch, :, mask] = values - values.mean(1, keepdim=True)
    return output


class WithinInstanceChangeEvidence(nn.Module):
    """Modality-specific projections followed by instance-relative comparison."""

    def __init__(self, rgb_channels: int, sar_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(rgb_channels, hidden, 1, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
        )
        self.sar_projection = nn.Sequential(
            nn.Conv2d(sar_channels, hidden, 1, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        instance_labels: torch.Tensor,
        *, use_rgb: bool = True,
        use_sar: bool = True,
        use_relative: bool = True,
    ) -> torch.Tensor:
        rgb = self.rgb_projection(rgb)
        sar = self.sar_projection(sar)
        if use_relative:
            rgb = region_relative(rgb, instance_labels)
            sar = region_relative(sar, instance_labels)
        if not use_rgb:
            rgb = torch.zeros_like(rgb)
        if not use_sar:
            sar = torch.zeros_like(sar)
        # Compare relations after modality-specific projection; never compare
        # raw RGB intensity directly with raw SAR amplitude.
        relation = (rgb.abs() - sar.abs()).abs()
        return self.fusion(torch.cat((rgb, sar, relation), dim=1))


class AffectedOccurrenceHead(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        # The head may learn immediately, while gamma below guarantees exact
        # parent-model identity at initialisation.

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(evidence))


class ExpectedConfusionConfidenceGate(nn.Module):
    """Isolated B3/B4 gate for externally computed cluster outcome estimates."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        expected_miou_delta: torch.Tensor,
        expected_damaged_f1_delta: torch.Tensor,
        outcome_probabilities: torch.Tensor,
        *, use_confidence: bool,
    ) -> torch.Tensor:
        safety = torch.sigmoid(expected_miou_delta / self.temperature)
        safety = safety * torch.sigmoid(expected_damaged_f1_delta / self.temperature)
        if use_confidence:
            probability = outcome_probabilities.clamp_min(1e-8)
            entropy = -(probability * probability.log()).sum(-1) / math.log(3.0)
            safety = safety * (1.0 - entropy).clamp(0, 1)
        return safety.clamp(0, 1)


class HierarchicalAffectedResidual(nn.Module):
    """Standalone AO residual; it does not own or modify the frozen parent."""

    def __init__(self, rgb_channels: int, sar_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.evidence = WithinInstanceChangeEvidence(rgb_channels, sar_channels, hidden)
        self.occurrence = AffectedOccurrenceHead(hidden)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        parent_logits: torch.Tensor,
        rgb_features: torch.Tensor,
        sar_features: torch.Tensor,
        instance_labels: torch.Tensor,
        building_mask: torch.Tensor,
        *, safety_gate: torch.Tensor | None = None,
        use_rgb: bool = True,
        use_sar: bool = True,
        use_relative: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_size = parent_logits.shape[-2:]
        labels = F.interpolate(
            instance_labels[:, None].float(), rgb_features.shape[-2:], mode="nearest"
        )[:, 0].long()
        evidence = self.evidence(
            rgb_features, sar_features, labels,
            use_rgb=use_rgb, use_sar=use_sar, use_relative=use_relative,
        )
        residual = self.occurrence(evidence)
        residual = F.interpolate(residual, target_size, mode="bilinear", align_corners=False)
        if safety_gate is not None:
            if safety_gate.ndim == 3:
                safety_gate = safety_gate[:, None]
            residual = residual * safety_gate
        scaled = self.gamma * residual
        output = equal_affected_residual(parent_logits, scaled, building_mask)
        return output, {
            "raw_occurrence_residual": residual,
            "scaled_occurrence_residual": scaled,
            "gamma": self.gamma,
        }


__all__ = [
    "ABLATIONS", "AffectedOccurrenceHead", "ExpectedConfusionConfidenceGate",
    "HierarchicalAffectedResidual", "WithinInstanceChangeEvidence",
    "region_relative",
]

