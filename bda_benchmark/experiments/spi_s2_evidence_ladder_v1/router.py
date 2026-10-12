from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import GATE_INPUT_CHANNELS


EVIDENCE_CHANNELS = {
    "cross_event_memory": 10,
    "im4cd_translation_residual": 8,
    "within_modality_relation_difference": 4,
    "soft_superpixel_grid": 4,
    "soft_superpixel_rgb": 4,
}


class IsolatedEvidenceGate(nn.Module):
    """Utility Gate extension with explicit, fingerprintable evidence groups."""

    def __init__(
        self,
        evidence_groups: Sequence[str],
        *,
        hidden: int = 32,
        allow_combination: bool = False,
    ) -> None:
        super().__init__()
        groups = tuple(evidence_groups)
        unknown = [name for name in groups if name not in EVIDENCE_CHANNELS]
        if unknown:
            raise ValueError(f"unknown evidence groups: {unknown}")
        if not allow_combination and len(groups) != 1:
            raise ValueError("isolated stages must contain exactly one evidence group")
        if allow_combination and len(groups) < 2:
            raise ValueError("combination stage requires at least two accepted groups")
        self.evidence_groups = groups
        self.allow_combination = bool(allow_combination)
        inputs = GATE_INPUT_CHANNELS + sum(EVIDENCE_CHANNELS[name] for name in groups)
        self.network = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -2.0)

    @property
    def input_channels(self) -> int:
        return int(self.network[0].in_channels)

    def forward(
        self,
        base_utility_features: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        changed: torch.Tensor,
        *,
        disabled_groups: Sequence[str] = (),
    ) -> torch.Tensor:
        if base_utility_features.shape[1] != GATE_INPUT_CHANNELS:
            raise ValueError("base utility feature contract changed")
        disabled = set(disabled_groups)
        if disabled and not self.allow_combination:
            raise ValueError("leave-one-module-out is only valid for combination")
        rows = [base_utility_features]
        for name in self.evidence_groups:
            value = evidence[name]
            expected = EVIDENCE_CHANNELS[name]
            if value.shape[:2] != (base_utility_features.shape[0], expected):
                raise ValueError(f"{name} must provide {expected} channels")
            rows.append(torch.zeros_like(value) if name in disabled else value)
        inputs = torch.cat(rows, dim=1)
        return torch.sigmoid(self.network(inputs.float())) * changed.float()

    def manifest(self) -> dict[str, object]:
        return {
            "model": "IsolatedEvidenceGate",
            "base_channels": GATE_INPUT_CHANNELS,
            "evidence_groups": list(self.evidence_groups),
            "evidence_channels": {name: EVIDENCE_CHANNELS[name] for name in self.evidence_groups},
            "input_channels": self.input_channels,
            "allow_combination": self.allow_combination,
        }

