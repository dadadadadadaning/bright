from __future__ import annotations

import torch
from torch import nn


class SingleEvidenceVerifier(nn.Module):
    def __init__(self, input_channels: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.float()).squeeze(1)


def standardized_inputs(cache: dict, group: str) -> torch.Tensor:
    return torch.cat((cache["features"].float(), cache["evidence"][group].float()), dim=1)
