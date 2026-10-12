# -*- coding: utf-8 -*-
"""Training-only occurrence projection used by CMR-H1.

The module is never called by the P3 validation/test forward path.  Its final
layer is zero-initialised, so the initial projected geometry is exactly the
normalised frozen P3 Mean-Key geometry.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMRTrainingOnlyOccurrenceProjector(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.act(self.fc1(self.norm(feature.float()))))
        return F.normalize(feature.float() + residual, dim=-1, eps=1e-8)


__all__ = ["CMRTrainingOnlyOccurrenceProjector"]

