from __future__ import annotations

import torch
import torch.nn as nn


class DamagedSpecialist(nn.Module):
    """Binary I/D specialist.  It outputs a ranking/correction score, not a 4-class replacement."""
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
