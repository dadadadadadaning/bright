from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    def __init__(self, dim: int, classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiPrototypeTail(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128, prototypes_per_class: int = 2, temperature: float = 0.10):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(3, prototypes_per_class, embed_dim) * 0.02)
        self.temperature = float(temperature)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(x), dim=-1)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)

    def class_scores(self, h: torch.Tensor) -> torch.Tensor:
        p = self.normalized_prototypes()
        sim = torch.einsum("nd,ckd->nck", h, p)
        tau = max(self.temperature, 1e-4)
        return tau * torch.logsumexp(sim / tau, dim=-1) - tau * torch.log(torch.tensor(float(p.shape[1]), device=h.device))

    def forward(self, x: torch.Tensor):
        h = self.embed(x)
        return self.class_scores(h), h
