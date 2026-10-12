from __future__ import annotations

from typing import Dict

import torch

from .base import FrozenFeatureSource, freeze_module


class H6NativeSARSource(FrozenFeatureSource):
    """Native one-channel SAR pyramid from a locked H6-A2 model."""

    output_channels = {"stage1": 96, "stage2": 192, "stage3": 384, "stage4": 768}

    def __init__(self, h6_model) -> None:
        super().__init__()
        self.encoder = freeze_module(h6_model.sar_encoder)

    @torch.no_grad()
    def forward(self, post_sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        native = post_sar[:, :1]
        stages = self.encoder(native)
        if len(stages) != 4:
            raise RuntimeError(f"H6 SAR encoder returned {len(stages)} stages, expected 4")
        return {f"stage{i}": value for i, value in enumerate(stages, start=1)}
