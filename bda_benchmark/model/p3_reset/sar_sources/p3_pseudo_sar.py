from __future__ import annotations

from typing import Dict

import torch

from .base import FrozenFeatureSource, freeze_module, raw_display_channel


class P3PseudoSARSource(FrozenFeatureSource):
    """P3's historical SAR-as-RGB encoder, retained as the B0 control."""

    output_channels = {"fused": 256}

    def __init__(self, p3_model) -> None:
        super().__init__()
        self.encoder = freeze_module(p3_model.encoder_2)
        self.owner = p3_model

    @staticmethod
    def to_p3_input(post_sar: torch.Tensor) -> torch.Tensor:
        raw = raw_display_channel(post_sar)
        pseudo = raw.repeat(1, 3, 1, 1)
        # Match H0's NumPy contract exactly: values stay in display 0--255.
        mean = pseudo.new_tensor((123.675, 116.28, 103.53))[None, :, None, None]
        std = pseudo.new_tensor((58.395, 57.12, 57.375))[None, :, None, None]
        return (pseudo - mean) / std

    @torch.no_grad()
    def forward(self, post_sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        pseudo = self.to_p3_input(post_sar)
        _, f1, f2, f3, f4 = self.encoder(pseudo)
        stages = tuple(map(self.owner._to_nchw, (f1, f2, f3, f4)))
        feature = self.owner.fusion_layer_2(self.owner._multi_scale_fuse(*stages))
        return {"fused": feature}
