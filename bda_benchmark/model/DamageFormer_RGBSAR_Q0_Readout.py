"""Building-conditioned frozen-SAR readout used by the Q0 diagnosis.

Q0 is deliberately not a new damage model.  It freezes a strict H6-A2 source,
reads only its native-SAR Stage4 tensor and trains one 1x1 three-class head.
The source model's official output is never modified.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_Heterogeneous import STAGE_CHANNELS


Q0_VARIANTS = ("q0_a0", "q0_a1", "q0_b0")


class Q0BuildingConditionedSARReadout(nn.Module):
    """A frozen H6-A2 Stage4 encoder plus a trainable 3-class linear readout."""

    def __init__(self, source: DamageFormerRGBSARH6SDCR, variant: str) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in Q0_VARIANTS:
            raise ValueError(f"Q0 variant must be one of {Q0_VARIANTS}, got {variant!r}")
        if str(getattr(source, "variant", "")).lower() != "a2":
            raise ValueError("Q0 requires the locked H6-A2 source")
        self.variant = variant
        self.source = source
        self.readout = nn.Conv2d(STAGE_CHANNELS[-1], 3, kernel_size=1, bias=True)
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.readout.bias)
        self.freeze_source()

    @property
    def uses_p3_mask(self) -> bool:
        return self.variant == "q0_b0"

    def freeze_source(self) -> None:
        self.source.eval()
        for parameter in self.source.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Prevent BatchNorm/dropout drift in the locked expert.
        self.source.eval()
        return self

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, value in self.named_parameters() if value.requires_grad]

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        if sar.ndim != 4 or sar.shape[1] < 1:
            raise ValueError("Q0 expects SAR shaped [B,C,H,W] with at least one channel")
        input_size = sar.shape[-2:]
        with torch.no_grad():
            stage4 = self.source.sar_encoder(sar[:, :1])[-1]
            source_sar_aux_low = self.source.sar_aux_head(stage4)
            if self.uses_p3_mask:
                rgb_stages = self.source.rgb_encoder(rgb)
                _, loc_low = self.source.p3_spatial_anchor(rgb_stages)
                building_probability_low = loc_low.float().softmax(1)[:, 1:2]
            else:
                building_probability_low = stage4.new_ones(
                    (stage4.shape[0], 1, *stage4.shape[-2:])
                )
        logits_low = self.readout(stage4.detach())
        return {
            "logits_low": logits_low,
            "logits": F.interpolate(
                logits_low, input_size, mode="bilinear", align_corners=False
            ),
            "building_probability": F.interpolate(
                building_probability_low,
                input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "stage4": stage4.detach(),
            "source_sar_aux_logits": F.interpolate(
                source_sar_aux_low,
                input_size,
                mode="bilinear",
                align_corners=False,
            ).detach(),
        }


def assert_q0_freeze_contract(model: Q0BuildingConditionedSARReadout) -> None:
    trainable = model.trainable_parameter_names()
    if not trainable or not all(name.startswith("readout.") for name in trainable):
        raise RuntimeError(f"Q0 freeze contract violated; trainable={trainable}")
    leaked = [name for name, value in model.source.named_parameters() if value.requires_grad]
    if leaked:
        raise RuntimeError(f"locked H6-A2 parameters are trainable: {leaked[:8]}")


__all__ = [
    "Q0_VARIANTS",
    "Q0BuildingConditionedSARReadout",
    "assert_q0_freeze_contract",
]
