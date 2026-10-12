# -*- coding: utf-8 -*-
"""H8-BCSC: building-conditioned cross-sensor calibration.

H8 is a controlled continuation of the locked H6-A2 native-SAR model.

* C0 keeps the H6-A2 model and state dictionary exactly unchanged.
* C1 replaces only the stage-3/4 sensor adapters with a building-conditioned
  refinement.  A zero-initialised blend makes its initial forward exactly A2.
* C2 adds a low-open, bounded spatial reliability gate to that refinement.

No ground-truth mask is consumed by the forward path.  Prototype weights are
formed exclusively from detached model probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import (
    DamageFormerRGBSARH6SDCR,
    SensorGapPrototypeAdapter,
)
from .DamageFormer_RGBSAR_Heterogeneous import ConvGNAct


VALID_H8_VARIANTS = ("c0", "c1", "c2")


@dataclass(frozen=True)
class H8Variant:
    name: str
    building_conditioned_calibration: bool
    reliability_gate: bool


H8_VARIANTS: Mapping[str, H8Variant] = {
    "c0": H8Variant("H8-C0-Locked-H6A2-Control", False, False),
    "c1": H8Variant("H8-C1-Building-Conditioned-Calibration", True, False),
    "c2": H8Variant("H8-C2-Calibrated-Reliable-Residual", True, True),
}


class BuildingConditionedSensorAdapter(SensorGapPrototypeAdapter):
    """Refine A2's nuisance prototype using confident intact buildings.

    The inherited A2 prototype uses background plus SAR-intact building
    confidence.  H8 additionally estimates a building-only intact prototype.
    ``prototype_blend`` starts at zero, so loading an A2 checkpoint produces
    the exact A2 output.  C2 spatially bounds the refinement to a maximum of
    ``maximum_gate``; C1 uses a unit gate for the clean calibration ablation.
    """

    def __init__(
        self,
        latent_channels: int,
        reliability_gate: bool = False,
        intact_floor: float = 0.60,
        maximum_blend: float = 0.50,
        maximum_gate: float = 0.25,
        gate_bias: float = -3.5,
    ) -> None:
        super().__init__(latent_channels, frequency_calibration=False)
        if not 0.0 < intact_floor < 1.0:
            raise ValueError("intact_floor must be in (0, 1)")
        if not 0.0 < maximum_blend <= 1.0:
            raise ValueError("maximum_blend must be in (0, 1]")
        if not 0.0 < maximum_gate <= 1.0:
            raise ValueError("maximum_gate must be in (0, 1]")
        self.reliability_gate_enabled = bool(reliability_gate)
        self.intact_floor = float(intact_floor)
        self.maximum_blend = float(maximum_blend)
        self.maximum_gate = float(maximum_gate)
        self.prototype_blend = nn.Parameter(torch.tensor(0.0))
        # Analysis-only transient context.  It is never part of a checkpoint
        # and remains disabled in training and ordinary validation/test.
        self._audit_oracle_intact_mask = None
        self._audit_forced_blend = None
        if self.reliability_gate_enabled:
            self.reliability_gate = nn.Sequential(
                ConvGNAct(4, 8, 3),
                nn.Conv2d(8, 1, 1, bias=True),
            )
            nn.init.zeros_(self.reliability_gate[-1].weight)
            nn.init.constant_(self.reliability_gate[-1].bias, float(gate_bias))

    def set_audit_context(
        self,
        oracle_intact_mask: torch.Tensor | None = None,
        forced_blend: float | None = None,
    ) -> None:
        self._audit_oracle_intact_mask = oracle_intact_mask
        self._audit_forced_blend = forced_blend

    def clear_audit_context(self) -> None:
        self._audit_oracle_intact_mask = None
        self._audit_forced_blend = None

    @staticmethod
    def _weighted_prototype(
        gap: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep the same dtype and reduction order as H6-A2.  This is required
        # for bitwise C1/A2 equality when prototype_blend is zero under AMP.
        weights = weights.to(gap.dtype)
        raw_mass = weights.sum((2, 3), keepdim=True)
        prototype = (gap * weights).sum((2, 3), keepdim=True) / raw_mass.clamp_min(1.0)
        return prototype, raw_mass

    def forward(
        self,
        joint: torch.Tensor,
        rgb_common: torch.Tensor,
        sar_common: torch.Tensor,
        rgb_unique: torch.Tensor,
        sar_unique: torch.Tensor,
        building_probability: torch.Tensor,
        sar_intact_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        building = self._resize(building_probability, rgb_common).detach()
        intact = self._resize(sar_intact_probability, rgb_common).detach()
        building = building.float().clamp(0.0, 1.0)
        intact = intact.float().clamp(0.0, 1.0)

        # This is the locked H6-A2 weighting rule.  Keeping it explicitly is
        # what makes prototype_blend==0 an exact A2 compatibility point.
        base_weights = ((1.0 - building) + building * intact).to(rgb_common.dtype)
        strict_intact = (
            (intact - self.intact_floor) / (1.0 - self.intact_floor)
        ).clamp(0.0, 1.0)
        building_weights = (building * strict_intact).to(rgb_common.dtype)
        if self._audit_oracle_intact_mask is not None:
            # Ground truth is accepted only through the explicitly named
            # audit context.  The production forward never sets this value.
            oracle = self._resize(
                self._audit_oracle_intact_mask, rgb_common
            ).detach().float().clamp(0.0, 1.0)
            building_weights = oracle.to(rgb_common.dtype)

        gap = sar_common - rgb_common
        base_prototype, base_mass = self._weighted_prototype(gap, base_weights)
        building_prototype, building_mass = self._weighted_prototype(
            gap, building_weights
        )
        building_support = building.sum((2, 3), keepdim=True).clamp_min(1.0)
        prototype_confidence = (
            building_mass.float() / building_support
        ).clamp(0.0, 1.0).to(gap.dtype)

        prototype_shift = building_prototype - base_prototype
        blend = self.maximum_blend * torch.tanh(self.prototype_blend)
        if self._audit_forced_blend is not None:
            forced = max(-1.0, min(1.0, float(self._audit_forced_blend)))
            blend = gap.new_tensor(forced)
        base_corrected = gap - base_prototype

        if self.reliability_gate_enabled:
            confidence_map = prototype_confidence.expand(
                -1, -1, *base_corrected.shape[-2:]
            )
            gate_input = torch.cat(
                [
                    building.to(gap.dtype),
                    intact.to(gap.dtype),
                    base_corrected.abs().mean(1, keepdim=True),
                    confidence_map,
                ],
                dim=1,
            )
            reliability = self.maximum_gate * torch.sigmoid(
                self.reliability_gate(gate_input)
            ) * building.to(gap.dtype)
        else:
            reliability = torch.ones_like(building, dtype=gap.dtype)

        calibrated = base_corrected - (
            reliability * blend * prototype_confidence * prototype_shift
        )
        residual = self.evidence(
            torch.cat(
                [calibrated, calibrated.abs(), rgb_unique, sar_unique], dim=1
            )
        )
        causal_transition = torch.tanh(self.causal_transition)
        output = joint + causal_transition * residual
        return output, {
            "sensor_gap_prototype": base_prototype,
            "sensor_gap_raw_abs": gap.abs().mean(1, keepdim=True),
            "sensor_gap_corrected_abs": calibrated.abs().mean(1, keepdim=True),
            "unchanged_confidence": base_weights,
            "unchanged_mass": base_mass,
            "causal_transition": causal_transition.reshape(1, 1, 1, 1),
            "low_frequency_abs": torch.zeros_like(building, dtype=gap.dtype),
            "frequency_gate": torch.zeros_like(building, dtype=gap.dtype),
            "frequency_transition": gap.new_zeros((1, 1, 1, 1)),
            "bcsc_building_prototype": building_prototype,
            "bcsc_prototype_shift_abs": prototype_shift.abs().mean(1, keepdim=True),
            "bcsc_prototype_confidence": prototype_confidence,
            "bcsc_building_mass": building_mass,
            "bcsc_prototype_blend": blend.reshape(1, 1, 1, 1),
            "bcsc_reliability_gate": reliability,
        }


class DamageFormerRGBSARH8BCSC(DamageFormerRGBSARH6SDCR):
    """H8 model with an exact locked H6-A2 control state contract."""

    def __init__(
        self,
        variant: str = "c1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
    ) -> None:
        variant = variant.lower()
        if variant not in VALID_H8_VARIANTS:
            raise ValueError(f"unknown H8 variant {variant!r}")
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h8_config = H8_VARIANTS[variant]
        if self.h8_config.building_conditioned_calibration:
            self.sensor_damage_adapters = nn.ModuleDict(
                {
                    stage_name: BuildingConditionedSensorAdapter(
                        latent_channels[int(stage_name[-1]) - 1],
                        reliability_gate=self.h8_config.reliability_gate,
                    )
                    for stage_name in self.h6_config.sensor_stages
                }
            )

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
        audit_oracle_intact_mask: torch.Tensor | None = None,
        audit_forced_blend: float | None = None,
    ):
        if audit_oracle_intact_mask is not None and self.training:
            raise RuntimeError("oracle prototype context is evaluation-only")
        adapters = [
            module
            for module in self.sensor_damage_adapters.values()
            if isinstance(module, BuildingConditionedSensorAdapter)
        ]
        if audit_oracle_intact_mask is not None and not adapters:
            raise RuntimeError("oracle prototype audit requires H8 C1/C2")
        try:
            for adapter in adapters:
                adapter.set_audit_context(
                    oracle_intact_mask=audit_oracle_intact_mask,
                    forced_blend=audit_forced_blend,
                )
            return super().forward(
                pre_rgb, post_sar, return_dict=return_dict
            )
        finally:
            for adapter in adapters:
                adapter.clear_audit_context()


def build_h8_bcsc_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH8BCSC:
    return DamageFormerRGBSARH8BCSC(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H8_VARIANTS",
    "H8_VARIANTS",
    "BuildingConditionedSensorAdapter",
    "DamageFormerRGBSARH8BCSC",
    "build_h8_bcsc_damageformer",
]
