# -*- coding: utf-8 -*-
"""P5 with an in-network occurrence/severity hierarchical decoder.

The P5 asymmetric RGB/native-SAR encoders and DCMF fusion are retained.  The
new decoder factorises building damage into intact-vs-affected and, only inside
the affected mass, damaged-vs-destroyed.  Both transitions start at exact zero,
so migration from a flat P3/P5 checkpoint is prediction preserving.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_S8_MICB_DSMIL import BuildingBagSpec, build_predicted_bag_specs
from .DamageFormer_S8_P5_DCMF import DamageFormer_S8_P5_DCMF


def factorize_damage_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("damage logits must be [B,4,H,W]")
    affected = torch.logsumexp(logits[:, 2:4], dim=1, keepdim=True)
    occurrence = affected - logits[:, 1:2]
    severity = logits[:, 3:4] - logits[:, 2:3]
    return occurrence, severity


def apply_hierarchical_change(
    base: torch.Tensor,
    occurrence_delta: torch.Tensor,
    severity_delta: torch.Tensor,
) -> torch.Tensor:
    """Change two conditional decisions while preserving their invariants."""
    if occurrence_delta.shape != base[:, :1].shape:
        raise ValueError("occurrence delta shape mismatch")
    if severity_delta.shape != base[:, :1].shape:
        raise ValueError("severity delta shape mismatch")
    background, intact, damaged, destroyed = base.split(1, dim=1)

    # Equal affected shift: damaged/destroyed odds are unchanged.
    damaged = damaged + occurrence_delta
    destroyed = destroyed + occurrence_delta

    # Conditional severity redistribution: total affected log-mass is fixed.
    old_severity = destroyed - damaged
    new_severity = old_severity + severity_delta
    damaged = damaged + F.logsigmoid(-new_severity) - F.logsigmoid(-old_severity)
    destroyed = destroyed + F.logsigmoid(new_severity) - F.logsigmoid(old_severity)
    candidate = torch.cat((background, intact, damaged, destroyed), dim=1)

    # The conditional formula is mathematically identical when both deltas are
    # zero, but sigmoid/log-sigmoid reconstruction can still introduce a few
    # floating-point ULPs.  Preserve the exact parent value at those pixels
    # with a straight-through expression: ``candidate-candidate.detach()`` is
    # exactly zero in the forward pass while retaining candidate's gradient.
    zero_delta = (occurrence_delta == 0) & (severity_delta == 0)
    exact_identity_with_gradient = base + (candidate - candidate.detach())
    return torch.where(zero_delta.expand_as(base), exact_identity_with_gradient, candidate)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class HierarchicalChangeDecoder(nn.Module):
    """Decode occurrence and severity from fused multi-modal evidence."""

    def __init__(self, channels: int = 256, hidden: int = 96) -> None:
        super().__init__()
        # rgb, sar, fused, and an explicit cross-modal disagreement map.
        self.evidence = nn.Sequential(
            ConvGNAct(4 * channels, hidden, 1),
            ConvGNAct(hidden, hidden, 3),
        )
        self.occurrence_head = nn.Sequential(
            ConvGNAct(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1)
        )
        self.severity_head = nn.Sequential(
            ConvGNAct(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1)
        )
        self.occurrence_transition = nn.Parameter(torch.tensor(0.0))
        self.severity_transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        base_logits: torch.Tensor,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        fused: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        evidence = self.evidence(torch.cat((rgb, sar, fused, (sar - rgb).abs()), 1))
        support = building_probability.detach()
        occurrence_candidate = self.occurrence_head(evidence) * support
        occurrence_delta = torch.tanh(self.occurrence_transition) * occurrence_candidate

        # Severity is supported by the current soft affected probability, never
        # by a hard threshold or a pseudo-instance parameter.
        base_occurrence, _ = factorize_damage_logits(base_logits)
        affected_support = (base_occurrence + occurrence_delta).sigmoid().detach()
        severity_candidate = self.severity_head(evidence) * support * affected_support
        severity_delta = torch.tanh(self.severity_transition) * severity_candidate
        output = apply_hierarchical_change(base_logits, occurrence_delta, severity_delta)
        occurrence, severity = factorize_damage_logits(output)
        return output, {
            "hierarchy_evidence": evidence,
            "occurrence_logits": occurrence,
            "severity_logits": severity,
            "occurrence_delta": occurrence_delta,
            "severity_delta": severity_delta,
            "occurrence_transition": torch.tanh(self.occurrence_transition),
            "severity_transition": torch.tanh(self.severity_transition),
        }


class DamageFormer_S8_P5_Hierarchical(DamageFormer_S8_P5_DCMF):
    """Model-centric P5: asymmetric encoding, DCMF, hierarchical decoding."""

    def __init__(self, hierarchy_hidden: int = 96, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hierarchical_decoder = HierarchicalChangeDecoder(256, hierarchy_hidden)

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        bag_specs: Optional[Sequence[BuildingBagSpec]] = None,
        return_aux: bool = False,
    ):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)
        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * prior

        fused_feature, fusion_aux = self.fusion_layer_3(rgb_feature, sar_feature)
        fused_roi = fused_feature * roi_weight
        flat_logits = self.clf_2(fused_roi)
        damage_logits_feature, hierarchy_aux = self.hierarchical_decoder(
            flat_logits, rgb_feature, sar_feature, fused_roi, p_building
        )

        active_specs: Sequence[BuildingBagSpec] = bag_specs or []
        if not active_specs and self.mil_residual_scale > 0:
            active_specs = build_predicted_bag_specs(
                loc_logits_feature, min_area_feature=self.predicted_min_area_feature
            )
        bag_result = None
        if active_specs:
            bag_result = self.micb_dsmil(
                rgb_feature=rgb_feature,
                sar_feature=sar_feature,
                fused_feature=fused_roi,
                bag_specs=active_specs,
            )

        output_loc = F.interpolate(loc_logits_feature, pre_data.shape[-2:],
                                   mode="bilinear", align_corners=False)
        output_damage = F.interpolate(damage_logits_feature, post_data.shape[-2:],
                                      mode="bilinear", align_corners=False)
        if not return_aux:
            return output_loc, output_damage
        auxiliary: Dict[str, object] = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "rgb_feature": rgb_feature,
            "sar_feature": sar_feature,
            "fused_roi": fused_roi,
            "flat_damage_logits": flat_logits,
            "bag_result": bag_result,
            "bag_count": 0 if bag_result is None else int(bag_result["bag_count"]),
            "mil_residual_scale": self.mil_residual_scale,
            "prototype_bypassed": True,
            **hierarchy_aux,
        }
        auxiliary.update(fusion_aux)
        return output_loc, output_damage, auxiliary

    def experiment_metadata(self) -> Dict[str, object]:
        metadata = super().experiment_metadata()
        metadata.update({
            "family": "S8-P5-H hierarchical DCMF",
            "decoder": "soft_occurrence_then_conditional_severity",
            "pseudo_instance_required": False,
            "zero_transition_initialization": True,
        })
        return metadata


__all__ = [
    "DamageFormer_S8_P5_Hierarchical",
    "HierarchicalChangeDecoder",
    "apply_hierarchical_change",
    "factorize_damage_logits",
]
