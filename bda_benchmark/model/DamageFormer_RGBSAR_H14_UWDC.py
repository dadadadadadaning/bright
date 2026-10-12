# -*- coding: utf-8 -*-
"""H14-UWDC: unique-aware weak-damage correction for RGB--SAR BDA.

H13 showed that changing pixel grouping does not recover weak damage.  H14
therefore removes superpixels and reads both the common and modality-unique
features already produced by the locked H6-A2 expert.  It only edits the
Intact/Affected log-odds; Damaged/Destroyed odds remain exactly unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H10_H3D import (
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from .DamageFormer_RGBSAR_H12_DGB import (
    OCCURRENCE_CANDIDATE_BOUND,
    ConvGNAct,
    bound_occurrence_candidate,
    bounded_occurrence_transition,
)
from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR


STAGE_INDEX = {"stage1": 0, "stage2": 1, "stage3": 2, "stage4": 3}


@dataclass(frozen=True)
class H14Variant:
    name: str
    evidence_stages: Tuple[str, ...]
    error_conditioned_loss: bool


H14_VARIANTS: Mapping[str, H14Variant] = {
    "u0": H14Variant("H14-U0-Locked-H6A2", (), False),
    "u1": H14Variant("H14-U1-Stage2-Common-Unique", ("stage2",), False),
    "u2": H14Variant("H14-U2-Stage2-Error-Conditioned", ("stage2",), True),
    "u3": H14Variant(
        "H14-U3-Stage2-Stage3-Error-Conditioned",
        ("stage2", "stage3"),
        True,
    ),
}
VALID_H14_VARIANTS = tuple(H14_VARIANTS)


class CommonUniqueEvidenceStage(nn.Module):
    """Learn weak-damage evidence without assuming unique-channel alignment."""

    def __init__(self, source_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.rgb_common = ConvGNAct(source_channels, hidden_channels, 1)
        self.sar_common = ConvGNAct(source_channels, hidden_channels, 1)
        self.rgb_unique = ConvGNAct(source_channels, hidden_channels, 1)
        self.sar_unique = ConvGNAct(source_channels, hidden_channels, 1)
        # Unique features are concatenated separately.  Subtracting them would
        # incorrectly assume that their channels share a semantic coordinate.
        self.mixer = nn.Sequential(
            ConvGNAct(6 * hidden_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels),
            ConvGNAct(hidden_channels, hidden_channels, 1),
        )

    def forward(
        self,
        rgb_common: torch.Tensor,
        sar_common: torch.Tensor,
        rgb_unique: torch.Tensor,
        sar_unique: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rc = self.rgb_common(rgb_common.float())
        sc = self.sar_common(sar_common.float())
        ru = self.rgb_unique(rgb_unique.float())
        su = self.sar_unique(sar_unique.float())
        common_gap = (rc - sc).abs()
        common_product = rc * sc
        feature = self.mixer(
            torch.cat([rc, sc, common_gap, common_product, ru, su], dim=1)
        )
        return feature, {
            "common_gap_abs": common_gap.detach().mean(1, keepdim=True),
            "rgb_unique_abs": ru.detach().abs().mean(1, keepdim=True),
            "sar_unique_abs": su.detach().abs().mean(1, keepdim=True),
        }


class UniqueWeakDamagePyramid(nn.Module):
    def __init__(
        self,
        stages: Sequence[str],
        latent_channels: Sequence[int],
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stages = tuple(stages)
        self.stage_heads = nn.ModuleDict(
            {
                stage: CommonUniqueEvidenceStage(
                    int(latent_channels[STAGE_INDEX[stage]]), hidden_channels
                )
                for stage in self.stages
            }
        )
        self.fuse = nn.Sequential(
            ConvGNAct(hidden_channels * len(self.stages), hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels),
        )
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = []
        diagnostics: Dict[str, torch.Tensor] = {}
        target_size = None
        for stage in self.stages:
            feature, stage_diag = self.stage_heads[stage](
                outputs[f"{stage}_rgb_common"],
                outputs[f"{stage}_sar_common"],
                outputs[f"{stage}_rgb_unique"],
                outputs[f"{stage}_sar_unique"],
            )
            if target_size is None:
                target_size = feature.shape[-2:]
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            features.append(feature)
            for key, value in stage_diag.items():
                diagnostics[f"{stage}_{key}"] = value
        if not features:
            raise RuntimeError("H14 evidence pyramid has no active stage")
        fused = self.fuse(torch.cat(features, dim=1))
        candidate = self.output(fused)
        diagnostics["fused_feature_abs"] = fused.detach().abs().mean(1, keepdim=True)
        return candidate, diagnostics


class DamageFormerRGBSARH14UWDC(DamageFormerRGBSARH6SDCR):
    """Locked H6-A2 plus one common/unique occurrence correction."""

    def __init__(
        self,
        variant: str = "u1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        hidden_channels: int = 32,
    ) -> None:
        variant = str(variant).lower()
        if variant not in VALID_H14_VARIANTS:
            raise ValueError(f"unknown H14 variant {variant!r}")
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h14_config = H14_VARIANTS[variant]
        if self.h14_config.evidence_stages:
            self.weak_damage_pyramid = UniqueWeakDamagePyramid(
                self.h14_config.evidence_stages,
                latent_channels,
                hidden_channels,
            )
            self.affected_transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        with torch.no_grad():
            outputs = super().forward(pre_rgb, post_sar, return_dict=True)
        base_logits = outputs["output_dam"].float()
        base_occurrence, base_severity = factorize_building_logits(base_logits)
        outputs.update(
            {
                "h10_base_affected_logits": base_occurrence,
                "h10_base_severity_logits": base_severity,
                "h10_affected_logits": base_occurrence,
                "h10_severity_logits": base_severity,
            }
        )
        if not self.h14_config.evidence_stages:
            return outputs if return_dict else (outputs["output_loc"], base_logits)

        device_type = "cuda" if pre_rgb.is_cuda else pre_rgb.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            candidate_low, diagnostics = self.weak_damage_pyramid(outputs)
            candidate = F.interpolate(
                candidate_low,
                size=base_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            # Use the complete soft RGB building probability.  H13's eroded
            # binary support could not correct weak damage near roof edges.
            building_support = (
                outputs["output_loc"].float().softmax(1)[:, 1:2].detach()
            )
            bounded = bound_occurrence_candidate(
                candidate, limit=OCCURRENCE_CANDIDATE_BOUND
            )
            transition = bounded_occurrence_transition(self.affected_transition)
            occurrence_residual = transition * bounded * building_support
            if not torch.isfinite(occurrence_residual).all():
                raise FloatingPointError("non-finite H14 occurrence residual")
            updated_logits = apply_hierarchical_residuals(
                base_logits, occurrence_residual=occurrence_residual
            )
            affected, severity = factorize_building_logits(updated_logits)

        outputs.update(
            {f"h14_{key}": value for key, value in diagnostics.items()}
        )
        outputs.update(
            {
                "h14_occurrence_transition": transition.reshape(1, 1, 1, 1),
                "h14_occurrence_candidate_abs": candidate.detach().abs(),
                "h14_occurrence_residual_abs": occurrence_residual.abs(),
                "h14_building_support": building_support,
                "h10_affected_logits": affected,
                "h10_severity_logits": severity,
                "output_dam": updated_logits,
            }
        )
        return outputs if return_dict else (outputs["output_loc"], updated_logits)


def build_h14_uwdc_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH14UWDC:
    return DamageFormerRGBSARH14UWDC(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "H14_VARIANTS",
    "VALID_H14_VARIANTS",
    "CommonUniqueEvidenceStage",
    "UniqueWeakDamagePyramid",
    "DamageFormerRGBSARH14UWDC",
    "build_h14_uwdc_damageformer",
]
