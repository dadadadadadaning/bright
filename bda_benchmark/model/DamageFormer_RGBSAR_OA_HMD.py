# -*- coding: utf-8 -*-
"""OA-HMD: optical-anchored heterogeneous multi-scale damage modelling.

This first controlled implementation intentionally exposes only C0 and C1.

* C0 is an exact frozen P3 guard.
* C1 adds one Stage-3 explicit heterogeneous-difference branch and modifies
  only the Intact-vs-Affected log odds through an exactly-zero-initialised
  residual.

The P3 localisation/damage path and the H6-A2 native-SAR feature path are kept
independent and frozen.  RGB and SAR are never stacked as a pseudo-time axis.
Stage2, Stage4 severity, ordinal supervision and FCA are deliberately absent;
they belong to later ablations only if C1 passes validation gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_H10_H3D import (
    ConvGNAct,
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


VALID_OA_HMD_VARIANTS = ("c0", "c1")


@dataclass(frozen=True)
class OAHMDVariant:
    name: str
    stage3_difference: bool


OA_HMD_VARIANTS: Mapping[str, OAHMDVariant] = {
    "c0": OAHMDVariant("OA-HMD-C0-Exact-P3-Guard", False),
    "c1": OAHMDVariant("OA-HMD-C1-Stage3-Explicit-Difference", True),
}


class Stage3ExplicitDifferenceHead(nn.Module):
    """Modality-preserving Stage3 relation head.

    Independent projections first place the modalities in a learned latent
    space.  The mixer then receives the two identities, their absolute
    difference and their elementwise product:

        [P_rgb(R3), P_sar(S3), |P_rgb(R3)-P_sar(S3)|,
         P_rgb(R3) * P_sar(S3)].

    This is not alignment, temporal Conv3D, a router, or a replacement expert.
    """

    def __init__(
        self,
        source_channels: int = 384,
        hidden_channels: int = 64,
        candidate_bound: float = 6.0,
    ) -> None:
        super().__init__()
        if candidate_bound <= 0:
            raise ValueError("candidate_bound must be positive")
        self.rgb_projection = ConvGNAct(source_channels, hidden_channels, 1)
        self.sar_projection = ConvGNAct(source_channels, hidden_channels, 1)
        self.mixer = nn.Sequential(
            ConvGNAct(4 * hidden_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3),
            ConvGNAct(hidden_channels, hidden_channels, 3),
        )
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        # The candidate must be non-zero so the zero transition receives a
        # first-step gradient. Safety comes from the transition, not by
        # zeroing both sides of the product.
        nn.init.kaiming_normal_(
            self.output.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.output.bias)
        self.candidate_bound = float(candidate_bound)

    def forward(
        self,
        rgb_stage3: torch.Tensor,
        sar_stage3: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb_stage3.shape[-2:] != sar_stage3.shape[-2:]:
            sar_stage3 = F.interpolate(
                sar_stage3,
                size=rgb_stage3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        rgb = self.rgb_projection(rgb_stage3)
        sar = self.sar_projection(sar_stage3)
        absolute = (rgb - sar).abs()
        product = rgb * sar
        feature = self.mixer(torch.cat((rgb, sar, absolute, product), dim=1))
        raw = self.output(feature)
        candidate = self.candidate_bound * torch.tanh(raw / self.candidate_bound)
        building = F.interpolate(
            building_probability,
            size=candidate.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).detach()
        candidate = candidate * building
        return candidate, {
            "oa_stage3_rgb_projected": rgb,
            "oa_stage3_sar_projected": sar,
            "oa_stage3_absolute_difference": absolute.mean(1, keepdim=True),
            "oa_stage3_product": product.mean(1, keepdim=True),
            "oa_stage3_feature": feature,
            "oa_building_support": building,
            "oa_occurrence_candidate_abs": candidate.abs(),
        }


class DamageFormerRGBSAROAHMD(DamageFormerRGBSARH6SDCR):
    """Frozen P3 anchor plus frozen native-SAR Stage3 evidence and safe C1."""

    def __init__(
        self,
        variant: str = "c1",
        pretrained_rgb: bool = False,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        hidden_channels: int = 64,
    ) -> None:
        variant = str(variant).lower()
        if variant not in VALID_OA_HMD_VARIANTS:
            raise ValueError(f"unknown OA-HMD variant {variant!r}")
        # A2 supplies native single-channel SAR Stage3 representations. Its
        # prediction is diagnostic only and never replaces the P3 output.
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.oa_hmd_config = OA_HMD_VARIANTS[variant]
        self.p3_anchor = DamageFormer_S8_MICB_DSMIL()
        if self.oa_hmd_config.stage3_difference:
            self.stage3_difference_head = Stage3ExplicitDifferenceHead(
                # Raw independent encoder Stage3 width. We intentionally do
                # not consume H6's common/unique or correspondence outputs.
                source_channels=384,
                hidden_channels=hidden_channels,
            )
            self.affected_transition = nn.Parameter(torch.tensor(0.0))

        self.register_buffer(
            "oa_imagenet_mean",
            torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=True,
        )
        self.register_buffer(
            "oa_imagenet_std",
            torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=True,
        )
        self.register_buffer(
            "oa_native_sar_mean",
            torch.tensor(56.64234075291341, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "oa_native_sar_std",
            torch.tensor(45.34399136641941, dtype=torch.float32),
            persistent=True,
        )

    def p3_sar_view(self, post_sar: torch.Tensor) -> torch.Tensor:
        """Reconstruct P3's historical three-channel display-SAR contract."""
        if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
            raise ValueError("OA-HMD expects [native] or [native,display] SAR")
        if post_sar.shape[1] == 2:
            display = post_sar[:, 1:2].float()
        else:
            # Compatibility fallback for old one-channel loaders. Strict C0
            # evaluation uses the two-channel H4 dataset contract.
            display = (
                post_sar[:, :1].float() * self.oa_native_sar_std
                + self.oa_native_sar_mean
            )
        repeated = display.repeat(1, 3, 1, 1)
        return ((repeated - self.oa_imagenet_mean) / self.oa_imagenet_std).to(
            post_sar.dtype
        )

    @staticmethod
    def _resize(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        # Both anchors are frozen by the trainer and evaluated without graph
        # construction. Only the compact Stage3 head and scalar transition are
        # trainable in C1.
        with torch.no_grad():
            p3_loc, p3_damage = self.p3_anchor(
                pre_rgb, self.p3_sar_view(post_sar)
            )
            rgb_stages = self.rgb_encoder(pre_rgb)
            sar_stages = self.sar_encoder(post_sar[:, :1])
            sar_aux = F.interpolate(
                self.sar_aux_head(sar_stages[-1]),
                size=p3_damage.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        base_occurrence, base_severity = factorize_building_logits(p3_damage)
        outputs: Dict[str, torch.Tensor] = {
            "output_loc": p3_loc,
            "output_dam": p3_damage,
            "p3_output_dam": p3_damage,
            "sar_aux_logits": sar_aux,
            "oa_base_affected_logits": base_occurrence,
            "oa_base_severity_logits": base_severity,
            # Reuse the already-audited H10 evaluator/loss contract.
            "h10_base_affected_logits": base_occurrence,
            "h10_base_severity_logits": base_severity,
            "h10_affected_logits": base_occurrence,
            "h10_severity_logits": base_severity,
        }
        if not self.oa_hmd_config.stage3_difference:
            if return_dict:
                return outputs
            return p3_loc, p3_damage

        building_probability = p3_loc.float().softmax(1)[:, 1:2]
        candidate_low, diagnostics = self.stage3_difference_head(
            rgb_stages[2],
            sar_stages[2],
            building_probability,
        )
        candidate = self._resize(candidate_low, p3_damage)
        transition = torch.tanh(self.affected_transition)
        occurrence_residual = transition * candidate
        updated = apply_hierarchical_residuals(
            p3_damage, occurrence_residual=occurrence_residual
        )
        affected, severity = factorize_building_logits(updated)
        outputs.update(diagnostics)
        outputs.update(
            {
                "output_dam": updated,
                "oa_affected_logits": affected,
                "oa_severity_logits": severity,
                "oa_affected_transition": transition.reshape(1, 1, 1, 1),
                "oa_occurrence_residual_abs": occurrence_residual.abs(),
                "h10_occurrence_transition": transition.reshape(1, 1, 1, 1),
                "h10_occurrence_residual_abs": occurrence_residual.abs(),
                "h10_affected_logits": affected,
                "h10_severity_logits": severity,
            }
        )
        if return_dict:
            return outputs
        return p3_loc, updated


def build_oa_hmd_damageformer(
    variant: str,
    pretrained_rgb: bool = False,
) -> DamageFormerRGBSAROAHMD:
    return DamageFormerRGBSAROAHMD(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_OA_HMD_VARIANTS",
    "OA_HMD_VARIANTS",
    "Stage3ExplicitDifferenceHead",
    "DamageFormerRGBSAROAHMD",
    "build_oa_hmd_damageformer",
]
