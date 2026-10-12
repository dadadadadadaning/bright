# -*- coding: utf-8 -*-
"""H10-H3D: hierarchical damage occurrence and conditional severity.

The model is a controlled descendant of the locked H6-A2 native-SAR expert.
It does not replace the RGB localisation branch, the SAR encoder, or the A2
fusion stack.  Instead, it factorises the three building damage logits into

    intact versus affected (damaged + destroyed), and
    damaged versus destroyed given affected.

H2 and H3 add an exactly-zero-initialised residual only to the first decision.
H2 uses a parameter-matched 2-D difference head; H3 uses a lightweight
HeteCD-inspired 3-D spatio-temporal attention head at Stage 4.  H4 retains the
trained H3 occurrence branch and adds a second safe residual to conditional
severity.  There is no teacher, distillation, prototype memory, or auxiliary
alignment loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR


VALID_H10_VARIANTS = ("h0", "h1", "h2", "h3", "h4")


@dataclass(frozen=True)
class H10Variant:
    name: str
    hierarchical_readout: bool
    occurrence_head: str
    conditional_severity: bool


H10_VARIANTS: Mapping[str, H10Variant] = {
    "h0": H10Variant("H10-H0-Locked-H6A2", False, "none", False),
    "h1": H10Variant("H10-H1-Exact-Hierarchical-Reparameterisation", True, "none", False),
    "h2": H10Variant("H10-H2-Stage4-2D-Affected-Control", True, "2d", False),
    "h3": H10Variant("H10-H3-Lite3DSTA-Affected", True, "3d", False),
    "h4": H10Variant("H10-H4-Lite3DSTA-Conditional-Severity", True, "3d", True),
}


def factorize_building_logits(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact occurrence and conditional-severity log odds.

    Class order is background, intact, damaged, destroyed.  Background is not
    part of either conditional decision.
    """

    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("damage logits must have shape [B,4,H,W]")
    intact = logits[:, 1:2]
    damaged = logits[:, 2:3]
    destroyed = logits[:, 3:4]
    affected = torch.logsumexp(torch.cat([damaged, destroyed], dim=1), dim=1, keepdim=True)
    occurrence_logit = affected - intact
    severity_logit = destroyed - damaged
    return occurrence_logit, severity_logit


def apply_hierarchical_residuals(
    base_logits: torch.Tensor,
    occurrence_residual: torch.Tensor | None = None,
    severity_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Modify the two conditional decisions without disturbing background.

    Adding the same occurrence residual to damaged and destroyed changes only
    ``intact vs affected`` and preserves the damaged/destroyed odds exactly.
    The severity correction then redistributes the fixed affected mass between
    damaged and destroyed.  Subtracting identical log-sigmoid terms makes a
    zero residual exactly safe while keeping a first-step gradient.
    """

    if base_logits.ndim != 4 or base_logits.shape[1] != 4:
        raise ValueError("base_logits must have shape [B,4,H,W]")
    background = base_logits[:, 0:1]
    intact = base_logits[:, 1:2]
    damaged = base_logits[:, 2:3]
    destroyed = base_logits[:, 3:4]
    if occurrence_residual is not None:
        if occurrence_residual.shape != base_logits[:, :1].shape:
            raise ValueError("occurrence_residual has an incompatible shape")
        damaged = damaged + occurrence_residual
        destroyed = destroyed + occurrence_residual

    if severity_residual is not None:
        if severity_residual.shape != base_logits[:, :1].shape:
            raise ValueError("severity_residual has an incompatible shape")
        base_severity = destroyed - damaged
        updated_severity = base_severity + severity_residual
        damaged_shift = (
            F.logsigmoid(-updated_severity) - F.logsigmoid(-base_severity)
        )
        destroyed_shift = (
            F.logsigmoid(updated_severity) - F.logsigmoid(base_severity)
        )
        damaged = damaged + damaged_shift
        destroyed = destroyed + destroyed_shift
    return torch.cat([background, intact, damaged, destroyed], dim=1)


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        group_count = min(8, out_channels)
        while out_channels % group_count:
            group_count -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
        )


class Stage4Projection(nn.Module):
    """Separate RGB/SAR projections; shallow modality parameters are not shared."""

    def __init__(self, source_channels: int = 160, hidden_channels: int = 64) -> None:
        super().__init__()
        self.rgb = ConvGNAct(source_channels, hidden_channels, 1)
        self.sar = ConvGNAct(source_channels, hidden_channels, 1)

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rgb(rgb), self.sar(sar)


class Matched2DAffectedHead(nn.Module):
    """Parameter-controlled 2-D alternative to Lite-3D-STA."""

    def __init__(self, source_channels: int = 160, hidden_channels: int = 64) -> None:
        super().__init__()
        self.projection = Stage4Projection(source_channels, hidden_channels)
        self.mixer = nn.Sequential(
            ConvGNAct(4 * hidden_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3),
            ConvGNAct(hidden_channels, hidden_channels, 3),
        )
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb, sar = self.projection(rgb, sar)
        feature = self.mixer(torch.cat([rgb, sar, sar - rgb, (sar - rgb).abs()], dim=1))
        residual = self.output(feature)
        building = F.interpolate(
            building_probability,
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).detach()
        residual = residual * building
        return residual, {
            "h10_affected_feature": feature,
            "h10_building_support": building,
            "h10_occurrence_candidate_abs": residual.abs().mean(1, keepdim=True),
        }


class Lite3DSTAAffectedHead(nn.Module):
    """Stage-4 lightweight 3-D spatio-temporal attention for affectedness.

    RGB and SAR are treated as a two-frame pseudo temporal sequence.  Three
    depthwise Conv3D branches collapse that axis with 3x3, 5x5 and 7x7 spatial
    receptive fields.  Channel and spatial attention refine the fused feature.
    This keeps the transferable idea of HeteCD's 3D-STA while avoiding its
    heavy four-scale block and its assumption of three SAR polarisations.
    """

    def __init__(self, source_channels: int = 160, hidden_channels: int = 64) -> None:
        super().__init__()
        self.projection = Stage4Projection(source_channels, hidden_channels)
        self.branches = nn.ModuleList(
            [
                nn.Conv3d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(2, kernel, kernel),
                    padding=(0, kernel // 2, kernel // 2),
                    groups=hidden_channels,
                    bias=False,
                )
                for kernel in (3, 5, 7)
            ]
        )
        self.branch_norms = nn.ModuleList(
            [nn.GroupNorm(8, hidden_channels) for _ in self.branches]
        )
        fused_channels = 3 * hidden_channels
        squeeze = max(hidden_channels // 4, 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, squeeze, 1),
            nn.GELU(),
            nn.Conv2d(squeeze, fused_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.mixer = nn.Sequential(
            ConvGNAct(fused_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3),
            ConvGNAct(hidden_channels, hidden_channels, 3),
        )
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb, sar = self.projection(rgb, sar)
        sequence = torch.stack([rgb, sar], dim=2)
        branches = []
        for convolution, normalization in zip(self.branches, self.branch_norms):
            value = convolution(sequence).squeeze(2)
            branches.append(F.gelu(normalization(value)))
        feature = torch.cat(branches, dim=1)
        channel_gate = self.channel_attention(feature)
        feature = feature * channel_gate
        spatial_descriptor = torch.cat(
            [feature.mean(1, keepdim=True), feature.amax(1, keepdim=True)], dim=1
        )
        spatial_gate = self.spatial_attention(spatial_descriptor)
        feature = self.mixer(feature * spatial_gate)
        residual = self.output(feature)
        building = F.interpolate(
            building_probability,
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).detach()
        residual = residual * building
        return residual, {
            "h10_affected_feature": feature,
            "h10_building_support": building,
            "h10_3d_channel_attention": channel_gate,
            "h10_3d_spatial_attention": spatial_gate,
            "h10_occurrence_candidate_abs": residual.abs().mean(1, keepdim=True),
        }


class ConditionalSeverityHead(nn.Module):
    """Light 2-D damaged/destroyed residual used only after H3 succeeds."""

    def __init__(self, source_channels: int = 160, hidden_channels: int = 64) -> None:
        super().__init__()
        self.projection = Stage4Projection(source_channels, hidden_channels)
        self.mixer = nn.Sequential(
            ConvGNAct(4 * hidden_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.kaiming_normal_(self.mixer[-1].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.mixer[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        affected_probability: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb, sar = self.projection(rgb, sar)
        residual = self.mixer(torch.cat([rgb, sar, sar - rgb, (sar - rgb).abs()], dim=1))
        support = F.interpolate(
            (affected_probability * building_probability).detach(),
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = residual * support
        return residual, {
            "h10_severity_support": support,
            "h10_severity_candidate_abs": residual.abs().mean(1, keepdim=True),
        }


class DamageFormerRGBSARH10H3D(DamageFormerRGBSARH6SDCR):
    """Safe hierarchical descendant of the locked H6-A2 model."""

    def __init__(
        self,
        variant: str = "h1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        hidden_channels: int = 64,
    ) -> None:
        variant = str(variant).lower()
        if variant not in VALID_H10_VARIANTS:
            raise ValueError(f"unknown H10 variant {variant!r}")
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h10_config = H10_VARIANTS[variant]
        stage4_channels = int(latent_channels[3])
        if self.h10_config.occurrence_head == "2d":
            self.affected_head = Matched2DAffectedHead(stage4_channels, hidden_channels)
            self.affected_transition = nn.Parameter(torch.tensor(0.0))
        elif self.h10_config.occurrence_head == "3d":
            self.affected_head = Lite3DSTAAffectedHead(stage4_channels, hidden_channels)
            self.affected_transition = nn.Parameter(torch.tensor(0.0))
        if self.h10_config.conditional_severity:
            self.severity_head = ConditionalSeverityHead(stage4_channels, hidden_channels)
            self.severity_transition = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _full_resolution(
        value: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
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
        outputs = super().forward(pre_rgb, post_sar, return_dict=True)
        base_logits = outputs["output_dam"]
        base_occurrence, base_severity = factorize_building_logits(base_logits)
        outputs.update(
            {
                "h10_base_affected_logits": base_occurrence,
                "h10_base_severity_logits": base_severity,
            }
        )

        # H0 and H1 are deliberately bitwise-safe readouts of H6-A2.  H1 adds
        # the conditional diagnostics but does not reconstruct the logits.
        if self.h10_config.occurrence_head == "none":
            outputs["h10_affected_logits"] = base_occurrence
            outputs["h10_severity_logits"] = base_severity
            if return_dict:
                return outputs
            return outputs["output_loc"], outputs["output_dam"]

        rgb_stage4 = outputs["stage4_rgb_common"]
        sar_stage4 = outputs["stage4_sar_common"]
        building_probability = outputs["output_loc"].float().softmax(1)[:, 1:2]
        candidate_low, diagnostics = self.affected_head(
            rgb_stage4, sar_stage4, building_probability
        )
        candidate = self._full_resolution(candidate_low, base_logits)
        occurrence_transition = torch.tanh(self.affected_transition)
        occurrence_residual = occurrence_transition * candidate
        updated_logits = apply_hierarchical_residuals(
            base_logits, occurrence_residual=occurrence_residual
        )
        affected_logits, severity_logits = factorize_building_logits(updated_logits)
        outputs.update(diagnostics)
        outputs.update(
            {
                "h10_occurrence_transition": occurrence_transition.reshape(1, 1, 1, 1),
                "h10_occurrence_residual_abs": occurrence_residual.abs(),
                "h10_affected_logits": affected_logits,
                "h10_severity_logits": severity_logits,
            }
        )

        if self.h10_config.conditional_severity:
            severity_candidate_low, severity_diagnostics = self.severity_head(
                rgb_stage4,
                sar_stage4,
                affected_logits.float().sigmoid(),
                building_probability,
            )
            severity_candidate = self._full_resolution(severity_candidate_low, base_logits)
            severity_transition = torch.tanh(self.severity_transition)
            severity_residual = severity_transition * severity_candidate
            updated_logits = apply_hierarchical_residuals(
                updated_logits, severity_residual=severity_residual
            )
            affected_logits, severity_logits = factorize_building_logits(updated_logits)
            outputs.update(severity_diagnostics)
            outputs.update(
                {
                    "h10_severity_transition": severity_transition.reshape(1, 1, 1, 1),
                    "h10_severity_residual_abs": severity_residual.abs(),
                    "h10_affected_logits": affected_logits,
                    "h10_severity_logits": severity_logits,
                }
            )

        outputs["output_dam"] = updated_logits
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_h10_h3d_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH10H3D:
    return DamageFormerRGBSARH10H3D(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H10_VARIANTS",
    "H10_VARIANTS",
    "factorize_building_logits",
    "apply_hierarchical_residuals",
    "Matched2DAffectedHead",
    "Lite3DSTAAffectedHead",
    "ConditionalSeverityHead",
    "DamageFormerRGBSARH10H3D",
    "build_h10_h3d_damageformer",
]
