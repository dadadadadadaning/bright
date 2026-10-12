# -*- coding: utf-8 -*-
"""P3-HR: safe hierarchical Occurrence--Severity residuals on accepted P3.

The accepted P3 remains frozen and supplies localisation, its four-class
prediction, and the mature deep RGB--SAR fusion.  HR1 only learns two bounded
residual logits from P3's fused ROI feature.  HR2 changes one variable: the
Occurrence residual reads an explicit Stage2+Stage3 RGB--SAR relation pyramid.
Severity still reads the accepted P3 fused feature.

HR3-O keeps the same soft reconstruction but trains only the Stage2+Stage3
Occurrence residual.  Its loss is region-level and explicitly protects P3's
already-correct Intact, Damaged, and Destroyed buildings.  Severity is exactly
zero in HR3-O and is not instantiated or optimised.

There is no hard router.  The output is reconstructed from soft conditional
probabilities and a difference-of-reconstructions update.  Zero residuals are
therefore bitwise identical to P3, so Step 0 is an exact safety guard.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
)


VALID_P3_HR_VARIANTS = ("hr0", "hr1", "hr1o", "hr1s", "hr2", "hr3o")


def occurrence_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Affected-vs-Intact log-odds for B/I/D/X logits."""

    if logits.ndim != 4 or int(logits.shape[1]) != 4:
        raise ValueError("logits must have shape [B,4,H,W]")
    return (
        torch.logsumexp(logits[:, 2:4].float(), dim=1, keepdim=True)
        - logits[:, 1:2].float()
    )


def severity_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Destroyed-vs-Damaged log-odds conditional on Affected."""

    if logits.ndim != 4 or int(logits.shape[1]) != 4:
        raise ValueError("logits must have shape [B,4,H,W]")
    return logits[:, 3:4].float() - logits[:, 2:3].float()


def hierarchical_residual_update(
    base_logits: torch.Tensor,
    occurrence_delta: torch.Tensor,
    severity_delta: torch.Tensor,
) -> torch.Tensor:
    """Update I/A and D/X odds while preserving BG and building mass.

    The old and new conditional states are reconstructed with the same
    operations.  When both residuals are exactly zero their difference is
    exactly zero, avoiding a numerically reconstructed approximation of P3.
    """

    if base_logits.ndim != 4 or int(base_logits.shape[1]) != 4:
        raise ValueError("base_logits must have shape [B,4,H,W]")
    expected = base_logits[:, :1].shape
    if occurrence_delta.shape != expected or severity_delta.shape != expected:
        raise ValueError("residuals must each have shape [B,1,H,W]")

    work = base_logits.float()
    building_mass = torch.logsumexp(work[:, 1:4], dim=1, keepdim=True)
    base_occurrence = occurrence_log_odds(work)
    base_severity = severity_log_odds(work)

    def reconstruct(
        occurrence: torch.Tensor,
        severity: torch.Tensor,
    ) -> torch.Tensor:
        log_intact = building_mass + F.logsigmoid(-occurrence)
        log_affected = building_mass + F.logsigmoid(occurrence)
        log_damaged = log_affected + F.logsigmoid(-severity)
        log_destroyed = log_affected + F.logsigmoid(severity)
        return torch.cat((log_intact, log_damaged, log_destroyed), dim=1)

    old_building = reconstruct(base_occurrence, base_severity)
    new_building = reconstruct(
        base_occurrence + occurrence_delta.float(),
        base_severity + severity_delta.float(),
    )
    update = torch.zeros_like(work)
    update[:, 1:4] = new_building - old_building
    return (work + update).to(dtype=base_logits.dtype)


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        norm_groups = min(8, out_channels)
        while out_channels % norm_groups:
            norm_groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )


class ZeroBoundedHead(nn.Module):
    """Zero-initialised bounded one-logit residual head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        maximum_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if maximum_delta <= 0:
            raise ValueError("maximum_delta must be positive")
        self.maximum_delta = float(maximum_delta)
        self.features = ConvGNAct(in_channels, hidden_channels, 3)
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.features(feature))
        support = F.interpolate(
            support.detach().float(),
            size=raw.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        return self.maximum_delta * torch.tanh(raw.float()) * support


class ScaleRelation(nn.Module):
    """Independent RGB/SAR projections followed by explicit relations."""

    def __init__(self, in_channels: int, channels: int = 64) -> None:
        super().__init__()
        self.rgb_projection = ConvGNAct(in_channels, channels, 1)
        self.sar_projection = ConvGNAct(in_channels, channels, 1)
        self.relation = nn.Sequential(
            ConvGNAct(4 * channels, 2 * channels, 1),
            ConvGNAct(2 * channels, channels, 3),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb.shape != sar.shape:
            raise ValueError("RGB/SAR scale features must have identical shapes")
        r = self.rgb_projection(rgb)
        s = self.sar_projection(sar)
        relation = self.relation(torch.cat((r, s, (r - s).abs(), r * s), dim=1))
        return relation, {"rgb": r, "sar": s}


class MultiReceptiveFieldBlock(nn.Module):
    """Lightweight local-to-context block with dilations 1, 2, and 3."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            ConvGNAct(
                channels,
                channels,
                3,
                groups=channels,
                dilation=dilation,
            )
            for dilation in (1, 2, 3)
        )
        self.fuse = nn.Sequential(
            ConvGNAct(3 * channels, 2 * channels, 1),
            ConvGNAct(2 * channels, channels, 3),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(feature) for branch in self.branches], dim=1))


class Stage23OccurrencePyramid(nn.Module):
    """HR2 Stage2-detail + Stage3-semantic Occurrence residual."""

    def __init__(
        self,
        channels: int = 64,
        maximum_delta: float = 2.0,
    ) -> None:
        super().__init__()
        self.stage2 = ScaleRelation(192, channels)
        self.stage3 = ScaleRelation(384, channels)
        self.merge = ConvGNAct(2 * channels, channels, 1)
        self.context = MultiReceptiveFieldBlock(channels)
        self.output = nn.Conv2d(channels, 1, 1)
        self.maximum_delta = float(maximum_delta)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rgb_stage2: torch.Tensor,
        sar_stage2: torch.Tensor,
        rgb_stage3: torch.Tensor,
        sar_stage3: torch.Tensor,
        building_support: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        relation2, projected2 = self.stage2(rgb_stage2, sar_stage2)
        relation3, projected3 = self.stage3(rgb_stage3, sar_stage3)
        relation3 = F.interpolate(
            relation3,
            size=relation2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        tokens = self.context(self.merge(torch.cat((relation2, relation3), dim=1)))
        raw = self.output(tokens)
        support = F.interpolate(
            building_support.detach().float(),
            size=raw.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        delta = self.maximum_delta * torch.tanh(raw.float()) * support
        return delta, {
            "hr_stage2_relation": relation2,
            "hr_stage3_relation": relation3,
            "hr_occurrence_tokens": tokens,
            "hr_stage2_rgb": projected2["rgb"],
            "hr_stage2_sar": projected2["sar"],
            "hr_stage3_rgb": projected3["rgb"],
            "hr_stage3_sar": projected3["sar"],
        }


class DamageFormerP3HierarchicalResidual(nn.Module):
    """Frozen accepted P3 plus isolated hierarchical residual modules."""

    def __init__(
        self,
        variant: str = "hr1",
        *,
        maximum_occurrence_delta: float = 2.0,
        maximum_severity_delta: float = 2.0,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in VALID_P3_HR_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_P3_HR_VARIANTS}")
        self.variant = variant
        self.p3 = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0")
        for parameter in self.p3.parameters():
            parameter.requires_grad_(False)

        if variant in ("hr2", "hr3o"):
            self.occurrence_head: nn.Module | None = Stage23OccurrencePyramid(
                maximum_delta=maximum_occurrence_delta
            )
        elif variant in ("hr1", "hr1o"):
            self.occurrence_head = ZeroBoundedHead(
                256, maximum_delta=maximum_occurrence_delta
            )
        else:
            self.occurrence_head = None
        self.severity_head: nn.Module | None = None
        if variant in ("hr1", "hr1s", "hr2"):
            self.severity_head = ZeroBoundedHead(
                256, maximum_delta=maximum_severity_delta
            )
        self.p3.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.p3.eval()
        if self.occurrence_head is not None:
            self.occurrence_head.train(mode)
        if self.severity_head is not None:
            self.severity_head.train(mode)
        return self

    @staticmethod
    def _to_nchw(feature: torch.Tensor) -> torch.Tensor:
        return feature.permute(0, 3, 1, 2).contiguous()

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        prefixes = ("occurrence_head.", "severity_head.")
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name.startswith(prefixes)
        }

    def load_trainable_state_dict(
        self,
        state: Dict[str, torch.Tensor],
    ) -> None:
        """Strictly restore only the isolated HR modules from a delta file."""

        expected = set(self.trainable_state_dict())
        supplied = set(state)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise RuntimeError(
                f"P3-HR delta contract mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)

    def _frozen_p3_features(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            pre_maps = self.p3.encoder_1(pre_data)
            post_maps = self.p3.encoder_2(post_data)
            if len(pre_maps) != 5 or len(post_maps) != 5:
                raise RuntimeError("P3 Swin extractors must return patch+four stages")
            pre_stages = [self._to_nchw(value) for value in pre_maps[1:5]]
            post_stages = [self._to_nchw(value) for value in post_maps[1:5]]
            pre_ms = self.p3._multi_scale_fuse(*pre_stages)
            post_ms = self.p3._multi_scale_fuse(*post_stages)
            rgb_feature = self.p3.fusion_layer_1(pre_ms)
            sar_feature = self.p3.fusion_layer_2(post_ms)
            loc_low = self.p3.clf_1(rgb_feature)
            building_probability = torch.softmax(loc_low, dim=1)[:, 1:2]
            building_prior = (
                building_probability.detach()
                if self.p3.detach_building_prior
                else building_probability
            )
            roi_weight = self.p3.roi_alpha + (1.0 - self.p3.roi_alpha) * building_prior
            fused_roi = self.p3.fusion_layer_3(
                torch.cat((rgb_feature, sar_feature), dim=1)
            ) * roi_weight
            base_low = self.p3.clf_2(fused_roi)
            output_loc = F.interpolate(
                loc_low,
                size=pre_data.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            base_logits = F.interpolate(
                base_low,
                size=post_data.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return output_loc, base_logits, {
            "rgb_stage2": pre_stages[1],
            "sar_stage2": post_stages[1],
            "rgb_stage3": pre_stages[2],
            "sar_stage3": post_stages[2],
            "fused_roi": fused_roi,
            "building_probability": building_probability,
        }

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        output_loc, base_logits, features = self._frozen_p3_features(
            pre_data, post_data
        )
        if self.variant == "hr0":
            output_logits = base_logits
            occurrence_delta = torch.zeros_like(base_logits[:, :1])
            severity_delta = torch.zeros_like(base_logits[:, :1])
            occurrence_aux: Dict[str, torch.Tensor] = {}
        else:
            if self.variant in ("hr2", "hr3o"):
                assert self.occurrence_head is not None
                occurrence_low, occurrence_aux = self.occurrence_head(
                    features["rgb_stage2"],
                    features["sar_stage2"],
                    features["rgb_stage3"],
                    features["sar_stage3"],
                    features["building_probability"],
                )
            elif self.occurrence_head is not None:
                occurrence_low = self.occurrence_head(
                    features["fused_roi"], features["building_probability"]
                )
                occurrence_aux = {}
            else:
                occurrence_low = None
                occurrence_aux = {}
            occurrence_delta = (
                F.interpolate(
                    occurrence_low,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if occurrence_low is not None
                else torch.zeros_like(base_logits[:, :1])
            )
            if self.severity_head is None:
                severity_delta = torch.zeros_like(base_logits[:, :1])
            else:
                severity_low = self.severity_head(
                    features["fused_roi"], features["building_probability"]
                )
                severity_delta = F.interpolate(
                    severity_low,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            output_logits = hierarchical_residual_update(
                base_logits, occurrence_delta, severity_delta
            )

        if not return_aux:
            return output_loc, output_logits
        return output_loc, output_logits, {
            **features,
            **occurrence_aux,
            "base_damage": base_logits,
            "output_damage": output_logits,
            "base_occurrence": occurrence_log_odds(base_logits),
            "new_occurrence": occurrence_log_odds(output_logits),
            "base_severity": severity_log_odds(base_logits),
            "new_severity": severity_log_odds(output_logits),
            "occurrence_delta": occurrence_delta,
            "severity_delta": severity_delta,
            "variant": self.variant,
            "inference_uses_memory": False,
        }


__all__ = [
    "DamageFormerP3HierarchicalResidual",
    "MultiReceptiveFieldBlock",
    "ScaleRelation",
    "Stage23OccurrencePyramid",
    "VALID_P3_HR_VARIANTS",
    "ZeroBoundedHead",
    "hierarchical_residual_update",
    "occurrence_log_odds",
    "severity_log_odds",
]
