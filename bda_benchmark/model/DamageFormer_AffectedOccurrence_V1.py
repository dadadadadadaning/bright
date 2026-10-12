"""Frozen-P3 Affected Occurrence Head (AO-H) experiment.

The module is intentionally isolated from the accepted P3 implementation.
It learns only an ``Intact/Background -> Affected`` residual and adds exactly
the same value to Damaged and Destroyed logits.  Therefore it cannot alter the
parent's Damaged-vs-Destroyed logit difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from .DamageFormer_ModelCentric_V1 import checkpoint_state
from .safe_affected_plugins import SafeAffectedResidual


@dataclass(frozen=True)
class OccurrenceVariant:
    boundary_feature: bool
    boundary_weight: bool
    small_damage_weight: bool
    sar_quality_gate: bool


VARIANTS: Mapping[str, OccurrenceVariant] = {
    "m1_occurrence": OccurrenceVariant(False, False, False, False),
    "m2_boundary": OccurrenceVariant(True, True, False, False),
    "m3_small_damage": OccurrenceVariant(True, True, True, False),
    "m4_sar_quality": OccurrenceVariant(True, True, True, True),
}


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class Stage2OccurrenceEvidence(nn.Module):
    """High-resolution RGB/SAR evidence with an optional SAR reliability gate.

    ``rgb_stage2`` and ``sar_stage2`` are the frozen modality-specific P3
    Stage-2 tensors.  The gate is continuous: noisy SAR is downweighted rather
    than rejected, so an optical structural cue can still reach the head.
    """

    def __init__(self, hidden: int, use_boundary: bool, use_sar_quality: bool) -> None:
        super().__init__()
        self.use_boundary = bool(use_boundary)
        self.use_sar_quality = bool(use_sar_quality)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(192, hidden, 1, bias=False), nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
        )
        self.sar_projection = nn.Sequential(
            nn.Conv2d(192, hidden, 1, bias=False), nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
        )
        # A conservative initial scale makes the first learned state close to
        # the ungated evidence path; the residual output itself starts at zero.
        self.quality_raw_scale = nn.Parameter(torch.tensor(-4.0))
        inputs = hidden * 4 + (1 if self.use_boundary else 0)
        self.fuse = nn.Sequential(
            nn.Conv2d(inputs, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    @staticmethod
    def _predicted_inner_boundary(p_building: torch.Tensor) -> torch.Tensor:
        """Prediction-only inner building boundary; no target is used here."""
        binary = (p_building.detach() >= 0.5).to(p_building.dtype)
        eroded = 1.0 - F.max_pool2d(1.0 - binary, 5, stride=1, padding=2)
        return (binary - eroded).clamp_(0.0, 1.0)

    def _quality(self, post_sar: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        value = F.interpolate(post_sar.float().mean(1, keepdim=True), size=size, mode="bilinear", align_corners=False)
        local_mean = F.avg_pool2d(value, 5, stride=1, padding=2)
        local_deviation = (value - local_mean).abs()
        normalized = local_deviation / local_deviation.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        if not self.use_sar_quality:
            return torch.ones_like(normalized)
        scale = F.softplus(self.quality_raw_scale)
        return torch.exp(-scale * normalized).clamp(0.0, 1.0)

    def forward(
        self,
        rgb_stage2: torch.Tensor,
        sar_stage2: torch.Tensor,
        post_sar: torch.Tensor,
        p_building: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb = F.interpolate(self.rgb_projection(rgb_stage2), size=size, mode="bilinear", align_corners=False)
        sar = F.interpolate(self.sar_projection(sar_stage2), size=size, mode="bilinear", align_corners=False)
        quality = self._quality(post_sar, size).to(sar.dtype)
        gated_sar = sar * quality
        terms = [rgb, gated_sar, (rgb - gated_sar).abs(), rgb * gated_sar]
        boundary = self._predicted_inner_boundary(p_building)
        if self.use_boundary:
            terms.append(boundary)
        return self.fuse(torch.cat(terms, dim=1)), {
            "rgb_stage2": rgb,
            "sar_stage2": sar,
            "sar_quality": quality,
            "predicted_boundary": boundary,
        }


class DamageFormerAffectedOccurrenceV1(DamageFormer_S8_MICB_DSMIL):
    """P3 wrapper with a frozen parent and trainable occurrence-only head."""

    plugin_prefixes = ("occurrence_evidence.", "occurrence_residual.")

    def __init__(self, variant: str = "m4_sar_quality", hidden: int = 64) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose {tuple(VARIANTS)}")
        super().__init__(mil_residual_scale=0.0)
        self.variant = variant
        self.variant_config = VARIANTS[variant]
        self.occurrence_evidence = Stage2OccurrenceEvidence(
            hidden, self.variant_config.boundary_feature, self.variant_config.sar_quality_gate
        )
        self.occurrence_residual = SafeAffectedResidual(maximum_shift=4.0)

    def load_parent_checkpoint(self, path: str) -> None:
        state = checkpoint_state(torch.load(path, map_location="cpu", weights_only=False))
        incompatible = self.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith(self.plugin_prefixes)]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f"P3 checkpoint mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}")

    def freeze_parent(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.plugin_prefixes))

    def set_plugin_training(self) -> None:
        self.eval()
        self.occurrence_evidence.train()
        self.occurrence_residual.train()

    def project_safe_parameters(self) -> None:
        self.occurrence_residual.project_parameters()

    def _encode_stage2_and_fuse(self, pre_data: torch.Tensor, post_data: torch.Tensor):
        """Repeat P3's exact encoder route while retaining native Stage-2 maps."""
        _, pre1, pre2, pre3, pre4 = self.encoder_1(pre_data)
        _, post1, post2, post3, post4 = self.encoder_2(post_data)
        pre1, pre2, pre3, pre4 = map(self._to_nchw, (pre1, pre2, pre3, pre4))
        post1, post2, post3, post4 = map(self._to_nchw, (post1, post2, post3, post4))
        rgb = self.fusion_layer_1(self._multi_scale_fuse(pre1, pre2, pre3, pre4))
        sar = self.fusion_layer_2(self._multi_scale_fuse(post1, post2, post3, post4))
        return rgb, sar, pre2, post2

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        rgb_feature, sar_feature, rgb_stage2, sar_stage2 = self._encode_stage2_and_fuse(pre_data, post_data)
        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * p_building.detach()
        fused = self.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        parent_logits_feature = self.clf_2(fused * roi_weight)
        occurrence_logits, diagnostics = self.occurrence_evidence(
            rgb_stage2, sar_stage2, post_data, p_building, parent_logits_feature.shape[-2:]
        )
        with torch.no_grad():
            parent_probability = parent_logits_feature.float().softmax(1)
            support = (0.10 + 0.90 * p_building.detach()) * parent_probability[:, 0:2].sum(1, keepdim=True)
        damage_logits_feature, delta = self.occurrence_residual(
            parent_logits_feature, occurrence_logits, support, torch.ones_like(support)
        )
        output_loc = F.interpolate(loc_logits_feature, pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_damage = F.interpolate(damage_logits_feature, post_data.shape[-2:], mode="bilinear", align_corners=False)
        if not return_aux:
            return output_loc, output_damage
        return output_loc, output_damage, {
            "parent_logits_feature": parent_logits_feature,
            "occurrence_logits": occurrence_logits,
            "affected_delta": delta,
            "p_building": p_building,
            "affected_gain": self.occurrence_residual.gain,
            **diagnostics,
        }


__all__ = ["DamageFormerAffectedOccurrenceV1", "OccurrenceVariant", "Stage2OccurrenceEvidence", "VARIANTS"]
