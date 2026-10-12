# -*- coding: utf-8 -*-
"""Isolated Stage-2 local Occurrence branch on the accepted P3.

The frozen P3 still owns localisation, Background-vs-Building mass and the
conditional Damaged-vs-Destroyed decision.  This module only learns a bounded
residual for the Intact-vs-Affected log-odds from higher-resolution Swin
Stage-2 RGB/SAR features.

No Memory, retrieval result, validation label, or Test-time support enters the
forward path.  A zero-initialised final layer makes Step 0 exactly P3 through a
difference-of-reconstructions update.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
)


def occurrence_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Return Affected-vs-Intact log-odds for B/I/D/X logits."""

    if logits.ndim != 4 or int(logits.shape[1]) != 4:
        raise ValueError("logits must have shape [B,4,H,W]")
    return torch.logsumexp(logits[:, 2:4].float(), dim=1, keepdim=True) - logits[:, 1:2].float()


def occurrence_only_update(
    base_logits: torch.Tensor,
    occurrence_delta: torch.Tensor,
) -> torch.Tensor:
    """Change I-vs-A odds while preserving Background, building mass and D/X.

    The update is expressed as ``base + (new_reconstruction - old_reconstruction)``.
    Consequently an exactly zero delta returns the original P3 tensor rather
    than a numerically reconstructed approximation.
    """

    if base_logits.ndim != 4 or int(base_logits.shape[1]) != 4:
        raise ValueError("base_logits must have shape [B,4,H,W]")
    if occurrence_delta.shape != base_logits[:, :1].shape:
        raise ValueError("occurrence_delta must have shape [B,1,H,W]")

    work = base_logits.float()
    building_mass = torch.logsumexp(work[:, 1:4], dim=1, keepdim=True)
    base_occurrence = occurrence_log_odds(work)
    severity_log_probability = F.log_softmax(work[:, 2:4], dim=1)

    def reconstruct(occurrence: torch.Tensor) -> torch.Tensor:
        log_intact = F.logsigmoid(-occurrence)
        log_affected = F.logsigmoid(occurrence)
        return torch.cat(
            (
                building_mass + log_intact,
                building_mass + log_affected + severity_log_probability[:, 0:1],
                building_mass + log_affected + severity_log_probability[:, 1:2],
            ),
            dim=1,
        )

    old_building = reconstruct(base_occurrence)
    new_building = reconstruct(base_occurrence + occurrence_delta.float())
    update = torch.zeros_like(work)
    update[:, 1:4] = new_building - old_building
    return (work + update).to(dtype=base_logits.dtype)


class Stage2LocalOccurrenceEncoder(nn.Module):
    """Dense 5x5-Stage2 local-patch encoder with independent modality stems."""

    def __init__(
        self,
        in_channels: int = 192,
        projection_channels: int = 64,
        hidden_channels: int = 128,
        maximum_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if projection_channels % 8:
            raise ValueError("projection_channels must be divisible by 8")
        if hidden_channels % 8:
            raise ValueError("hidden_channels must be divisible by 8")
        if maximum_delta <= 0:
            raise ValueError("maximum_delta must be positive")
        self.maximum_delta = float(maximum_delta)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(in_channels, projection_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, projection_channels),
            nn.GELU(),
        )
        self.sar_projection = nn.Sequential(
            nn.Conv2d(in_channels, projection_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, projection_channels),
            nn.GELU(),
        )
        relation_channels = projection_channels * 4
        self.local_patch_encoder = nn.Sequential(
            nn.Conv2d(relation_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, projection_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, projection_channels),
            nn.GELU(),
        )
        self.occurrence_head = nn.Conv2d(projection_channels, 1, kernel_size=1)
        nn.init.zeros_(self.occurrence_head.weight)
        nn.init.zeros_(self.occurrence_head.bias)

    def forward(
        self,
        rgb_stage2: torch.Tensor,
        sar_stage2: torch.Tensor,
        building_support: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb_stage2.shape != sar_stage2.shape:
            raise ValueError("RGB and SAR Stage2 tensors must have identical shapes")
        if rgb_stage2.ndim != 4 or int(rgb_stage2.shape[1]) != 192:
            raise ValueError("Stage2 tensors must have shape [B,192,H,W]")
        rgb = self.rgb_projection(rgb_stage2)
        sar = self.sar_projection(sar_stage2)
        relation = torch.cat((rgb, sar, torch.abs(rgb - sar), rgb * sar), dim=1)
        local_tokens = self.local_patch_encoder(relation)
        raw_delta = self.occurrence_head(local_tokens)
        support = F.interpolate(
            building_support.detach().float(),
            size=raw_delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        delta = self.maximum_delta * torch.tanh(raw_delta.float()) * support
        return delta, {
            "stage2_rgb": rgb,
            "stage2_sar": sar,
            "stage2_relation": relation,
            "stage2_local_tokens": local_tokens,
            "stage2_raw_delta": raw_delta,
            "stage2_support": support,
        }


class DamageFormerP3Stage2Occurrence(nn.Module):
    """Accepted P3 plus an isolated, zero-safe Stage2 Occurrence branch."""

    def __init__(
        self,
        *,
        maximum_delta: float = 2.0,
        projection_channels: int = 64,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.p3 = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0")
        self.occurrence_encoder = Stage2LocalOccurrenceEncoder(
            projection_channels=projection_channels,
            hidden_channels=hidden_channels,
            maximum_delta=maximum_delta,
        )
        for parameter in self.p3.parameters():
            parameter.requires_grad_(False)
        self.p3.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.p3.eval()
        self.occurrence_encoder.train(mode)
        return self

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def _frozen_p3_with_stage2(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the frozen P3 path once and retain native Stage2 maps."""

        with torch.no_grad():
            pre_maps = self.p3.encoder_1(pre_data)
            post_maps = self.p3.encoder_2(post_data)
            if len(pre_maps) != 5 or len(post_maps) != 5:
                raise RuntimeError("P3 Swin extractors must return patch+four stages")
            _, pre1, pre2, pre3, pre4 = pre_maps
            _, post1, post2, post3, post4 = post_maps
            pre1, pre2, pre3, pre4 = map(
                self._to_nchw, (pre1, pre2, pre3, pre4)
            )
            post1, post2, post3, post4 = map(
                self._to_nchw, (post1, post2, post3, post4)
            )
            pre_ms = self.p3._multi_scale_fuse(pre1, pre2, pre3, pre4)
            post_ms = self.p3._multi_scale_fuse(post1, post2, post3, post4)
            rgb_feature = self.p3.fusion_layer_1(pre_ms)
            sar_feature = self.p3.fusion_layer_2(post_ms)
            loc_logits_low = self.p3.clf_1(rgb_feature)
            building_probability = torch.softmax(loc_logits_low, dim=1)[:, 1:2]
            building_prior = (
                building_probability.detach()
                if self.p3.detach_building_prior
                else building_probability
            )
            roi_weight = self.p3.roi_alpha + (1.0 - self.p3.roi_alpha) * building_prior
            fused = self.p3.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
            base_damage_low = self.p3.clf_2(fused * roi_weight)
            output_loc = F.interpolate(
                loc_logits_low,
                size=pre_data.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            base_damage = F.interpolate(
                base_damage_low,
                size=post_data.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return output_loc, base_damage, {
            "rgb_stage2": pre2,
            "sar_stage2": post2,
            "building_probability": building_probability,
        }

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        output_loc, base_damage, base_aux = self._frozen_p3_with_stage2(
            pre_data, post_data
        )
        stage2_delta, occurrence_aux = self.occurrence_encoder(
            base_aux["rgb_stage2"],
            base_aux["sar_stage2"],
            base_aux["building_probability"],
        )
        full_delta = F.interpolate(
            stage2_delta,
            size=base_damage.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_damage = occurrence_only_update(base_damage, full_delta)
        if not return_aux:
            return output_loc, output_damage
        return output_loc, output_damage, {
            **base_aux,
            **occurrence_aux,
            "base_damage": base_damage,
            "base_occurrence": occurrence_log_odds(base_damage),
            "occurrence_delta": full_delta,
            "new_occurrence": occurrence_log_odds(output_damage),
            "inference_uses_memory": False,
        }


__all__ = [
    "DamageFormerP3Stage2Occurrence",
    "Stage2LocalOccurrenceEncoder",
    "occurrence_log_odds",
    "occurrence_only_update",
]
