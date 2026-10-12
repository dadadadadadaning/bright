"""D0-compatible building-ROI local correlation alignment ablations.

The parent pixel path is the original DamageFormer control.  Every added
branch is injected through a zero-initialised residual, so all variants are
exactly equal to the supplied D0 checkpoint before optimisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_Heterogeneous import OriginalDamageFormerControl


@dataclass(frozen=True)
class LCAVariant:
    window_size: int
    confidence_gate: bool
    orientation_key: bool
    title: str


LCA_VARIANTS: Mapping[str, LCAVariant] = {
    "a2_lca3": LCAVariant(3, False, False, "ROI + 3x3 local correlation"),
    "a3_conf3": LCAVariant(3, True, False, "A2 + entropy confidence"),
    "a4_orient3": LCAVariant(3, True, True, "A3 + gradient/orientation key"),
    "a5_conf5": LCAVariant(5, True, False, "A3 with a 5x5 window"),
}


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _fixed_gradient(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    weight = kernel.to(device=x.device, dtype=x.dtype).expand(x.shape[1], 1, 3, 3)
    return F.conv2d(x, weight, padding=1, groups=x.shape[1])


class ROILocalCorrelationResidual(nn.Module):
    """RGB-anchored soft lookup of nearby SAR features inside a building ROI."""

    def __init__(
        self,
        channels: int = 256,
        match_channels: int = 32,
        value_channels: int = 64,
        window_size: int = 3,
        confidence_gate: bool = True,
        orientation_key: bool = False,
        roi_radius: int = 2,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = int(window_size)
        self.confidence_gate = bool(confidence_gate)
        self.orientation_key = bool(orientation_key)
        self.roi_radius = int(roi_radius)
        self.temperature = float(temperature)
        self.query = nn.Conv2d(channels, match_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, match_channels, 1, bias=False)
        self.rgb_value = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, value_channels, 1, bias=False)
        key_multiplier = 3 if orientation_key else 1
        self.relation = nn.Sequential(
            nn.Conv2d(value_channels * 4, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        # Kept for checkpoint diagnostics; keys have match_channels*key_multiplier channels.
        self.key_channels = match_channels * key_multiplier
        self.gain = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "scharr_x",
            torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
            .div(32.0)[None, None],
            persistent=False,
        )
        self.register_buffer("scharr_y", self.scharr_x.transpose(-1, -2).contiguous(), persistent=False)
        radius = window_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.register_buffer(
            "offset_table", torch.stack((xx.flatten(), yy.flatten()), dim=1).float(),
            persistent=False,
        )

    def _matching_key(self, projected: torch.Tensor) -> torch.Tensor:
        if not self.orientation_key:
            return F.normalize(projected.float(), dim=1)
        # Absolute directional responses are more tolerant to contrast reversal
        # between optical and SAR than signed raw gradients.
        gx = _fixed_gradient(projected.float(), self.scharr_x).abs()
        gy = _fixed_gradient(projected.float(), self.scharr_y).abs()
        return F.normalize(torch.cat((projected.float(), gx, gy), dim=1), dim=1)

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        parent_fused: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, _, height, width = rgb.shape
        neighbours = self.window_size ** 2
        locations = height * width
        query = self._matching_key(self.query(rgb)).flatten(2)
        key = self._matching_key(self.key(sar))
        key_patches = F.unfold(
            key, self.window_size, padding=self.window_size // 2
        ).view(batch, key.shape[1], neighbours, locations)
        scores = torch.einsum("bcl,bckl->bkl", query, key_patches) / self.temperature
        valid = F.unfold(
            torch.ones((batch, 1, height, width), device=rgb.device, dtype=scores.dtype),
            self.window_size, padding=self.window_size // 2,
        ).view(batch, neighbours, locations)
        weights = torch.softmax(scores.masked_fill(valid < 0.5, -1e4), dim=1)

        rgb_values = self.rgb_value(rgb).float()
        values = self.value(sar).float()
        value_patches = F.unfold(
            values, self.window_size, padding=self.window_size // 2
        ).view(batch, values.shape[1], neighbours, locations)
        matched = torch.einsum("bkl,bckl->bcl", weights, value_patches)
        matched = matched.view(batch, values.shape[1], height, width).to(rgb.dtype)

        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)
        entropy = entropy / valid.sum(1).clamp_min(2.0).log()
        entropy = entropy.view(batch, 1, height, width).to(rgb.dtype)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        if not self.confidence_gate:
            confidence = torch.ones_like(confidence)

        p_building = F.interpolate(
            building_probability, (height, width), mode="bilinear", align_corners=False
        )
        kernel = 2 * self.roi_radius + 1
        expanded_roi = F.max_pool2d(p_building, kernel, stride=1, padding=self.roi_radius)
        matched_float = matched.float()
        relation = torch.cat(
            (rgb_values, matched_float, (rgb_values - matched_float).abs(), rgb_values * matched_float),
            dim=1,
        ).to(rgb.dtype)
        update = self.relation(relation)
        residual_gate = expanded_roi.detach() * confidence
        feature = parent_fused + torch.tanh(self.gain) * residual_gate * update
        offset = torch.einsum("bkl,kd->bdl", weights, self.offset_table.to(weights))
        offset = offset.view(batch, 2, height, width).to(rgb.dtype)
        return {
            "feature": feature,
            # Exposed only for the downstream region-token audit.  These are
            # pre-classification content maps; no class probability or GT is
            # mixed into the retrieval representation.
            "rgb_alignment_value": rgb_values.to(rgb.dtype),
            "sar_alignment_value": values.to(rgb.dtype),
            "aligned_sar_value": matched,
            "alignment_confidence": confidence,
            "alignment_entropy": entropy,
            "alignment_offset": offset,
            "alignment_roi": expanded_roi,
            "alignment_update": update,
        }


class DamageFormerROILCA(OriginalDamageFormerControl):
    """Original DamageFormer plus one isolated ROI-LCA residual."""

    plugin_prefix = "roi_lca."

    def __init__(self, variant: str, pretrained: bool = True) -> None:
        if variant not in LCA_VARIANTS:
            raise ValueError(f"unknown LCA variant {variant!r}")
        super().__init__(pretrained=pretrained)
        self.variant = variant
        spec = LCA_VARIANTS[variant]
        self.roi_lca = ROILocalCorrelationResidual(
            window_size=spec.window_size,
            confidence_gate=spec.confidence_gate,
            orientation_key=spec.orientation_key,
        )

    def freeze_parent(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.plugin_prefix))

    def set_plugin_training(self) -> None:
        self.eval()
        self.roi_lca.train()

    def forward(self, pre_rgb: torch.Tensor, post_sar: torch.Tensor, return_dict: bool = False):
        size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)
        rgb = self.fusion_layer_1(self._original_multiscale(rgb_stages))
        sar = self.fusion_layer_2(self._original_multiscale(sar_stages))
        parent_fused = self.fusion_layer_3(torch.cat((rgb, sar), dim=1))
        loc_low = self.localization_head(rgb)
        building = torch.softmax(loc_low, dim=1)[:, 1:2]
        aligned = self.roi_lca(rgb, sar, parent_fused, building)
        damage_low = self.damage_head(aligned["feature"])
        outputs = {
            "output_loc": F.interpolate(loc_low, size=size, mode="bilinear", align_corners=False),
            "output_dam": F.interpolate(damage_low, size=size, mode="bilinear", align_corners=False),
            "parent_damage_logits": F.interpolate(
                self.damage_head(parent_fused), size=size, mode="bilinear", align_corners=False
            ),
            "localization_probability_low": building,
            **aligned,
        }
        return outputs if return_dict else (outputs["output_loc"], outputs["output_dam"])

    def checkpoint_contract(self) -> Dict[str, object]:
        spec = LCA_VARIANTS[self.variant]
        return {
            "family": "DAMAGEFORMER_ROI_LCA_V1",
            "variant": self.variant,
            "window_size": spec.window_size,
            "confidence_gate": spec.confidence_gate,
            "orientation_key": spec.orientation_key,
            "zero_initialised_residual": True,
            "flat_four_class_head": True,
        }


__all__ = ["DamageFormerROILCA", "LCA_VARIANTS", "LCAVariant", "ROILocalCorrelationResidual"]
