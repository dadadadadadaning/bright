"""Isolated A1-LCA cross-modal learnable-superpixel model.

The parent A1 network is inherited directly so an A1 checkpoint keeps its
original key names.  Every new parameter lives under ``cmsp.`` and both LCA
and region feedback are zero-initialised residuals.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_ROI_LCA import ROILocalCorrelationResidual
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL

from .association import (
    candidate_map,
    fixed_gradient_magnitude,
    hard_pool_unpool,
    masked_local_softmax,
    soft_pool_unpool,
)
from .variants import VariantSpec, get_variant


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class CrossModalSuperpixelPlugin(nn.Module):
    """LCA followed by reliability-aware local region association."""

    def __init__(
        self,
        spec: VariantSpec,
        channels: int = 256,
        token_dim: int = 128,
        region_stride: int = 8,
        slico_compactness: float = 10.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.region_stride = int(region_stride)
        self.slico_compactness = float(slico_compactness)
        if self.region_stride < 2:
            raise ValueError("region_stride must be at least two")
        self.lca = ROILocalCorrelationResidual(
            channels=channels,
            match_channels=32,
            value_channels=64,
            window_size=3,
            confidence_gate=True,
            orientation_key=False,
        )
        self.rgb_structure = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Conv2d(64 * 4, 64, 1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.GELU(),
        )
        self.damage_hint = nn.Sequential(
            nn.Conv2d(4, 8, 1, bias=False),
            nn.GELU(),
        )
        # RGB structure (32), relation (64), edge (1), building prior (1), hint (8).
        self.association_head = nn.Sequential(
            nn.Conv2d(106, 96, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(96), 96),
            nn.GELU(),
            nn.Conv2d(96, 9, 1),
        )
        self.content_projection = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, token_dim, 1, bias=False),
            nn.GroupNorm(_groups(token_dim), token_dim),
            nn.GELU(),
        )
        self.region_update = nn.Sequential(
            nn.Conv2d(channels + token_dim, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.region_gain = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _relation(rgb_value: torch.Tensor, sar_value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (rgb_value, sar_value, (rgb_value - sar_value).abs(), rgb_value * sar_value),
            dim=1,
        )

    @staticmethod
    def _shift_without_wrap(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        output = torch.zeros_like(value)
        h, w = value.shape[-2:]
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        src_x0, src_x1 = max(0, -dx), min(w, w - dx)
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
        dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
        if src_y1 > src_y0 and src_x1 > src_x0:
            output[..., dst_y0:dst_y1, dst_x0:dst_x1] = value[..., src_y0:src_y1, src_x0:src_x1]
        return output

    def _slico_ids(
        self,
        pre_rgb: torch.Tensor,
        building_probability: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        """Fixed RGB SLICO split at the predicted building boundary."""
        try:
            from skimage.segmentation import slic
        except ImportError as exc:
            raise RuntimeError("c2_slico requires scikit-image") from exc
        resized = F.interpolate(pre_rgb.detach(), size=size, mode="bilinear", align_corners=False)
        images = resized.float().cpu().permute(0, 2, 3, 1).numpy()
        h, w = size
        target_regions = max(4, ((h + self.region_stride - 1) // self.region_stride)
                             * ((w + self.region_stride - 1) // self.region_stride))
        building_masks = F.interpolate(
            building_probability.detach(), size=size, mode="bilinear", align_corners=False
        ).cpu().numpy()[:, 0] >= 0.5
        results = []
        for image, building_mask in zip(images, building_masks):
            low = image.min(axis=(0, 1), keepdims=True)
            high = image.max(axis=(0, 1), keepdims=True)
            normalized = (image - low) / np.maximum(high - low, 1.0e-6)
            labels = slic(
                normalized,
                n_segments=target_regions,
                compactness=self.slico_compactness,
                start_label=0,
                channel_axis=-1,
                slic_zero=True,
                enforce_connectivity=True,
            )
            # SLICO is a local-region proposal, not a building instance.  The
            # predicted A1 boundary only prevents one region from mixing clear
            # foreground and background; GT never participates in partitioning.
            labels = labels.astype(np.int64) * 2 + building_mask.astype(np.int64)
            _, compact = np.unique(labels, return_inverse=True)
            results.append(torch.from_numpy(compact.reshape(h, w)).long())
        return torch.stack(results).to(device=pre_rgb.device)

    def forward(
        self,
        pre_rgb: torch.Tensor,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
        parent_fused: torch.Tensor,
        p_building: torch.Tensor,
        roi_weight: torch.Tensor,
        parent_damage_probability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.spec.use_lca:
            feature_roi = parent_fused * roi_weight
            return {
                "feature_roi": feature_roi,
                "parent_feature_roi": feature_roi,
                "region_gain": self.region_gain,
            }

        aligned = self.lca(rgb_feature, sar_feature, parent_fused, p_building)
        parent_feature_roi = aligned["feature"] * roi_weight
        output: Dict[str, torch.Tensor] = {
            **aligned,
            "parent_feature_roi": parent_feature_roi,
            "feature_roi": parent_feature_roi,
            "region_gain": self.region_gain,
        }
        if not self.spec.use_regions:
            return output

        rgb_value = aligned["rgb_alignment_value"]
        raw_sar = aligned["sar_alignment_value"]
        aligned_sar = aligned["aligned_sar_value"]
        confidence = aligned["alignment_confidence"]
        if not self.spec.use_alignment_confidence:
            confidence = torch.ones_like(confidence)

        if self.spec.sar_source == "none":
            chosen_sar = torch.zeros_like(aligned_sar)
            relation_gate = torch.zeros_like(confidence)
        elif self.spec.sar_source == "raw":
            chosen_sar = raw_sar
            relation_gate = torch.ones_like(confidence)
        else:
            chosen_sar = aligned_sar
            relation_gate = confidence
        if self.spec.shuffle_sar:
            shift_y = max(1, chosen_sar.shape[-2] // 4)
            shift_x = max(1, chosen_sar.shape[-1] // 4)
            chosen_sar = self._shift_without_wrap(chosen_sar, shift_y, shift_x)

        relation = self.relation_encoder(self._relation(rgb_value, chosen_sar)) * relation_gate
        rgb_structure = self.rgb_structure(rgb_feature)
        edge = fixed_gradient_magnitude(rgb_structure).clamp_max(10.0)
        building = F.interpolate(p_building.detach(), rgb_structure.shape[-2:], mode="bilinear", align_corners=False)
        if self.spec.use_damage_hint:
            if parent_damage_probability is None:
                raise ValueError("c8_damage_hint requires detached parent class probabilities")
            hint = self.damage_hint(parent_damage_probability.detach())
        else:
            hint = rgb_structure.new_zeros((rgb_structure.shape[0], 8, *rgb_structure.shape[-2:]))

        content = self.content_projection(torch.cat((rgb_value, chosen_sar * relation_gate, relation), dim=1))
        diagnostics: Dict[str, torch.Tensor] = {
            "region_content": content,
            "region_relation": relation,
            "region_confidence": confidence,
            "region_edge": edge,
        }
        h, w = content.shape[-2:]
        if self.spec.partition == "slico":
            hard_ids = self._slico_ids(pre_rgb, p_building, (h, w))
            pooled = hard_pool_unpool(content, hard_ids)
            diagnostics["hard_region_ids"] = hard_ids
            diagnostics["token_count"] = content.new_tensor(float(pooled["tokens"].shape[1]))
        else:
            logits = self.association_head(torch.cat((rgb_structure, relation, edge, building, hint), dim=1))
            ids, valid, gh, gw = candidate_map(h, w, self.region_stride, logits.device)
            probabilities = masked_local_softmax(logits, valid)
            pooled = soft_pool_unpool(content, probabilities, ids, gh * gw)
            diagnostics.update({
                "association_logits": logits,
                "association_probabilities": probabilities,
                "candidate_ids": ids,
                "candidate_valid": valid,
                "token_grid_h": content.new_tensor(float(gh)),
                "token_grid_w": content.new_tensor(float(gw)),
                "token_count": content.new_tensor(float(gh * gw)),
            })

        region_context = pooled["reconstruction"]
        update = self.region_update(torch.cat((parent_feature_roi, region_context), dim=1))
        residual_gate = building * confidence
        feature_roi = parent_feature_roi + torch.tanh(self.region_gain) * residual_gate * update
        output.update(diagnostics)
        output.update({
            "feature_roi": feature_roi,
            "region_tokens": pooled["tokens"],
            "region_mass": pooled["mass"],
            "region_context": region_context,
            "region_update": update,
            "region_residual_gate": residual_gate,
        })
        return output


class DamageFormerA1CrossModalSuperpixel(DamageFormer_S8_MICB_DSMIL):
    """A1 soft-ROI parent with one isolated ``cmsp`` plugin."""

    plugin_prefix = "cmsp."

    def __init__(self, variant: str, region_stride: int = 8) -> None:
        super().__init__(
            roi_alpha=0.1,
            detach_building_prior=True,
            token_dim=128,
            mil_residual_scale=0.0,
        )
        self.variant = variant
        self.variant_spec = get_variant(variant)
        self.cmsp = CrossModalSuperpixelPlugin(
            self.variant_spec,
            channels=256,
            token_dim=128,
            region_stride=region_stride,
        )

    def freeze_parent(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.plugin_prefix))

    def set_plugin_training(self) -> None:
        self.eval()
        self.cmsp.train()

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        return_aux: bool = False,
    ):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)
        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior
        parent_fused = self.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        parent_feature_roi = parent_fused * roi_weight
        parent_damage_logits = self.clf_2(parent_feature_roi)
        parent_probability = torch.softmax(parent_damage_logits.detach(), dim=1)

        plugin = self.cmsp(
            pre_rgb=pre_data,
            rgb_feature=rgb_feature,
            sar_feature=sar_feature,
            parent_fused=parent_fused,
            p_building=p_building,
            roi_weight=roi_weight,
            parent_damage_probability=parent_probability,
        )
        damage_logits_feature = self.clf_2(plugin["feature_roi"])
        output_loc = F.interpolate(loc_logits_feature, pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_damage = F.interpolate(damage_logits_feature, post_data.shape[-2:], mode="bilinear", align_corners=False)
        if not return_aux:
            return output_loc, output_damage
        aux = {
            "variant": self.variant,
            "p_building": p_building,
            "roi_weight": roi_weight,
            "rgb_feature": rgb_feature,
            "sar_feature": sar_feature,
            "parent_damage_logits_feature": parent_damage_logits,
            **plugin,
        }
        return output_loc, output_damage, aux

    def checkpoint_contract(self) -> Dict[str, object]:
        return {
            "protocol": "DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1",
            "parent": "P3 DamageFormer ablation A1 soft ROI",
            "variant": self.variant_spec.to_dict(),
            "plugin_prefix": self.plugin_prefix,
            "zero_initialised_lca": True,
            "zero_initialised_region_feedback": True,
            "flat_four_class_output": True,
            "test_accessed": False,
        }


__all__ = ["CrossModalSuperpixelPlugin", "DamageFormerA1CrossModalSuperpixel"]
