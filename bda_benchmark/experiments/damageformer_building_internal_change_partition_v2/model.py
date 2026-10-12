"""Building-Internal Structure-Constrained Change-Aware Partition V2.

Design principles
-----------------
1. Frozen A1 produces RGB/SAR features, building probability and parent damage logits.
2. Predicted building instances are a HARD geometric constraint. GT never enters forward.
3. RGB structure defines base association logits.
4. Aligned RGB-SAR change evidence can only refine those logits through a bounded,
   zero-initialised residual branch.
5. Alignment confidence/entropy are FEATURES, never repeated multiplicative gates.
6. Region tokens preserve weak evidence via mean/std/soft-peak statistics.
7. Stage-0/1 never writes regions into A1. Stage-2 uses a conservative uncertainty
   + learned utility gate with a zero-initialised residual gain.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_ROI_LCA import ROILocalCorrelationResidual
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL

from .association import (
    association_entropy,
    fixed_gradient_magnitude,
    masked_local_softmax,
    pool_scalar_mean,
    soft_peak_pool,
    soft_pool_mean_std,
    soft_unpool_tokens,
)
from .building_ops import adaptive_building_candidates, predicted_building_instances
from .variants import VariantSpec, get_variant


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class BuildingInternalChangePartition(nn.Module):
    """Predicted-building constrained soft partition and evidence encoder."""

    VALID_STAGES = ("structure", "change", "feedback")

    def __init__(
        self,
        spec: VariantSpec,
        channels: int = 256,
        token_dim: int = 128,
        target_region_area: int = 16,
        min_region_area: int = 4,
        max_regions_per_building: int = 16,
        building_threshold: float = 0.50,
        min_building_area: int = 8,
        change_alpha_max: float = 0.30,
        peak_temperature: float = 0.20,
        distance_weight: float = 4.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.token_dim = int(token_dim)
        self.target_region_area = int(target_region_area)
        self.min_region_area = int(min_region_area)
        self.max_regions_per_building = int(max_regions_per_building)
        self.building_threshold = float(building_threshold)
        self.min_building_area = int(min_building_area)
        self.change_alpha_max = float(change_alpha_max)
        self.peak_temperature = float(peak_temperature)
        self.distance_weight = float(distance_weight)
        self.stage = "structure"

        self.lca = ROILocalCorrelationResidual(
            channels=channels,
            match_channels=32,
            value_channels=64,
            window_size=3,
            confidence_gate=True,
            orientation_key=False,
        )
        # The LCA parent-feature residual is deliberately not used in V2.  The
        # matching branch only provides aligned SAR/evidence features.
        if hasattr(self.lca, "gain"):
            self.lca.gain.requires_grad_(False)

        self.rgb_structure = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
        )
        # 32 RGB structure + 1 edge + 2 building-local coordinates.
        self.structural_head = nn.Sequential(
            nn.Conv2d(35, 96, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(96), 96),
            nn.GELU(),
            nn.Conv2d(96, 9, 1),
        )

        # RGB value + aligned SAR + abs-diff + product + confidence + entropy.
        self.change_encoder = nn.Sequential(
            nn.Conv2d(64 * 4 + 2, 96, 1, bias=False),
            nn.GroupNorm(_groups(96), 96),
            nn.GELU(),
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.GELU(),
        )
        # relation64 + confidence + entropy + A1 uncertainty + local variation.
        change_head_in = 64 + 4
        self.change_head = nn.Sequential(
            nn.Conv2d(change_head_in, 96, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(96), 96),
            nn.GELU(),
            nn.Conv2d(96, 9, 1),
        )
        self.change_gate_head = nn.Sequential(
            nn.Conv2d(change_head_in, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        # Exact-zero residual right at initialisation while keeping a nonzero
        # derivative w.r.t. the scalar gain.
        self.change_gain = nn.Parameter(torch.zeros(()))

        self.evidence_score = nn.Sequential(
            nn.Conv2d(64 + 2, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

        # token = RGB mean64 + SAR mean64 + relation mean/std/peak(3*64)
        #       + confidence mean + entropy mean + mass/building-area.
        token_input_dim = 64 + 64 + 64 * 3 + 3
        self.token_encoder = nn.Sequential(
            nn.Linear(token_input_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.region_evidence_head = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # Intact / Damaged / Destroyed evidence.
        )

        self.region_update = nn.Sequential(
            nn.Conv2d(channels + token_dim, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        utility_in = 64 + token_dim + 4 + 2
        self.utility_gate = nn.Sequential(
            nn.Conv2d(utility_in, 128, 1, bias=False),
            nn.GroupNorm(_groups(128), 128),
            nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )
        self.feedback_gain = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _relation_input(rgb_value: torch.Tensor, sar_value: torch.Tensor) -> torch.Tensor:
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

    def configure_stage(self, stage: str) -> None:
        if stage not in self.VALID_STAGES:
            raise ValueError(f"stage must be one of {self.VALID_STAGES}, got {stage!r}")
        self.stage = stage

    def _alignment_entropy(self, aligned: Dict[str, torch.Tensor], confidence: torch.Tensor) -> torch.Tensor:
        entropy = aligned.get("alignment_entropy")
        if entropy is None:
            # Defensive compatibility fallback. Official current LCA exposes
            # alignment_entropy, but never fail just because an older copy does not.
            entropy = 1.0 - confidence
        if entropy.ndim == 1:
            entropy = entropy[:, None, None, None].expand_as(confidence)
        elif entropy.ndim == 2 and entropy.shape[1] == 1:
            entropy = entropy[:, :, None, None].expand_as(confidence)
        elif entropy.ndim == 3:
            entropy = entropy.unsqueeze(1)
        if entropy.shape[-2:] != confidence.shape[-2:]:
            entropy = F.interpolate(entropy.float(), size=confidence.shape[-2:], mode="bilinear", align_corners=False)
        return entropy.to(confidence.dtype)

    def forward(
        self,
        *,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
        parent_fused: torch.Tensor,
        parent_feature_roi: torch.Tensor,
        p_building: torch.Tensor,
        parent_damage_probability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # LCA is evidence-only in V2: aligned["feature"] is intentionally ignored.
        aligned = self.lca(rgb_feature, sar_feature, parent_fused, p_building)
        rgb_value = aligned["rgb_alignment_value"]
        aligned_sar = aligned["aligned_sar_value"]
        if self.spec.shuffle_sar:
            aligned_sar = self._shift_without_wrap(
                aligned_sar,
                max(1, aligned_sar.shape[-2] // 4),
                max(1, aligned_sar.shape[-1] // 4),
            )
        confidence = aligned["alignment_confidence"]
        if confidence.ndim == 3:
            confidence = confidence.unsqueeze(1)
        entropy = self._alignment_entropy(aligned, confidence)

        rgb_structure = self.rgb_structure(rgb_feature)
        h, w = rgb_structure.shape[-2:]
        edge = fixed_gradient_magnitude(rgb_structure).clamp_max(10.0)
        building_ids = predicted_building_instances(
            p_building,
            size=(h, w),
            threshold=self.building_threshold,
            min_area=self.min_building_area,
        )
        candidates = adaptive_building_candidates(
            building_ids,
            target_region_area=self.target_region_area,
            min_region_area=self.min_region_area,
            max_regions_per_building=self.max_regions_per_building,
        )
        local_xy = candidates["building_local_xy"].to(rgb_structure.dtype)
        distance2 = candidates["candidate_distance2"].to(rgb_structure.dtype).reshape(
            rgb_structure.shape[0], 9, h, w
        )

        structural_input = torch.cat((rgb_structure, edge, local_xy), dim=1)
        learned_structure = self.structural_head(structural_input)
        compact_bias = -self.distance_weight * distance2
        if self.spec.use_structure_logits:
            structural_logits = learned_structure + compact_bias
        else:
            # Change-only control still receives a weak geometric compactness bias
            # so it cannot make arbitrary nonlocal assignments.
            structural_logits = compact_bias

        confidence_feature = confidence.to(rgb_value.dtype)
        entropy_feature = entropy.to(rgb_value.dtype)
        relation_input = torch.cat(
            (self._relation_input(rgb_value, aligned_sar), confidence_feature, entropy_feature), dim=1
        )
        relation = self.change_encoder(relation_input)
        uncertainty = 1.0 - parent_damage_probability.max(dim=1, keepdim=True).values
        uncertainty_feature = uncertainty.to(relation.dtype)
        local_variation = fixed_gradient_magnitude(relation).clamp_max(10.0)
        change_context = torch.cat(
            (relation, confidence.to(relation.dtype), entropy.to(relation.dtype), uncertainty_feature, local_variation), dim=1
        )
        delta_logits = self.change_head(change_context)
        gate_probability = torch.sigmoid(self.change_gate_head(change_context))
        alpha = (
            self.change_alpha_max
            * torch.tanh(self.change_gain)
            * gate_probability
        )
        if not self.spec.use_change_refinement or self.stage == "structure":
            alpha = alpha * 0.0
        logits = structural_logits + alpha * delta_logits
        probabilities = masked_local_softmax(logits, candidates["candidate_valid"])

        token_count = int(candidates["token_count"].item())
        evidence_logit = self.evidence_score(
            torch.cat((relation, confidence.to(relation.dtype), entropy.to(relation.dtype)), dim=1)
        )
        rgb_stats = soft_pool_mean_std(rgb_value, probabilities, candidates["candidate_ids"], token_count)
        sar_stats = soft_pool_mean_std(aligned_sar, probabilities, candidates["candidate_ids"], token_count)
        rel_stats = soft_pool_mean_std(relation, probabilities, candidates["candidate_ids"], token_count)
        if self.spec.use_soft_peak:
            rel_peak = soft_peak_pool(
                relation,
                evidence_logit,
                probabilities,
                candidates["candidate_ids"],
                token_count,
                temperature=self.peak_temperature,
            )
        else:
            rel_peak = rel_stats["mean"].detach() * 0.0
        rel_std = rel_stats["std"] if self.spec.use_std else rel_stats["std"].detach() * 0.0
        conf_token = pool_scalar_mean(confidence, probabilities, candidates["candidate_ids"], token_count)
        entropy_token = pool_scalar_mean(entropy, probabilities, candidates["candidate_ids"], token_count)
        mass = rel_stats["mass"].float()
        building_area = candidates["token_building_area"].float().clamp_min(1.0)
        area_ratio = (mass / building_area).clamp(0.0, 1.0)

        token_input = torch.cat(
            (
                rgb_stats["mean"],
                sar_stats["mean"],
                rel_stats["mean"],
                rel_std,
                rel_peak,
                conf_token.to(rel_stats["mean"].dtype).unsqueeze(-1),
                entropy_token.to(rel_stats["mean"].dtype).unsqueeze(-1),
                area_ratio.to(rel_stats["mean"].dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        region_tokens = self.token_encoder(token_input)
        token_is_building = (candidates["token_building_ids"] > 0).to(region_tokens.dtype)
        region_tokens = region_tokens * token_is_building.unsqueeze(-1)
        region_logits = self.region_evidence_head(region_tokens)
        region_context = soft_unpool_tokens(
            region_tokens, probabilities, candidates["candidate_ids"], h, w
        )

        feature_roi = parent_feature_roi
        update = parent_feature_roi.new_zeros(parent_feature_roi.shape)
        utility = parent_feature_roi.new_zeros((parent_feature_roi.shape[0], 1, h, w))
        hard_building = (building_ids > 0).unsqueeze(1).to(parent_feature_roi.dtype)
        if self.stage == "feedback":
            update = self.region_update(
                torch.cat((parent_feature_roi, region_context.to(parent_feature_roi.dtype)), dim=1)
            )
            utility_input = torch.cat(
                (
                    relation.to(parent_feature_roi.dtype),
                    region_context.to(parent_feature_roi.dtype),
                    parent_damage_probability.to(parent_feature_roi.dtype),
                    confidence.to(parent_feature_roi.dtype),
                    entropy.to(parent_feature_roi.dtype),
                ),
                dim=1,
            )
            utility = torch.sigmoid(self.utility_gate(utility_input))
            if self.spec.use_uncertainty_feedback:
                uncertainty_gate = uncertainty.to(parent_feature_roi.dtype)
            else:
                uncertainty_gate = torch.ones_like(uncertainty, dtype=parent_feature_roi.dtype)
            residual_gate = hard_building * uncertainty_gate * utility
            feature_roi = (
                parent_feature_roi
                + torch.tanh(self.feedback_gain) * residual_gate * update
            )
        else:
            residual_gate = hard_building * 0.0

        assoc_ent = association_entropy(probabilities, candidates["candidate_valid"])
        return {
            **aligned,
            **candidates,
            "feature_roi": feature_roi,
            "parent_feature_roi": parent_feature_roi,
            "building_ids": building_ids,
            "rgb_structure": rgb_structure,
            "region_edge": edge,
            "region_relation": relation,
            "region_confidence": confidence,
            "region_alignment_entropy": entropy,
            "a1_uncertainty": uncertainty,
            "local_relation_variation": local_variation,
            "structural_logits": structural_logits,
            "change_delta_logits": delta_logits,
            "change_gate_probability": gate_probability,
            "change_alpha": alpha,
            "association_logits": logits,
            "association_probabilities": probabilities,
            "association_entropy": assoc_ent,
            "evidence_logit": evidence_logit,
            "rgb_token_mean": rgb_stats["mean"],
            "sar_token_mean": sar_stats["mean"],
            "relation_token_mean": rel_stats["mean"],
            "relation_token_std": rel_std,
            "relation_token_peak": rel_peak,
            "region_confidence_token": conf_token,
            "region_alignment_entropy_token": entropy_token,
            "region_tokens": region_tokens,
            "region_logits": region_logits,
            "region_mass": mass,
            "region_area_ratio": area_ratio,
            "region_context": region_context,
            "region_update": update,
            "utility_gate": utility,
            "region_residual_gate": residual_gate,
            "change_gain": self.change_gain,
            "feedback_gain": self.feedback_gain,
        }


class DamageFormerBuildingInternalChangePartitionV2(DamageFormer_S8_MICB_DSMIL):
    """Frozen A1 parent plus isolated ``biscp`` plugin."""

    plugin_prefix = "biscp."

    def __init__(
        self,
        variant: str = "p_sc_residual",
        *,
        target_region_area: int = 16,
        min_region_area: int = 4,
        max_regions_per_building: int = 16,
        building_threshold: float = 0.50,
        min_building_area: int = 8,
        change_alpha_max: float = 0.30,
        peak_temperature: float = 0.20,
    ) -> None:
        super().__init__(
            roi_alpha=0.1,
            detach_building_prior=True,
            token_dim=128,
            mil_residual_scale=0.0,
        )
        self.variant = variant
        self.variant_spec = get_variant(variant)
        self.stage = "structure"
        self.biscp = BuildingInternalChangePartition(
            self.variant_spec,
            channels=256,
            token_dim=128,
            target_region_area=target_region_area,
            min_region_area=min_region_area,
            max_regions_per_building=max_regions_per_building,
            building_threshold=building_threshold,
            min_building_area=min_building_area,
            change_alpha_max=change_alpha_max,
            peak_temperature=peak_temperature,
        )

    def configure_stage(self, stage: str) -> None:
        self.stage = stage
        self.biscp.configure_stage(stage)
        # Parent A1 is frozen in every V2 diagnostic/training stage.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        if stage == "structure":
            prefixes = (
                "biscp.rgb_structure.",
                "biscp.structural_head.",
            )
        elif stage == "change":
            prefixes = (
                "biscp.lca.",
                "biscp.rgb_structure.",
                "biscp.structural_head.",
                "biscp.change_encoder.",
                "biscp.change_head.",
                "biscp.change_gate_head.",
                "biscp.evidence_score.",
                "biscp.token_encoder.",
                "biscp.region_evidence_head.",
            )
        elif stage == "feedback":
            prefixes = (
                "biscp.region_update.",
                "biscp.utility_gate.",
            )
        else:
            raise ValueError(f"unknown stage: {stage}")

        for name, parameter in self.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad_(True)
        if stage == "change":
            self.biscp.change_gain.requires_grad_(True)
            # LCA parent-feature gain is explicitly excluded because V2 never
            # injects aligned["feature"] into the A1 branch.
            if hasattr(self.biscp.lca, "gain"):
                self.biscp.lca.gain.requires_grad_(False)
        if stage == "feedback":
            self.biscp.feedback_gain.requires_grad_(True)

    def set_plugin_training(self) -> None:
        self.eval()
        self.biscp.train()

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)
        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior
        parent_fused = self.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        parent_feature_roi = parent_fused * roi_weight
        parent_damage_logits = self.clf_2(parent_feature_roi)
        parent_damage_probability = torch.softmax(parent_damage_logits.detach(), dim=1)

        plugin = self.biscp(
            rgb_feature=rgb_feature,
            sar_feature=sar_feature,
            parent_fused=parent_fused,
            parent_feature_roi=parent_feature_roi,
            p_building=p_building,
            parent_damage_probability=parent_damage_probability,
        )
        damage_logits_feature = self.clf_2(plugin["feature_roi"])
        output_loc = F.interpolate(
            loc_logits_feature, pre_data.shape[-2:], mode="bilinear", align_corners=False
        )
        output_damage = F.interpolate(
            damage_logits_feature, post_data.shape[-2:], mode="bilinear", align_corners=False
        )
        if not return_aux:
            return output_loc, output_damage
        aux = {
            "variant": self.variant,
            "stage": self.stage,
            "p_building": p_building,
            "roi_weight": roi_weight,
            "rgb_feature": rgb_feature,
            "sar_feature": sar_feature,
            "parent_damage_logits_feature": parent_damage_logits,
            "parent_damage_probability": parent_damage_probability,
            **plugin,
        }
        return output_loc, output_damage, aux

    def checkpoint_contract(self) -> Dict[str, object]:
        return {
            "protocol": "DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2",
            "variant": self.variant_spec.to_dict(),
            "stage": self.stage,
            "parent_a1_frozen": True,
            "building_forward_source": "predicted A1 p_building only",
            "gt_used_in_partition_forward": False,
            "hard_same_building_candidates": True,
            "change_refinement": "bounded zero-init residual logits",
            "confidence_usage": "feature only, not multiplicative evidence/feedback gate",
            "region_statistics": "mean/std/soft-peak",
            "test_accessed": False,
        }


__all__ = [
    "BuildingInternalChangePartition",
    "DamageFormerBuildingInternalChangePartitionV2",
]
