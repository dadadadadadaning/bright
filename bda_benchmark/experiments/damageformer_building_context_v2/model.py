"""DamageFormer learned cross-modal regions + Building Context V2."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    soft_pool_unpool,
)
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import (
    DamageFormerA1CrossModalSuperpixel,
)

from .building_ops import (
    predicted_building_instances,
    hard_region_ids_from_soft_assignment,
    region_building_membership,
    soft_unpool_tokens,
)
from .variants import get_variant


def _groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class BuildingContextPlugin(nn.Module):
    """Region -> Building -> Region -> Pixel.

    V2 intentionally does NOT add Memory, a GNN, or a building-class head.
    This isolates whether same-building context improves pixel damage prediction.
    """

    def __init__(
        self,
        token_dim: int = 128,
        feature_dim: int = 256,
        purity_threshold: float = 0.60,
        building_threshold: float = 0.50,
        min_building_area: int = 8,
        shuffle_context: bool = False,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.purity_threshold = float(purity_threshold)
        self.building_threshold = float(building_threshold)
        self.min_building_area = int(min_building_area)
        self.shuffle_context = bool(shuffle_context)

        self.region_attention = nn.Sequential(
            nn.Linear(token_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.context_projection = nn.Sequential(
            nn.Linear(token_dim, token_dim, bias=False),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

        self.feedback_gate = nn.Sequential(
            nn.Linear(token_dim * 2 + 3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

        self.pixel_update = nn.Sequential(
            nn.Conv2d(feature_dim + token_dim, feature_dim, 1, bias=False),
            nn.GroupNorm(_groups(feature_dim), feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
        )

        # Identity-safe start: new model initially equals its region parent.
        self.pixel_gain = nn.Parameter(torch.zeros(()))

    def _pool_confidence(
        self,
        confidence_map: torch.Tensor,
        probabilities: torch.Tensor,
        candidate_ids: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        # IMPORTANT BF16 contract:
        # soft_pool_unpool uses content.new_zeros; scatter_add_ requires exact
        # dtype equality between destination and source.
        confidence_for_pool = confidence_map.to(dtype=probabilities.dtype)

        pooled = soft_pool_unpool(
            confidence_for_pool,
            probabilities,
            candidate_ids,
            token_count,
        )
        return pooled["tokens"].squeeze(-1).float().clamp(0.0, 1.0)

    def _context_per_region(
        self,
        tokens: torch.Tensor,
        mass: torch.Tensor,
        confidence: torch.Tensor,
        building_of_region: torch.Tensor,
    ):
        b, _, _ = tokens.shape
        contexts = torch.zeros_like(tokens)
        area_ratio = torch.zeros_like(mass, dtype=torch.float32)
        building_count = torch.zeros(b, device=tokens.device)
        accepted_regions = torch.zeros(b, device=tokens.device)

        for bi in range(b):
            ids = torch.unique(building_of_region[bi])
            ids = ids[ids > 0]
            if ids.numel() == 0:
                continue

            built_contexts = []
            groups = []

            for building_id in ids.tolist():
                idx = torch.nonzero(
                    building_of_region[bi] == int(building_id), as_tuple=False
                ).flatten()
                if idx.numel() == 0:
                    continue

                m = mass[bi, idx].float()
                area = m / m.sum().clamp_min(1.0e-6)
                area_ratio[bi, idx] = area

                node = tokens[bi, idx]
                conf = confidence[bi, idx]
                attn_input = torch.cat(
                    [node, conf.to(node.dtype)[:, None], area.to(node.dtype)[:, None]],
                    dim=-1,
                )
                score = self.region_attention(attn_input).squeeze(-1).float()
                score = score + conf.clamp_min(1.0e-4).log()
                alpha = torch.softmax(score, dim=0).to(node.dtype)
                z = (alpha[:, None] * node).sum(dim=0)

                built_contexts.append(z)
                groups.append(idx)

            if not built_contexts:
                continue

            stacked = torch.stack(built_contexts, dim=0)

            if self.shuffle_context:
                if stacked.shape[0] > 1:
                    stacked = torch.roll(stacked, shifts=1, dims=0)
                else:
                    # One-building crop: zero context is a conservative wrong-context control.
                    stacked = torch.zeros_like(stacked)

            for z, idx in zip(stacked, groups):
                contexts[bi, idx] = z[None].expand(idx.numel(), -1)

            building_count[bi] = float(len(groups))
            accepted_regions[bi] = float(sum(int(x.numel()) for x in groups))

        return contexts, area_ratio, building_count, accepted_regions

    def forward(
        self,
        *,
        region_tokens: torch.Tensor,
        region_mass: torch.Tensor,
        region_confidence_map: torch.Tensor,
        association_probabilities: torch.Tensor,
        candidate_ids: torch.Tensor,
        p_building: torch.Tensor,
        base_feature_roi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, token_count, token_dim = region_tokens.shape
        h, w = base_feature_roi.shape[-2:]
        if token_dim != self.token_dim:
            raise ValueError(
                f"BuildingContext expected token_dim={self.token_dim}, got {token_dim}"
            )

        building_ids = predicted_building_instances(
            p_building,
            size=(h, w),
            threshold=self.building_threshold,
            min_area=self.min_building_area,
        )

        hard_region_ids = hard_region_ids_from_soft_assignment(
            association_probabilities,
            candidate_ids,
            h,
            w,
        )

        membership = region_building_membership(
            hard_region_ids,
            building_ids,
            token_count=token_count,
            purity_threshold=self.purity_threshold,
        )
        building_of_region = membership["building_of_region"]
        purity = membership["region_building_purity"]

        region_confidence = self._pool_confidence(
            region_confidence_map,
            association_probabilities,
            candidate_ids,
            token_count,
        )

        building_context, area_ratio, building_count, accepted_regions = (
            self._context_per_region(
                region_tokens,
                region_mass,
                region_confidence,
                building_of_region,
            )
        )

        has_context = (building_of_region > 0).float()
        gate_input = torch.cat(
            [
                region_tokens,
                building_context,
                region_confidence.to(region_tokens.dtype)[..., None],
                area_ratio.to(region_tokens.dtype)[..., None],
                purity.to(region_tokens.dtype)[..., None],
            ],
            dim=-1,
        )
        beta = torch.sigmoid(self.feedback_gate(gate_input)).squeeze(-1)
        beta = beta * has_context.to(beta.dtype)

        refined_tokens = region_tokens + beta[..., None] * self.context_projection(
            building_context
        )

        pixel_context = soft_unpool_tokens(
            refined_tokens,
            association_probabilities,
            candidate_ids,
            h,
            w,
        )

        update = self.pixel_update(
            torch.cat([base_feature_roi, pixel_context.to(base_feature_roi.dtype)], dim=1)
        )

        building_gate = F.interpolate(
            p_building.detach(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).to(base_feature_roi.dtype)

        confidence_gate = region_confidence_map.detach().to(base_feature_roi.dtype)

        feature_roi = (
            base_feature_roi
            + torch.tanh(self.pixel_gain)
            * building_gate
            * confidence_gate
            * update
        )

        return {
            "feature_roi": feature_roi,
            "building_ids": building_ids,
            "hard_region_ids": hard_region_ids,
            "building_of_region": building_of_region,
            "region_building_purity": purity,
            "region_confidence_token": region_confidence,
            "region_area_ratio": area_ratio,
            "building_context": building_context,
            "building_feedback_gate": beta,
            "refined_region_tokens": refined_tokens,
            "building_pixel_context": pixel_context,
            "building_update": update,
            "bir_pixel_gain": self.pixel_gain,
            "building_count_per_image": building_count,
            "accepted_region_count_per_image": accepted_regions,
        }


class DamageFormerBuildingContextV2(DamageFormerA1CrossModalSuperpixel):
    """Wrapper around the existing C3-C7 region implementation."""

    def __init__(
        self,
        experiment_variant: str,
        region_stride: int = 8,
        purity_threshold: float = 0.60,
    ):
        self.experiment_spec = get_variant(experiment_variant)

        super().__init__(
            variant=self.experiment_spec.base_region_variant,
            region_stride=region_stride,
        )

        if self.variant_spec.partition != "learnable":
            raise ValueError(
                "BuildingContextV2 requires a learnable association variant; "
                f"got partition={self.variant_spec.partition!r}"
            )

        self.experiment_variant = experiment_variant
        self.bir = BuildingContextPlugin(
            token_dim=128,
            feature_dim=256,
            purity_threshold=purity_threshold,
            shuffle_context=self.experiment_spec.shuffle_building_context,
        )

    def freeze_for_experiment(self):
        # Always freeze the inherited DamageFormer parent.
        for name, parameter in self.named_parameters():
            trainable = False

            if name.startswith("cmsp.") and not self.experiment_spec.freeze_cmsp:
                trainable = True

            if name.startswith("bir.") and self.experiment_spec.use_building_context:
                trainable = True

            parameter.requires_grad_(trainable)

    def set_plugin_training(self):
        self.eval()

        if any(p.requires_grad for p in self.cmsp.parameters()):
            self.cmsp.train()

        if any(p.requires_grad for p in self.bir.parameters()):
            self.bir.train()

    def forward(self, pre_data, post_data, return_aux: bool = False):
        output_loc, _, aux = super().forward(pre_data, post_data, return_aux=True)

        required = (
            "region_tokens",
            "region_mass",
            "region_confidence",
            "association_probabilities",
            "candidate_ids",
            "feature_roi",
            "p_building",
        )
        missing = [key for key in required if key not in aux]
        if missing:
            raise RuntimeError(
                f"BuildingContext requires learned-region aux values, missing={missing}"
            )

        bir = self.bir(
            region_tokens=aux["region_tokens"],
            region_mass=aux["region_mass"],
            region_confidence_map=aux["region_confidence"],
            association_probabilities=aux["association_probabilities"],
            candidate_ids=aux["candidate_ids"],
            p_building=aux["p_building"],
            base_feature_roi=aux["feature_roi"],
        )

        damage_logits_feature = self.clf_2(bir["feature_roi"])
        output_damage = F.interpolate(
            damage_logits_feature,
            post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        aux.update(bir)
        aux["experiment_variant"] = self.experiment_variant

        if return_aux:
            return output_loc, output_damage, aux
        return output_loc, output_damage

    def checkpoint_contract(self):
        return {
            "protocol": "DAMAGEFORMER_BUILDING_CONTEXT_V2",
            "experiment_variant": self.experiment_spec.to_dict(),
            "base_region_variant": self.variant_spec.to_dict(),
            "main_region_uses_raw_rgb_color_clustering": False,
            "region_partition": "learned_SCN_style_9_neighbor_association",
            "building_feedback_zero_init": True,
            "final_prediction_unit": "pixel",
            "test_accessed": False,
        }
