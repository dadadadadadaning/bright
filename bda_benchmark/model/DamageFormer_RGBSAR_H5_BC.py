# -*- coding: utf-8 -*-
"""Controlled H5-B/H5-C upgrades for heterogeneous RGB/SAR DamageFormer.

This branch deliberately removes the hard building-query decoder used by H5
BIQ.  It tests two independent hypotheses on top of the validated H4-A4 model:

* B1: instance-conditioned K-point SAR sampling at stages 3 and 4 only.
* C1: train-only joint RGB/SAR feature pretraining for small symmetric
  stage-3/4 adapters.

B1 keeps the standard pixel damage decoder.  C1 uses its pretraining losses
only in a separate pretraining program; supervised fine-tuning receives only
the learned adapter weights and continues to use the existing H4 objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H4 import (
    DamageFormerRGBSARH4,
    ReliabilityGatedCommonUniqueStage,
    hierarchical_probabilities,
)
from .DamageFormer_RGBSAR_H5_BIQ import (
    InstanceSparseDeformableFusion,
    compact_instance_map,
    connected_building_instances,
    resize_instance_map,
)
from .DamageFormer_RGBSAR_Heterogeneous import STAGE_CHANNELS


VALID_H5_BC_VARIANTS = ("b0", "b1", "c1")


@dataclass(frozen=True)
class H5BCVariant:
    name: str
    building_sparse_fusion: bool
    joint_pretrained_adapters: bool
    hard_instance_decode: bool = False
    extra_supervised_losses: bool = False


H5_BC_VARIANTS: Mapping[str, H5BCVariant] = {
    "b0": H5BCVariant("H5-B0-H4A4-Control", False, False),
    "b1": H5BCVariant(
        "H5-B1-Building-Sparse-Deformable-Fusion",
        True,
        False,
    ),
    "c1": H5BCVariant(
        "H5-C1-Joint-Pretrained-RGBSAR-Adapters",
        False,
        True,
    ),
}


class ResidualModalityAdapter(nn.Module):
    """Small bottleneck adapter with an exact identity initialization."""

    def __init__(self, channels: int, rank: int = 32) -> None:
        super().__init__()
        rank = min(int(rank), int(channels))
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.depthwise = nn.Conv2d(
            rank,
            rank,
            3,
            padding=1,
            groups=rank,
            bias=False,
        )
        self.activation = nn.GELU()
        self.up = nn.Conv2d(rank, channels, 1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.up(
            self.activation(
                self.depthwise(self.activation(self.down(feature)))
            )
        )
        return feature + residual


class SymmetricStageAdapters(nn.Module):
    """Independent RGB and SAR adapters; parameters are never shared."""

    def __init__(self, channels: int, rank: int = 32) -> None:
        super().__init__()
        self.rgb = ResidualModalityAdapter(channels, rank)
        self.sar = ResidualModalityAdapter(channels, rank)

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rgb(rgb), self.sar(sar)


class DamageFormerRGBSARH5BC(DamageFormerRGBSARH4):
    """H5-B/H5-C model with an exact H4-A4 control path."""

    def __init__(
        self,
        variant: str = "b1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        instance_threshold: float = 0.5,
        min_instance_area: int = 4,
        max_instances: int = 256,
    ) -> None:
        variant = variant.lower()
        if variant not in VALID_H5_BC_VARIANTS:
            raise ValueError(f"unknown H5-B/C variant {variant!r}")
        super().__init__(
            variant="a4",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h5_bc_config = H5_BC_VARIANTS[variant]
        self.instance_threshold = float(instance_threshold)
        self.min_instance_area = int(min_instance_area)
        self.max_instances = int(max_instances)

        if self.h5_bc_config.building_sparse_fusion:
            self.building_sparse_fusions = nn.ModuleDict(
                {
                    "stage3": InstanceSparseDeformableFusion(
                        STAGE_CHANNELS[2],
                        latent_channels[2],
                        points=4,
                    ),
                    "stage4": InstanceSparseDeformableFusion(
                        STAGE_CHANNELS[3],
                        latent_channels[3],
                        points=4,
                    ),
                }
            )
        if self.h5_bc_config.joint_pretrained_adapters:
            self.cross_modal_adapters = nn.ModuleDict(
                {
                    "stage3": SymmetricStageAdapters(STAGE_CHANNELS[2], rank=32),
                    "stage4": SymmetricStageAdapters(STAGE_CHANNELS[3], rank=48),
                }
            )

    def _resolve_instances(
        self,
        localization_probability: torch.Tensor,
        instance_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if instance_map is None:
            return connected_building_instances(
                localization_probability,
                threshold=self.instance_threshold,
                min_area=self.min_instance_area,
                max_instances=self.max_instances,
            )
        if instance_map.ndim == 4 and instance_map.shape[1] == 1:
            instance_map = instance_map[:, 0]
        if instance_map.ndim != 3:
            raise ValueError("external instance_map must be [B,H,W]")
        resized = resize_instance_map(
            instance_map.detach().long().to(localization_probability.device),
            localization_probability.shape[-2:],
        )
        return compact_instance_map(
            resized,
            max_instances=self.max_instances,
        )

    def adapted_deep_stages(
        self,
        rgb_stages: Sequence[torch.Tensor],
        sar_stages: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        rgb_values = list(rgb_stages)
        sar_values = list(sar_stages)
        if not self.h5_bc_config.joint_pretrained_adapters:
            return rgb_values, sar_values
        for index in (2, 3):
            key = f"stage{index + 1}"
            rgb_values[index], sar_values[index] = self.cross_modal_adapters[key](
                rgb_values[index],
                sar_values[index],
            )
        return rgb_values, sar_values

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
        instance_map: torch.Tensor | None = None,
    ):
        if self.variant == "b0":
            return super().forward(pre_rgb, post_sar, return_dict=return_dict)

        input_size = pre_rgb.shape[-2:]
        rgb_stages_raw = self.rgb_encoder(pre_rgb)
        sar_stages_raw = self.sar_encoder(post_sar)

        # Localization always uses the untouched RGB features.
        localization_feature = self.localization_decoder(rgb_stages_raw)
        localization_logits_low = self.localization_head(localization_feature)
        localization_probability = localization_logits_low.softmax(1)[:, 1:2]

        rgb_stages, sar_stages = self.adapted_deep_stages(
            rgb_stages_raw,
            sar_stages_raw,
        )
        sar_aux_low = self.sar_aux_head(sar_stages[-1])
        sar_confidence = sar_aux_low.softmax(1).amax(1, keepdim=True)
        instances = None
        if self.h5_bc_config.building_sparse_fusion:
            instances = self._resolve_instances(
                localization_probability,
                instance_map,
            )

        joint_stages: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for index, (module, rgb, sar) in enumerate(
            zip(self.joint_stages, rgb_stages, sar_stages),
            start=1,
        ):
            if isinstance(module, ReliabilityGatedCommonUniqueStage):
                joint, stage_diagnostics = module(
                    rgb,
                    sar,
                    building_prior=localization_probability,
                    sar_confidence=sar_confidence,
                )
            else:
                joint, stage_diagnostics = module(rgb, sar)

            key = f"stage{index}"
            if (
                self.h5_bc_config.building_sparse_fusion
                and key in self.building_sparse_fusions
            ):
                assert instances is not None
                joint, sparse_diagnostics = self.building_sparse_fusions[key](
                    joint,
                    rgb,
                    sar,
                    instances,
                )
                stage_diagnostics.update(sparse_diagnostics)
            joint_stages.append(joint)
            for name, value in stage_diagnostics.items():
                diagnostics[f"stage{index}_{name}"] = value

        damage_feature = self.damage_decoder(joint_stages)
        damage_feature = damage_feature * (
            0.5 + 0.5 * localization_probability.detach()
        )
        base_damage_logits_low = self.damage_head(damage_feature)

        building_low = self.hierarchy_building_head(damage_feature)
        affected_low = self.hierarchy_affected_head(damage_feature)
        destroyed_low = self.hierarchy_destroyed_head(damage_feature)
        hierarchy_probability = hierarchical_probabilities(
            building_low,
            affected_low,
            destroyed_low,
        )
        hierarchy_logits_low = hierarchy_probability.clamp_min(1e-7).log()
        hierarchy_transition = torch.tanh(self.hierarchy_transition)
        damage_logits_low = base_damage_logits_low + hierarchy_transition * (
            hierarchy_logits_low - base_damage_logits_low
        )

        outputs: Dict[str, torch.Tensor] = {
            "output_loc": F.interpolate(
                localization_logits_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "output_dam": F.interpolate(
                damage_logits_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "base_damage_logits": F.interpolate(
                base_damage_logits_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "sar_aux_logits": F.interpolate(
                sar_aux_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "localization_probability_low": localization_probability,
            "hierarchy_building_logits": F.interpolate(
                building_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "hierarchy_affected_logits": F.interpolate(
                affected_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "hierarchy_destroyed_logits": F.interpolate(
                destroyed_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "hierarchy_transition": hierarchy_transition.reshape(1, 1, 1, 1),
            **diagnostics,
        }
        if instances is not None:
            outputs["sparse_instance_map_low"] = instances
            outputs["sparse_instance_count"] = torch.stack(
                [
                    instances.new_tensor(
                        int(instances[index].max().item()),
                        dtype=torch.float32,
                    )
                    for index in range(instances.shape[0])
                ]
            )
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _feature_block_mask(
    reference: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return (
        torch.rand(
            reference.shape[0],
            1,
            reference.shape[2],
            reference.shape[3],
            device=reference.device,
            generator=generator,
        )
        < float(ratio)
    ).to(reference.dtype)


class JointRGBSARFeaturePretrainer(nn.Module):
    """Train-only feature predictor for the C1 symmetric adapters.

    RGB/SAR encoders remain frozen.  Only stage-3/4 adapters and disposable
    projectors/predictors are optimized.  Damage labels are used solely to form
    background/intact/changed masks; damaged and destroyed are never presented
    as distinct pretraining targets.
    """

    def __init__(
        self,
        model: DamageFormerRGBSARH5BC,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if model.variant != "c1":
            raise ValueError("joint pretraining requires a C1 model")
        self.model = model
        self.projectors = nn.ModuleDict()
        self.rgb_to_sar = nn.ModuleDict()
        self.sar_to_rgb = nn.ModuleDict()
        for key, channels in (
            ("stage3", STAGE_CHANNELS[2]),
            ("stage4", STAGE_CHANNELS[3]),
        ):
            self.projectors[f"{key}_rgb"] = nn.Conv2d(
                channels, embedding_dim, 1
            )
            self.projectors[f"{key}_sar"] = nn.Conv2d(
                channels, embedding_dim, 1
            )
            self.rgb_to_sar[key] = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )
            self.sar_to_rgb[key] = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )

    def trainable_adapter_state(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.model.cross_modal_adapters.state_dict().items()
        }

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        damage: torch.Tensor,
        *,
        mask_ratio: float = 0.60,
        changed_margin: float = 0.20,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        # The base A4 encoders are fixed targets, not an EMA teacher.
        with torch.no_grad():
            rgb_raw = [value.detach() for value in self.model.rgb_encoder(rgb)]
            sar_raw = [value.detach() for value in self.model.sar_encoder(sar)]
        rgb_adapted, sar_adapted = self.model.adapted_deep_stages(
            rgb_raw,
            sar_raw,
        )

        alignment_terms: List[torch.Tensor] = []
        unique_terms: List[torch.Tensor] = []
        prediction_terms: List[torch.Tensor] = []
        variance_terms: List[torch.Tensor] = []
        unchanged_cosine_terms: List[torch.Tensor] = []
        changed_cosine_terms: List[torch.Tensor] = []
        for index in (2, 3):
            key = f"stage{index + 1}"
            rgb_feature = rgb_adapted[index]
            sar_feature = sar_adapted[index]
            target = F.interpolate(
                damage.unsqueeze(1).float(),
                size=rgb_feature.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
            valid = target != 255
            background = valid & (target == 0)
            intact = valid & (target == 1)
            changed = valid & ((target == 2) | (target == 3))

            rgb_embedding = self.projectors[f"{key}_rgb"](rgb_feature)
            sar_embedding = self.projectors[f"{key}_sar"](sar_feature)
            rgb_normalized = F.normalize(rgb_embedding.float(), dim=1)
            sar_normalized = F.normalize(sar_embedding.float(), dim=1)
            cosine = (rgb_normalized * sar_normalized).sum(1)
            class_alignment = []
            for mask in (background, intact):
                if mask.any():
                    class_alignment.append((1.0 - cosine[mask]).mean())
                    unchanged_cosine_terms.append(cosine[mask].mean())
            if class_alignment:
                alignment_terms.append(torch.stack(class_alignment).mean())
            if changed.any():
                changed_cosine_terms.append(cosine[changed].mean())
                unique_terms.append(
                    F.relu(cosine[changed] - float(changed_margin)).mean()
                )

            block_mask = _feature_block_mask(
                rgb_feature,
                mask_ratio,
                generator,
            )
            masked_intact = block_mask[:, 0].bool() & intact
            if masked_intact.any():
                rgb_input = rgb_feature * (1.0 - block_mask)
                sar_input = sar_feature * (1.0 - block_mask)
                predicted_sar = self.rgb_to_sar[key](rgb_input)
                predicted_rgb = self.sar_to_rgb[key](sar_input)
                # Cosine distance has an O(1) scale regardless of channel
                # count.  A channel-averaged Smooth-L1 between normalized
                # vectors is O(1/C) and made this pretext task effectively
                # inactive for the 384/768-channel deep stages.
                sar_distance = 1.0 - F.cosine_similarity(
                    predicted_sar.float(),
                    sar_raw[index].float(),
                    dim=1,
                )
                rgb_distance = 1.0 - F.cosine_similarity(
                    predicted_rgb.float(),
                    rgb_raw[index].float(),
                    dim=1,
                )
                prediction_terms.append(
                    0.5
                    * (
                        sar_distance[masked_intact].mean()
                        + rgb_distance[masked_intact].mean()
                    )
                )

            for embedding in (rgb_embedding, sar_embedding):
                tokens = embedding.float().permute(0, 2, 3, 1)[valid]
                if tokens.shape[0] > 1:
                    standard_deviation = torch.sqrt(
                        tokens.var(dim=0, unbiased=False) + 1e-4
                    )
                    variance_terms.append(
                        F.relu(1.0 - standard_deviation).mean()
                    )

        zero = rgb.float().sum() * 0.0
        alignment = (
            torch.stack(alignment_terms).mean() if alignment_terms else zero
        )
        unique = torch.stack(unique_terms).mean() if unique_terms else zero
        prediction = (
            torch.stack(prediction_terms).mean() if prediction_terms else zero
        )
        variance = (
            torch.stack(variance_terms).mean() if variance_terms else zero
        )
        unchanged_cosine = (
            torch.stack(unchanged_cosine_terms).mean()
            if unchanged_cosine_terms
            else zero
        )
        changed_cosine = (
            torch.stack(changed_cosine_terms).mean()
            if changed_cosine_terms
            else zero
        )
        # Changed pixels are much rarer and share the same projectors as the
        # unchanged pixels.  Equal weighting keeps the common representation
        # from winning by construction while retaining a single, compact
        # pretraining objective rather than adding another supervised loss.
        total = alignment + 0.50 * prediction + 0.50 * unique + 0.05 * variance
        return {
            "joint_pretrain_loss": total,
            "unchanged_alignment_loss": alignment,
            "building_feature_prediction_loss": prediction,
            "changed_unique_loss": unique,
            "variance_loss": variance,
            "unchanged_cosine_similarity": unchanged_cosine,
            "changed_cosine_similarity": changed_cosine,
        }


def build_h5_bc_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH5BC:
    return DamageFormerRGBSARH5BC(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H5_BC_VARIANTS",
    "H5_BC_VARIANTS",
    "ResidualModalityAdapter",
    "SymmetricStageAdapters",
    "DamageFormerRGBSARH5BC",
    "JointRGBSARFeaturePretrainer",
    "build_h5_bc_damageformer",
]
