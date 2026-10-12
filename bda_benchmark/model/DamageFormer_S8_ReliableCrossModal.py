# -*- coding: utf-8 -*-
"""S8-R0...R5: reliable cross-modal RGB-SAR damage assessment.

This is a new ablation family.  It does not modify either
``DamageFormer_S8_IA_MLA`` or ``DamageFormer_S8_MICB_DSMIL``.

Variants
--------
R0
    Stable dual-Swin baseline, without HCDM or intact alignment.  Retains the
    accepted suppressive soft building ROI used to train the base classifier.
R1
    Stage2/Stage3 local structural-relation difference, auxiliary only.
R2
    R1 + bounded spatial reliability gate on the main damage feature.
R3
    R2 + near-identity residual SAR input adapter.
R4
    R3 + pixel DamageExist and conditional Severity auxiliary heads.
R5
    R4 + conservative Stage3 cross-modal rectification.

R6 common/private decomposition is intentionally not implemented until the
preceding ablations demonstrate stable validation gains.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .DamageFormer_S3_BGMP_K3 import ResBlock, SwinTransformerFeatureExtractor
    from .reliable_cross_modal import (
        BoundedSpatialReliabilityGate,
        ChangeRelationPyramid,
        ConservativeStage3Rectifier,
        PixelDamageHierarchyHead,
        ResidualSARAdapter,
    )
else:
    from model.DamageFormer_S3_BGMP_K3 import ResBlock, SwinTransformerFeatureExtractor
    from model.reliable_cross_modal import (
        BoundedSpatialReliabilityGate,
        ChangeRelationPyramid,
        ConservativeStage3Rectifier,
        PixelDamageHierarchyHead,
        ResidualSARAdapter,
    )

from torchvision.models import Swin_T_Weights, swin_t


@dataclass(frozen=True)
class ReliableVariantSpec:
    name: str
    relation_difference: bool
    spatial_gate: bool
    sar_adapter: bool
    pixel_hierarchy: bool
    stage3_rectifier: bool


VARIANT_SPECS: Dict[str, ReliableVariantSpec] = {
    "r0": ReliableVariantSpec("S8-R0", False, False, False, False, False),
    "r1": ReliableVariantSpec("S8-R1", True, False, False, False, False),
    "r2": ReliableVariantSpec("S8-R2", True, True, False, False, False),
    "r3": ReliableVariantSpec("S8-R3", True, True, True, False, False),
    "r4": ReliableVariantSpec("S8-R4", True, True, True, True, False),
    "r5": ReliableVariantSpec("S8-R5", True, True, True, True, True),
}


def get_variant_spec(variant: str) -> ReliableVariantSpec:
    key = str(variant).strip().lower().replace("s8-", "")
    if key not in VARIANT_SPECS:
        raise ValueError(
            f"unknown reliable-cross-modal variant {variant!r}; "
            f"choose one of {sorted(VARIANT_SPECS)}"
        )
    return VARIANT_SPECS[key]


class DamageFormer_S8_ReliableCrossModal(nn.Module):
    """Reliable RGB-SAR S8 model with strictly staged ablation switches."""

    STAGE_CHANNELS = (96, 192, 384, 768)

    def __init__(
        self,
        variant: str = "r1",
        *,
        relation_projection_channels: int = 32,
        relation_channels: int = 64,
        gate_alpha_max: float = 0.1,
        rectifier_alpha_max: float = 0.05,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= roi_alpha <= 1.0:
            raise ValueError("roi_alpha must be in [0,1]")
        self.variant = str(variant).strip().lower().replace("s8-", "")
        self.variant_spec = get_variant_spec(self.variant)
        self.return_dict = bool(return_dict)
        self.roi_alpha = float(roi_alpha)
        self.detach_building_prior = bool(detach_building_prior)

        weights = Swin_T_Weights.DEFAULT if use_pretrained_swin else None
        # The wrapper removes unused Swin norm/head exactly like the accepted
        # S8_MICB_DSMIL lineage, making its core checkpoint directly migratable.
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_t(weights=weights))
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_t(weights=weights))
        self.fusion_layer_1 = ResBlock(
            1440, 256, downsample=nn.Conv2d(1440, 256, 1)
        )
        self.fusion_layer_2 = ResBlock(
            1440, 256, downsample=nn.Conv2d(1440, 256, 1)
        )
        self.fusion_layer_3 = ResBlock(
            512, 256, downsample=nn.Conv2d(512, 256, 1)
        )
        self.clf_1 = nn.Conv2d(256, 2, kernel_size=1)
        self.clf_2 = nn.Conv2d(256, 4, kernel_size=1)

        if self.variant_spec.relation_difference:
            self.relation_pyramid = ChangeRelationPyramid(
                stage_channels=(self.STAGE_CHANNELS[1], self.STAGE_CHANNELS[2]),
                projection_channels=relation_projection_channels,
                relation_channels=relation_channels,
                out_channels=relation_channels,
            )
        if self.variant_spec.spatial_gate:
            self.reliability_gate = BoundedSpatialReliabilityGate(
                base_channels=256,
                relation_channels=relation_channels,
                alpha_max=gate_alpha_max,
            )
        if self.variant_spec.sar_adapter:
            self.sar_adapter = ResidualSARAdapter(3, 32)
        if self.variant_spec.pixel_hierarchy:
            self.pixel_hierarchy = PixelDamageHierarchyHead(256, relation_channels)
        if self.variant_spec.stage3_rectifier:
            self.stage3_rectifier = ConservativeStage3Rectifier(
                channels=self.STAGE_CHANNELS[2],
                hidden_channels=relation_channels,
                alpha_max=rectifier_alpha_max,
            )
            self.rectified_rgb_delta_to_final = nn.Conv2d(
                self.STAGE_CHANNELS[2], 256, kernel_size=1, bias=False
            )
            self.rectified_sar_delta_to_final = nn.Conv2d(
                self.STAGE_CHANNELS[2], 256, kernel_size=1, bias=False
            )
            nn.init.zeros_(self.rectified_rgb_delta_to_final.weight)
            nn.init.zeros_(self.rectified_sar_delta_to_final.weight)

    @staticmethod
    def _to_nchw(feature: torch.Tensor) -> torch.Tensor:
        return feature.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _aggregate_swin_stages(
        stages: list[torch.Tensor],
        fusion: nn.Module,
    ) -> torch.Tensor:
        if len(stages) != 4:
            raise ValueError(f"expected four Swin stages, got {len(stages)}")
        target_size = stages[0].shape[-2:]
        aligned = [
            stages[0],
            *[
                F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in stages[1:]
            ],
        ]
        return fusion(torch.cat((aligned[3], aligned[2], aligned[1], aligned[0]), dim=1))

    def _encode(self, pre_data: torch.Tensor, post_data: torch.Tensor):
        sar_input = self.sar_adapter(post_data) if hasattr(self, "sar_adapter") else post_data
        pre_maps = self.encoder_1(pre_data)
        post_maps = self.encoder_2(sar_input)
        if len(pre_maps) != 5 or len(post_maps) != 5:
            raise RuntimeError(
                f"expected five Swin sequential outputs, got {len(pre_maps)}/{len(post_maps)}"
            )
        rgb_stages = [self._to_nchw(feature) for feature in pre_maps[1:5]]
        sar_stages = [self._to_nchw(feature) for feature in post_maps[1:5]]
        return sar_input, rgb_stages, sar_stages

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        return_dict: Optional[bool] = None,
    ):
        sar_input, rgb_stages, sar_stages = self._encode(pre_data, post_data)

        # Localization deliberately uses the unrectified RGB path, preventing a
        # late ablation from changing the building prior through a circular gate.
        final_rgb_original = self._aggregate_swin_stages(
            rgb_stages, self.fusion_layer_1
        )
        final_sar_original = self._aggregate_swin_stages(
            sar_stages, self.fusion_layer_2
        )
        loc_logits_feature = self.clf_1(final_rgb_original)
        building_probability = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]

        outputs: Dict[str, torch.Tensor | str | Dict[str, object]] = {
            "building_probability": building_probability,
            "final_feat_rgb": final_rgb_original,
            "final_feat_sar": final_sar_original,
            "sar_adapter_delta_mean": (sar_input - post_data).detach().abs().mean(),
        }

        relation_feature: Optional[torch.Tensor] = None
        relation_stats: Dict[str, torch.Tensor] = {}
        if self.variant_spec.relation_difference:
            relation_feature, change_logits, relation_stats = self.relation_pyramid(
                rgb_stages[1],
                sar_stages[1],
                rgb_stages[2],
                sar_stages[2],
                target_size=final_rgb_original.shape[-2:],
            )
            outputs["relation_feature"] = relation_feature
            outputs["change_logits_feature"] = change_logits
            outputs.update(relation_stats)

        final_rgb = final_rgb_original
        final_sar = final_sar_original
        if self.variant_spec.stage3_rectifier:
            if "relation_energy_s3" not in relation_stats:
                raise RuntimeError("R5 rectification requires Stage3 relation energy")
            rgb_stage3, sar_stage3, rectify_stats = self.stage3_rectifier(
                rgb_stages[2],
                sar_stages[2],
                building_probability,
                relation_stats["relation_energy_s3"],
            )
            # Project only the bounded Stage3 correction into the final
            # feature.  This avoids running the same BatchNorm fusion blocks
            # twice in one training forward.
            rgb_delta = self.rectified_rgb_delta_to_final(
                rgb_stage3 - rgb_stages[2]
            )
            sar_delta = self.rectified_sar_delta_to_final(
                sar_stage3 - sar_stages[2]
            )
            rgb_delta = F.interpolate(
                rgb_delta,
                size=final_rgb_original.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            sar_delta = F.interpolate(
                sar_delta,
                size=final_sar_original.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            final_rgb = final_rgb_original + rgb_delta
            final_sar = final_sar_original + sar_delta
            outputs.update(rectify_stats)
            outputs["final_feat_rgb_rectified"] = final_rgb
            outputs["final_feat_sar_rectified"] = final_sar

        damage_fused = self.fusion_layer_3(torch.cat((final_rgb, final_sar), dim=1))
        # Preserve the accepted S8_MICB_DSMIL pixel path.  Its classifier was
        # trained on the soft building-ROI feature, not on the raw fused map.
        building_prior = (
            building_probability.detach()
            if self.detach_building_prior
            else building_probability
        )
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior
        damage_base = damage_fused * roi_weight
        damage_feature = damage_base
        outputs["roi_weight"] = roi_weight
        outputs["final_feat_damage_fused"] = damage_fused
        outputs["final_feat_damage_base"] = damage_base
        if self.variant_spec.spatial_gate:
            if relation_feature is None:
                raise RuntimeError("spatial reliability gating requires relation features")
            damage_feature, gate_stats = self.reliability_gate(
                damage_base,
                relation_feature,
                building_probability,
                final_rgb,
                final_sar,
            )
            outputs.update(gate_stats)
        outputs["final_feat_damage"] = damage_feature

        damage_logits_feature = self.clf_2(damage_feature)
        output_loc = F.interpolate(
            loc_logits_feature,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_dam = F.interpolate(
            damage_logits_feature,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        outputs["output_loc"] = output_loc
        outputs["output_dam"] = output_dam
        outputs["main_dam_logits"] = output_dam

        if self.variant_spec.pixel_hierarchy:
            exist_logits, severity_logits = self.pixel_hierarchy(damage_feature)
            outputs["pixel_exist_logits_feature"] = exist_logits
            outputs["pixel_severity_logits_feature"] = severity_logits

        use_dict = self.return_dict if return_dict is None else bool(return_dict)
        if not use_dict:
            return output_loc, output_dam
        outputs["variant"] = self.variant_spec.name
        outputs["variant_spec"] = asdict(self.variant_spec)
        return outputs


_CORE_PREFIXES = (
    "encoder_1.",
    "encoder_2.",
    "fusion_layer_1.",
    "fusion_layer_2.",
    "fusion_layer_3.",
    "clf_1.",
    "clf_2.",
)


def _extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a state-dict mapping or contain one")
    return checkpoint


def _clean_state_dict(
    state: Mapping[str, torch.Tensor],
) -> MutableMapping[str, torch.Tensor]:
    cleaned: MutableMapping[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_s8_core_checkpoint(
    model: DamageFormer_S8_ReliableCrossModal,
    checkpoint_path: str | Path,
    *,
    strict_core: bool = True,
) -> Dict[str, object]:
    """Strictly migrate the shared S8 core and isolate all new parameters.

    Extra IA/HCDM/MICB/prototype keys in an older S8 checkpoint are reported
    but never loaded.  A missing or shape-mismatched encoder/fusion/classifier
    key raises immediately when ``strict_core=True``.
    """

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source = _clean_state_dict(_extract_state_dict(checkpoint))
    target = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    shape_mismatch: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for key, value in source.items():
        if key not in target:
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatch[key] = (tuple(value.shape), tuple(target[key].shape))
            continue
        compatible[key] = value

    missing_core = [
        key
        for key in target
        if key.startswith(_CORE_PREFIXES) and key not in compatible
    ]
    mismatch_core = {
        key: shapes
        for key, shapes in shape_mismatch.items()
        if key.startswith(_CORE_PREFIXES)
    }
    if strict_core and (missing_core or mismatch_core):
        raise RuntimeError(
            "S8 core checkpoint migration failed: "
            f"missing_core={missing_core[:12]}, mismatch_core={mismatch_core}"
        )

    incompatible = model.load_state_dict(compatible, strict=False)
    source_only = sorted(key for key in source if key not in target)
    new_only = sorted(key for key in target if key not in compatible)
    report: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "loaded_key_count": len(compatible),
        "source_only_key_count": len(source_only),
        "source_only_prefix_sample": source_only[:20],
        "new_or_missing_key_count": len(new_only),
        "new_or_missing_key_sample": new_only[:20],
        "shape_mismatch": shape_mismatch,
        "missing_core": missing_core,
        "strict_core": bool(strict_core),
        "load_missing_keys": list(incompatible.missing_keys),
        "load_unexpected_keys": list(incompatible.unexpected_keys),
    }
    return report


DamageFormer = DamageFormer_S8_ReliableCrossModal


__all__ = [
    "DamageFormer",
    "DamageFormer_S8_ReliableCrossModal",
    "ReliableVariantSpec",
    "VARIANT_SPECS",
    "get_variant_spec",
    "load_s8_core_checkpoint",
]


if __name__ == "__main__":
    model = DamageFormer_S8_ReliableCrossModal(
        variant="r4", use_pretrained_swin=False, return_dict=True
    ).eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    for name in (
        "output_loc",
        "output_dam",
        "change_logits_feature",
        "pixel_exist_logits_feature",
        "pixel_severity_logits_feature",
    ):
        print(name, tuple(result[name].shape))
