# -*- coding: utf-8 -*-
"""H4 controlled upgrades for heterogeneous RGB/SAR DamageFormer.

H4 is deliberately implemented next to, rather than inside, the validated H3
branch.  This keeps old checkpoints and the active H3-15000 run untouched.

Variants
--------
A0
    Exact H3 architecture used as a checkpoint/loading guard.
A1
    Reliability-gated SAR candidate with a zero-initialized transition from H3.
A2
    A1 plus bounded learnable flow alignment before local sparse matching.
A3
    A2 architecture, trained with a train-split-only SAR-MAE initialization.
A4
    A3 plus a hierarchical building/affected/severity damage head.  Its
    transition is also zero initialized, so loading an H3 checkpoint preserves
    the H3 prediction exactly before optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_Heterogeneous import (
    STAGE_CHANNELS,
    CommonUniqueStage,
    ConvGNAct,
    DamageFormerRGBSARHeterogeneous,
    MultiScaleDecoder,
    NativeSARPyramid,
)


VALID_H4_VARIANTS = ("a0", "a1", "a2", "a3", "a4")


@dataclass(frozen=True)
class H4Variant:
    name: str
    reliability_gate: bool
    flow_alignment: bool
    sar_masked_pretraining: bool
    hierarchical_damage: bool


H4_VARIANTS: Mapping[str, H4Variant] = {
    "a0": H4Variant("H4-A0-H3-Control", False, False, False, False),
    "a1": H4Variant("H4-A1-Reliability-Gated-Fusion", True, False, False, False),
    "a2": H4Variant("H4-A2-Unchanged-Flow-Alignment", True, True, False, False),
    "a3": H4Variant("H4-A3-SAR-Masked-Pretraining", True, True, True, False),
    "a4": H4Variant(
        "H4-A4-Hierarchical-Balanced-Severity", True, True, True, True
    ),
}


class BoundedFlowAlignment(nn.Module):
    """Predict a bounded local displacement and warp SAR features.

    The last convolution is zero initialized, making this module an exact
    identity at initialization.  Offsets are expressed in feature pixels and
    bounded with tanh before conversion to grid-sample coordinates.
    """

    def __init__(self, channels: int, max_offset: float = 2.0) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.max_offset = float(max_offset)
        self.predictor = nn.Sequential(
            ConvGNAct(3 * channels, hidden, 3),
            nn.Conv2d(hidden, 2, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    @staticmethod
    def _base_grid(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(
        self, rgb_common: torch.Tensor, sar_common: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb_common.shape != sar_common.shape:
            raise ValueError("flow alignment requires equal RGB/SAR feature shapes")
        descriptor = torch.cat(
            [rgb_common, sar_common, (rgb_common - sar_common).abs()], dim=1
        )
        flow = torch.tanh(self.predictor(descriptor)) * self.max_offset
        batch, _, height, width = flow.shape
        normalized = torch.empty_like(flow)
        normalized[:, 0] = flow[:, 0] * (2.0 / max(width - 1, 1))
        normalized[:, 1] = flow[:, 1] * (2.0 / max(height - 1, 1))
        grid = self._base_grid(
            batch, height, width, sar_common.device, sar_common.dtype
        )
        grid = grid + normalized.permute(0, 2, 3, 1)
        aligned = F.grid_sample(
            sar_common,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return aligned, flow


class ReliabilityGatedCommonUniqueStage(CommonUniqueStage):
    """H3 stage with a safe, learnable gated candidate.

    ``self.fuse`` and all H3 parameter names remain unchanged.  A scalar
    transition starts at zero, so ``base + transition*(candidate-base)`` is
    exactly the H3 output immediately after loading an H3 checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        use_flow: bool,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__(
            in_channels,
            latent_channels,
            use_correspondence=True,
            sparse=True,
        )
        self.flow_alignment = (
            BoundedFlowAlignment(latent_channels, max_offset=max_offset)
            if use_flow
            else None
        )
        gate_hidden = max(8, latent_channels // 8)
        self.gate_predictor = nn.Sequential(
            ConvGNAct(6, gate_hidden, 3),
            nn.Conv2d(gate_hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.gate_predictor[-1].weight)
        nn.init.zeros_(self.gate_predictor[-1].bias)
        self.gated_transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_prior: torch.Tensor | None = None,
        sar_confidence: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_common = self.rgb_common(rgb)
        sar_common_raw = self.sar_common(sar)
        rgb_unique = self.rgb_unique(rgb)
        sar_unique = self.sar_unique(sar)

        if self.flow_alignment is None:
            sar_common_for_match = sar_common_raw
            flow = torch.zeros(
                sar_common_raw.shape[0],
                2,
                sar_common_raw.shape[2],
                sar_common_raw.shape[3],
                dtype=sar_common_raw.dtype,
                device=sar_common_raw.device,
            )
        else:
            sar_common_for_match, flow = self.flow_alignment(
                rgb_common, sar_common_raw
            )

        assert self.correspondence is not None
        aligned_sar, reliability = self.correspondence(
            rgb_common, sar_common_for_match
        )
        difference = (rgb_common - aligned_sar).abs()
        cosine = F.cosine_similarity(
            rgb_common.float(), aligned_sar.float(), dim=1
        ).unsqueeze(1).to(rgb_common.dtype)
        if building_prior is None:
            building_prior = torch.ones_like(reliability)
        elif building_prior.shape[-2:] != reliability.shape[-2:]:
            building_prior = F.interpolate(
                building_prior,
                size=reliability.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if sar_confidence is None:
            sar_confidence = torch.ones_like(reliability)
        elif sar_confidence.shape[-2:] != reliability.shape[-2:]:
            sar_confidence = F.interpolate(
                sar_confidence,
                size=reliability.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate_input = torch.cat(
            [
                reliability,
                difference.mean(1, keepdim=True),
                cosine,
                aligned_sar.abs().mean(1, keepdim=True),
                building_prior.detach().to(reliability.dtype),
                sar_confidence.detach().to(reliability.dtype),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.gate_predictor(gate_input))

        base_descriptor = torch.cat(
            [
                rgb_common,
                aligned_sar,
                difference,
                rgb_unique,
                sar_unique,
                reliability,
            ],
            dim=1,
        )
        base_joint = self.fuse(base_descriptor)

        gated_sar = rgb_common + gate * (aligned_sar - rgb_common)
        gated_descriptor = torch.cat(
            [
                rgb_common,
                gated_sar,
                (rgb_common - gated_sar).abs(),
                rgb_unique,
                gate * sar_unique,
                gate,
            ],
            dim=1,
        )
        gated_joint = self.fuse(gated_descriptor)
        transition = torch.tanh(self.gated_transition)
        joint = base_joint + transition * (gated_joint - base_joint)
        diagnostics = {
            "rgb_common": rgb_common,
            "sar_common": aligned_sar,
            "rgb_unique": rgb_unique,
            "sar_unique": sar_unique,
            "reliability": reliability,
            "sar_gate": gate,
            "flow": flow,
            "building_prior": building_prior,
            "sar_confidence": sar_confidence,
            "gate_transition": transition.reshape(1, 1, 1, 1),
        }
        return joint, diagnostics


def hierarchical_probabilities(
    building_logits: torch.Tensor,
    affected_logits: torch.Tensor,
    destroyed_logits: torch.Tensor,
) -> torch.Tensor:
    """Convert three conditional binary heads to four normalized classes."""

    building = building_logits.float().softmax(1)
    affected = affected_logits.float().softmax(1)
    destroyed = destroyed_logits.float().softmax(1)
    probabilities = torch.cat(
        [
            building[:, 0:1],
            building[:, 1:2] * affected[:, 0:1],
            building[:, 1:2] * affected[:, 1:2] * destroyed[:, 0:1],
            building[:, 1:2] * affected[:, 1:2] * destroyed[:, 1:2],
        ],
        dim=1,
    )
    return probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)


class DamageFormerRGBSARH4(DamageFormerRGBSARHeterogeneous):
    """H4 model retaining H3 state-dict names for controlled migration."""

    def __init__(
        self,
        variant: str = "a1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
    ) -> None:
        variant = variant.lower()
        if variant not in VALID_H4_VARIANTS:
            raise ValueError(f"unknown H4 variant {variant!r}")
        super().__init__(
            variant="h3",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h4_config = H4_VARIANTS[variant]
        if self.h4_config.reliability_gate:
            upgraded: List[nn.Module] = []
            for source_channels, latent in zip(STAGE_CHANNELS, latent_channels):
                upgraded.append(
                    ReliabilityGatedCommonUniqueStage(
                        source_channels,
                        latent,
                        use_flow=self.h4_config.flow_alignment,
                    )
                )
            self.joint_stages = nn.ModuleList(upgraded)

        if self.h4_config.hierarchical_damage:
            self.hierarchy_building_head = nn.Conv2d(256, 2, 1)
            self.hierarchy_affected_head = nn.Conv2d(256, 2, 1)
            self.hierarchy_destroyed_head = nn.Conv2d(256, 2, 1)
            self.hierarchy_transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)

        localization_feature = self.localization_decoder(rgb_stages)
        localization_logits_low = self.localization_head(localization_feature)
        localization_probability = localization_logits_low.softmax(1)[:, 1:2]
        sar_aux_low = self.sar_aux_head(sar_stages[-1])
        sar_confidence = sar_aux_low.softmax(1).amax(1, keepdim=True)

        joint_stages: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for index, (module, rgb, sar) in enumerate(
            zip(self.joint_stages, rgb_stages, sar_stages), start=1
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
            joint_stages.append(joint)
            for key, value in stage_diagnostics.items():
                diagnostics[f"stage{index}_{key}"] = value

        damage_feature = self.damage_decoder(joint_stages)
        damage_feature = damage_feature * (
            0.5 + 0.5 * localization_probability.detach()
        )
        base_damage_logits_low = self.damage_head(damage_feature)
        damage_logits_low = base_damage_logits_low
        hierarchy_outputs: Dict[str, torch.Tensor] = {}
        if self.h4_config.hierarchical_damage:
            building_low = self.hierarchy_building_head(damage_feature)
            affected_low = self.hierarchy_affected_head(damage_feature)
            destroyed_low = self.hierarchy_destroyed_head(damage_feature)
            hierarchy_probability = hierarchical_probabilities(
                building_low, affected_low, destroyed_low
            )
            hierarchy_logits_low = hierarchy_probability.clamp_min(1e-7).log()
            hierarchy_transition = torch.tanh(self.hierarchy_transition)
            damage_logits_low = base_damage_logits_low + hierarchy_transition * (
                hierarchy_logits_low - base_damage_logits_low
            )
            hierarchy_outputs = {
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
            }

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
            **hierarchy_outputs,
            **diagnostics,
        }
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


class SARMaskedAutoencoder(nn.Module):
    """Lightweight train-only SAR masked reconstruction pretext model."""

    def __init__(self, depths: Sequence[int] = (2, 2, 4, 2)) -> None:
        super().__init__()
        self.sar_encoder = NativeSARPyramid(depths=depths)
        self.decoder = MultiScaleDecoder(STAGE_CHANNELS, 128)
        self.reconstruction_head = nn.Sequential(
            ConvGNAct(128, 64, 3),
            nn.Conv2d(64, 1, 1),
        )

    @staticmethod
    def block_mask(
        sar: torch.Tensor,
        mask_ratio: float,
        patch_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch, _, height, width = sar.shape
        low_h = max(1, (height + patch_size - 1) // patch_size)
        low_w = max(1, (width + patch_size - 1) // patch_size)
        random_values = torch.rand(
            batch,
            1,
            low_h,
            low_w,
            device=sar.device,
            generator=generator,
        )
        mask = (random_values < float(mask_ratio)).to(sar.dtype)
        return F.interpolate(mask, size=(height, width), mode="nearest")

    def forward(
        self,
        sar: torch.Tensor,
        mask_ratio: float = 0.60,
        patch_size: int = 16,
    ) -> Dict[str, torch.Tensor]:
        mask = self.block_mask(sar, mask_ratio, patch_size)
        masked = sar * (1.0 - mask)
        features = self.sar_encoder(masked)
        reconstruction = self.reconstruction_head(self.decoder(features))
        reconstruction = F.interpolate(
            reconstruction,
            size=sar.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "masked_sar": masked,
        }


def build_h4_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH4:
    return DamageFormerRGBSARH4(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H4_VARIANTS",
    "H4_VARIANTS",
    "BoundedFlowAlignment",
    "ReliabilityGatedCommonUniqueStage",
    "DamageFormerRGBSARH4",
    "SARMaskedAutoencoder",
    "build_h4_damageformer",
    "hierarchical_probabilities",
]
