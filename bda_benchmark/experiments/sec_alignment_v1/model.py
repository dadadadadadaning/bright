# -*- coding: utf-8 -*-
"""D1 architecture reused under a separate SEC checkpoint contract."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_SALT_CDE import (
    DamageFormerSALT,
    StableAnchorTransportStage,
    warp_with_offsets,
)

from .objectives import SEC_VARIANTS


class ShiftEquivariantDamageFormer(DamageFormerSALT):
    """D1-compatible SEC model with an optional parameter-free C2F schedule."""

    def __init__(self, variant: str, pretrained_rgb: bool = True) -> None:
        variant = str(variant).lower()
        if variant not in SEC_VARIANTS:
            raise ValueError(f"SEC variant must be one of {sorted(SEC_VARIANTS)}")
        super().__init__("d1", pretrained_rgb=pretrained_rgb)
        self.variant = variant
        self.sec_variant = SEC_VARIANTS[variant]
        self.experiment_name = "SEC-Shift-Equivariant-Correspondence"
        self.resume_protocol = "SEC_resume_v1"
        self.best_weights_protocol = "SEC_best_weights_v1"

    def checkpoint_contract(self):
        return {
            "family": "SEC_v1",
            "variant": self.variant,
            "architecture": (
                "SALT_D1_coarse_to_fine_transport"
                if self.sec_variant.coarse_to_fine
                else "SALT_D1_exact"
            ),
            "offset_delta": self.sec_variant.offset_delta,
            "prediction_consistency": self.sec_variant.prediction_consistency,
            "alignment_only": self.sec_variant.alignment_only,
            "support_mode": self.sec_variant.support_mode,
            "coarse_to_fine": self.sec_variant.coarse_to_fine,
            "official_instances": False,
            "test_access": False,
        }

    @staticmethod
    def _resize_offset(
        offset: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize feature-cell offsets while preserving pixel displacement."""
        source_h, source_w = offset.shape[-2:]
        target_h, target_w = target_size
        resized = F.interpolate(
            offset.float(), size=target_size, mode="bilinear", align_corners=False
        )
        scale = resized.new_tensor(
            (target_w / float(source_w), target_h / float(source_h))
        ).view(1, 2, 1, 1)
        return resized * scale

    def _decode_coarse_to_fine(
        self,
        rgb_stages: Sequence[torch.Tensor],
        sar_stages: Sequence[torch.Tensor],
        size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Stage-3 coarse transport followed by a Stage-2 residual match."""
        loc_feature = self.localization_decoder(rgb_stages)
        loc_low = self.localization_head(loc_feature)
        building = loc_low.softmax(1)[:, 1:2]
        joint: List[torch.Tensor] = [None] * len(rgb_stages)  # type: ignore[list-item]
        bases: List[torch.Tensor] = [None] * len(rgb_stages)  # type: ignore[list-item]
        diagnostics: Dict[str, torch.Tensor] = {}

        # Stages 1 and 4 retain the exact D1 colocated fusion.
        for index in (0, 3):
            value, info = self.joint_stages[index](rgb_stages[index], sar_stages[index])
            joint[index] = value
            bases[index] = value
            diagnostics.update(
                {f"stage{index + 1}_{key}": item for key, item in info.items()}
            )

        coarse_module = self.joint_stages[2]
        fine_module = self.joint_stages[1]
        if not isinstance(coarse_module, StableAnchorTransportStage) or not isinstance(
            fine_module, StableAnchorTransportStage
        ):
            raise RuntimeError("coarse-to-fine SEC requires D1 transport at stages 2/3")

        coarse_value, coarse_info = coarse_module(
            rgb_stages[2], sar_stages[2], building.detach()
        )
        coarse_base = coarse_info.pop("base_feature")
        coarse_offset = torch.cat(
            (coarse_info["offset_x"], coarse_info["offset_y"]), dim=1
        )
        joint[2], bases[2] = coarse_value, coarse_base
        diagnostics.update(
            {f"stage3_{key}": item for key, item in coarse_info.items()}
        )

        prior = self._resize_offset(coarse_offset, rgb_stages[1].shape[-2:])
        coarse_aligned_sar = warp_with_offsets(
            sar_stages[1], prior.to(dtype=sar_stages[1].dtype)
        )
        fine_value, fine_info = fine_module(
            rgb_stages[1], coarse_aligned_sar, building.detach()
        )
        fine_base = fine_info.pop("base_feature")
        residual = torch.cat(
            (fine_info["offset_x"], fine_info["offset_y"]), dim=1
        )
        total = prior + residual
        fine_info["residual_offset_x"] = residual[:, 0:1]
        fine_info["residual_offset_y"] = residual[:, 1:2]
        fine_info["coarse_prior_offset_x"] = prior[:, 0:1]
        fine_info["coarse_prior_offset_y"] = prior[:, 1:2]
        fine_info["offset_x"] = total[:, 0:1]
        fine_info["offset_y"] = total[:, 1:2]
        joint[1], bases[1] = fine_value, fine_base
        diagnostics.update(
            {f"stage2_{key}": item for key, item in fine_info.items()}
        )

        damage_low = self.damage_head(self.damage_decoder(joint))
        base_low = self.damage_head(self.damage_decoder(bases))
        return {
            "output_loc": F.interpolate(
                loc_low, size=size, mode="bilinear", align_corners=False
            ),
            "output_dam": F.interpolate(
                damage_low, size=size, mode="bilinear", align_corners=False
            ),
            "base_damage_logits": F.interpolate(
                base_low, size=size, mode="bilinear", align_corners=False
            ),
            "sar_aux_logits": F.interpolate(
                self.sar_aux_head(sar_stages[-1]),
                size=size,
                mode="bilinear",
                align_corners=False,
            ),
            "localization_probability_low": building,
            **diagnostics,
        }

    def _decode(
        self,
        rgb_stages: Sequence[torch.Tensor],
        sar_stages: Sequence[torch.Tensor],
        size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if self.sec_variant.coarse_to_fine:
            return self._decode_coarse_to_fine(rgb_stages, sar_stages, size)
        return super()._decode(rgb_stages, sar_stages, size)


def build_sec_model(variant: str, pretrained_rgb: bool = True):
    return ShiftEquivariantDamageFormer(variant, pretrained_rgb=pretrained_rgb)


__all__ = ["ShiftEquivariantDamageFormer", "build_sec_model"]
