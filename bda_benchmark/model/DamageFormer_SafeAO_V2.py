from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from .DamageFormer_ModelCentric_V1 import checkpoint_state
from .safe_affected_plugins import SafeAffectedResidual, ShallowAffectedEvidence


@dataclass(frozen=True)
class AOVariant:
    rgb: bool
    sar: bool
    local: bool
    confidence: bool


VARIANTS: Mapping[str, AOVariant] = {
    "a0_parent": AOVariant(False, False, False, False),
    "a1_rgb_shallow": AOVariant(True, False, False, False),
    "a2_sar_shallow": AOVariant(False, True, False, False),
    "a3_same_position": AOVariant(True, True, False, False),
    "a4_local_match": AOVariant(True, True, True, False),
    "a5_confident_local": AOVariant(True, True, True, True),
}


class DamageFormerSafeAOV2(DamageFormer_S8_MICB_DSMIL):
    """P3 plus shallow evidence that can only promote Affected occurrence."""

    plugin_prefixes = ("affected_evidence.", "affected_residual.")

    def __init__(self, variant: str = "a5_confident_local", hidden: int = 32) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown AO variant {variant!r}; choose {tuple(VARIANTS)}")
        super().__init__(mil_residual_scale=0.0)
        self.variant = variant
        self.variant_config = VARIANTS[variant]
        self.affected_evidence = ShallowAffectedEvidence(
            hidden=hidden,
            use_rgb=self.variant_config.rgb,
            use_sar=self.variant_config.sar,
            local_match=self.variant_config.local,
            confidence_gate=self.variant_config.confidence,
        )
        self.affected_residual = SafeAffectedResidual(maximum_shift=4.0)

    def load_parent_checkpoint(self, path: str) -> None:
        state = checkpoint_state(torch.load(path, map_location="cpu", weights_only=False))
        incompatible = self.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith(self.plugin_prefixes)]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"parent checkpoint mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}"
            )

    def freeze_parent(self) -> None:
        enabled = self.variant != "a0_parent"
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(enabled and name.startswith(self.plugin_prefixes))

    def set_plugin_training(self) -> None:
        self.eval()
        self.affected_evidence.train(self.variant != "a0_parent")
        self.affected_residual.train(self.variant != "a0_parent")

    def project_safe_parameters(self) -> None:
        self.affected_residual.project_parameters()

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)
        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior
        fused = self.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        fused_roi = fused * roi_weight
        parent_logits_feature = self.clf_2(fused_roi)

        evidence_diagnostics: Dict[str, torch.Tensor] = {}
        occurrence_logits = torch.zeros_like(parent_logits_feature[:, :1])
        delta = torch.zeros_like(occurrence_logits)
        if self.variant != "a0_parent":
            occurrence_logits, confidence, evidence_diagnostics = self.affected_evidence(
                pre_data, post_data, parent_logits_feature.shape[-2:]
            )
            with torch.no_grad():
                parent_probability = parent_logits_feature.float().softmax(1)
                nonaffected = parent_probability[:, 0:2].sum(1, keepdim=True)
                # Keep background gate-miss reachable, but emphasize P3 building support.
                support = (0.10 + 0.90 * p_building.detach()) * nonaffected
            damage_logits_feature, delta = self.affected_residual(
                parent_logits_feature, occurrence_logits, support, confidence
            )
        else:
            damage_logits_feature = parent_logits_feature

        output_loc = F.interpolate(
            loc_logits_feature, pre_data.shape[-2:], mode="bilinear", align_corners=False
        )
        output_damage = F.interpolate(
            damage_logits_feature, post_data.shape[-2:], mode="bilinear", align_corners=False
        )
        if not return_aux:
            return output_loc, output_damage
        return output_loc, output_damage, {
            "p_building": p_building,
            "parent_logits_feature": parent_logits_feature,
            "occurrence_logits": occurrence_logits,
            "affected_delta": delta,
            "affected_gain": self.affected_residual.gain,
            **evidence_diagnostics,
        }


__all__ = ["AOVariant", "DamageFormerSafeAOV2", "VARIANTS"]
