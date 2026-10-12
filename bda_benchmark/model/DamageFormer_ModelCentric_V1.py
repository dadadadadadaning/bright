from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from .model_centric_plugins import (
    HierarchicalOccurrenceSeverityDecoder,
    RGBStructureAdapter,
    RegistrationTolerantInteraction,
    SARScatteringFrequencyExperts,
)


@dataclass(frozen=True)
class Variant:
    rgb: bool = False
    sar: bool = False
    interaction: bool = False
    hierarchy: bool = False


VARIANTS: Mapping[str, Variant] = {
    "p0_parent": Variant(),
    "h1_hierarchy": Variant(hierarchy=True),
    "h2_rgb_structure": Variant(rgb=True),
    "h3_sar_frequency": Variant(sar=True),
    "h4_local_interaction": Variant(interaction=True),
    "h5_modality_specific": Variant(rgb=True, sar=True),
    "h6_full": Variant(rgb=True, sar=True, interaction=True, hierarchy=True),
}


def checkpoint_state(payload) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                state = dict(payload[key])
                return {
                    name[7:] if name.startswith("module.") else name: value
                    for name, value in state.items()
                }
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return {
                name[7:] if name.startswith("module.") else name: value
                for name, value in payload.items()
            }
    raise ValueError("unsupported parent checkpoint format")


class DamageFormerModelCentricV1(DamageFormer_S8_MICB_DSMIL):
    """Checkpoint-compatible, zero-initialized model-centric P3 extension."""

    plugin_prefixes = ("rgb_structure.", "sar_frequency.", "interaction.", "hierarchy.")

    def __init__(self, variant: str = "h1_hierarchy") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose {tuple(VARIANTS)}")
        super().__init__(mil_residual_scale=0.0)
        self.variant = variant
        self.variant_config = VARIANTS[variant]
        self.rgb_structure = RGBStructureAdapter(192)
        self.sar_frequency = SARScatteringFrequencyExperts(192)
        self.interaction = RegistrationTolerantInteraction()
        self.hierarchy = HierarchicalOccurrenceSeverityDecoder()

    def load_parent_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint_state(payload)
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key for key in incompatible.missing_keys if not key.startswith(self.plugin_prefixes)
        ]
        if unexpected or missing:
            raise RuntimeError(f"parent checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    def freeze_parent(self) -> None:
        active = {
            "rgb_structure.": self.variant_config.rgb,
            "sar_frequency.": self.variant_config.sar,
            "interaction.": self.variant_config.interaction,
            "hierarchy.": self.variant_config.hierarchy,
        }
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                any(name.startswith(prefix) and enabled for prefix, enabled in active.items())
            )

    def plugin_parameters(self):
        return [parameter for name, parameter in self.named_parameters() if name.startswith(self.plugin_prefixes)]

    def set_plugin_training(self) -> None:
        """Keep frozen BatchNorm/statistics fixed while training active plugins."""
        self.eval()
        active = (
            (self.rgb_structure, self.variant_config.rgb),
            (self.sar_frequency, self.variant_config.sar),
            (self.interaction, self.variant_config.interaction),
            (self.hierarchy, self.variant_config.hierarchy),
        )
        for module, enabled in active:
            module.train(enabled)

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        return_aux: bool = False,
    ):
        _, pre1, pre2, pre3, pre4 = self.encoder_1(pre_data)
        _, post1, post2, post3, post4 = self.encoder_2(post_data)
        pre1, pre2, pre3, pre4 = map(self._to_nchw, (pre1, pre2, pre3, pre4))
        post1, post2, post3, post4 = map(self._to_nchw, (post1, post2, post3, post4))

        # Localization remains on the untouched RGB parent path.
        parent_rgb = self.fusion_layer_1(self._multi_scale_fuse(pre1, pre2, pre3, pre4))
        parent_sar = self.fusion_layer_2(self._multi_scale_fuse(post1, post2, post3, post4))
        loc_logits_feature = self.clf_1(parent_rgb)

        rgb2 = self.rgb_structure(pre2) if self.variant_config.rgb else pre2
        sar_diagnostics: Dict[str, torch.Tensor] = {}
        if self.variant_config.sar:
            post2, sar_diagnostics = self.sar_frequency(post2, return_diagnostics=True)
        damage_rgb = self.fusion_layer_1(self._multi_scale_fuse(pre1, rgb2, pre3, pre4))
        damage_sar = self.fusion_layer_2(self._multi_scale_fuse(post1, post2, post3, post4))

        parent_fused = self.fusion_layer_3(torch.cat((damage_rgb, damage_sar), dim=1))
        interaction_diagnostics: Dict[str, torch.Tensor] = {}
        if self.variant_config.interaction:
            interaction = self.interaction(damage_rgb, damage_sar, parent_fused)
            fused = interaction.feature
            interaction_diagnostics = {
                "interaction_confidence": interaction.confidence,
                "interaction_entropy": interaction.entropy,
                "interaction_offset": interaction.offset,
            }
        else:
            fused = parent_fused

        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior
        fused_roi = fused * roi_weight
        flat_logits = self.clf_2(fused_roi)
        true_parent_fused = self.fusion_layer_3(torch.cat((parent_rgb, parent_sar), dim=1))
        true_parent_flat = self.clf_2(true_parent_fused * roi_weight)
        hierarchy_diagnostics: Dict[str, torch.Tensor] = {}
        if self.variant_config.hierarchy:
            hierarchy_diagnostics = self.hierarchy(fused_roi, flat_logits)
            damage_logits_feature = hierarchy_diagnostics["logits"]
        else:
            damage_logits_feature = flat_logits

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
            "roi_weight": roi_weight,
            "parent_rgb_feature": parent_rgb,
            "parent_sar_feature": parent_sar,
            "rgb_damage_feature": damage_rgb,
            "sar_damage_feature": damage_sar,
            "fused_roi": fused_roi,
            "flat_logits_feature": flat_logits,
            "parent_logits_feature": true_parent_flat,
            **sar_diagnostics,
            **interaction_diagnostics,
            **hierarchy_diagnostics,
        }

    def diagnostics(self) -> Dict[str, float]:
        return {
            "rgb_gain": float(torch.tanh(self.rgb_structure.gain.detach())),
            "sar_gain": float(torch.tanh(self.sar_frequency.gain.detach())),
            "interaction_gain": float(torch.tanh(self.interaction.gain.detach())),
            "hierarchy_gain": float(torch.tanh(self.hierarchy.gain.detach())),
        }


__all__ = ["DamageFormerModelCentricV1", "VARIANTS", "Variant", "checkpoint_state"]
