"""A clean, one-variable building-prior ablation for ChangeMamba-BRIGHT.

``cm0_original`` is the unmodified ChangeMambaMMBDA model.
``cm1_building_prior`` changes only the feature presented to ``main_clf``.
The prior is detached and the residual transition is zero-initialised, so CM1
is functionally identical to CM0 at initialisation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
from changedetection.models.builders import resize_to_input


VALID_VARIANTS = (
    "cm0_original",
    "cm1_building_prior",
    "cm1s_building_prior_stable",
)


def apply_soft_building_prior(
    damage_feature: torch.Tensor,
    building_probability: torch.Tensor,
    transition: torch.Tensor,
    *,
    roi_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an identity-initialised, bounded soft building prior."""
    if not 0.0 <= float(roi_floor) <= 1.0:
        raise ValueError("roi_floor must be in [0, 1]")
    if damage_feature.ndim != 4 or building_probability.ndim != 4:
        raise ValueError("features and probabilities must be BCHW tensors")
    if building_probability.shape[1] != 1:
        raise ValueError("building_probability must have one channel")
    if building_probability.shape[-2:] != damage_feature.shape[-2:]:
        building_probability = torch.nn.functional.interpolate(
            building_probability,
            size=damage_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    target_weight = float(roi_floor) + (1.0 - float(roi_floor)) * (
        building_probability.detach().float()
    )
    strength = torch.tanh(transition.float())
    weight = 1.0 + strength * (target_weight - 1.0)
    guided = damage_feature * weight.to(damage_feature.dtype)
    return guided, weight


class ChangeMambaMMBDABuildingPrior(ChangeMambaMMBDA):
    """Original ChangeMamba plus a detached RGB building-probability prior."""

    def __init__(self, *args, roi_floor: float = 0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_floor = float(roi_floor)
        self.building_transition = nn.Parameter(torch.zeros(()))

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor):
        pre_features = self.encoder_1(pre_data)
        post_features = self.encoder_2(post_data)

        building_feature = self.decoder_building(pre_features)
        building_logits_low = self.aux_clf(building_feature)
        building_probability = building_logits_low.float().softmax(1)[:, 1:2]

        damage_feature = self.decoder_damage(pre_features, post_features)
        guided_damage, _ = apply_soft_building_prior(
            damage_feature,
            building_probability,
            self.building_transition,
            roi_floor=self.roi_floor,
        )
        damage_logits_low = self.main_clf(guided_damage)

        output_building = resize_to_input(building_logits_low, pre_data)
        output_damage = resize_to_input(damage_logits_low, post_data)
        return output_building, output_damage

    def experiment_metadata(self) -> Dict[str, object]:
        return {
            "variant": "cm1_building_prior",
            "building_prior": "detached RGB building probability",
            "roi_floor": self.roi_floor,
            "transition": "tanh scalar, zero initialised",
            "native_sar_adapter": False,
            "diff_fusion": False,
        }


def build_model(variant: str, *, pretrained, roi_floor: float, **kwargs):
    variant = str(variant).lower()
    if variant == "cm0_original":
        return ChangeMambaMMBDA(
            output_building=2,
            output_damage=4,
            pretrained=pretrained,
            **kwargs,
        )
    if variant in {"cm1_building_prior", "cm1s_building_prior_stable"}:
        return ChangeMambaMMBDABuildingPrior(
            output_building=2,
            output_damage=4,
            pretrained=pretrained,
            roi_floor=roi_floor,
            **kwargs,
        )
    raise ValueError(f"unknown variant: {variant!r}")
