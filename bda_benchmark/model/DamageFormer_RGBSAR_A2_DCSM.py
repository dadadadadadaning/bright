# -*- coding: utf-8 -*-
"""A2-DCSM: damage-conditioned structural migration for the native SAR expert.

The branch is deliberately built on the validated H6-A2 expert.  It does not
change RGB localisation, SDCR, the damage decoder, or the supervised losses.
The controlled variants only change how the native-SAR pathway is prepared:

* M0: exact H6-A2 control;
* M1: damage-conditioned feature-predictive pretraining of the SAR encoder;
* M2: M1 plus one fixed Scharr structural view that exists during pretraining;
* M3: M2 plus migration of the stage-3/4 SAR projection neck.

The optical anchor is used only by the *pretraining target builder* on
unchanged pixels.  It is not a teacher-logit loss and it is absent from the
supervised fine-tuning objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_Heterogeneous import ConvGNAct


VALID_A2_DCSM_VARIANTS = ("m0", "m1", "m2", "m3")


@dataclass(frozen=True)
class A2DCSMVariant:
    name: str
    feature_pretraining: bool
    structural_adapter: bool
    migrate_deep_neck: bool


A2_DCSM_VARIANTS: Mapping[str, A2DCSMVariant] = {
    "m0": A2DCSMVariant("A2-M0-Locked-H6A2", False, False, False),
    "m1": A2DCSMVariant(
        "A2-M1-Damage-Conditioned-SAR-Feature-Pretraining", True, False, False
    ),
    "m2": A2DCSMVariant(
        "A2-M2-Structural-SAR-Feature-Pretraining", True, True, False
    ),
    "m3": A2DCSMVariant(
        "A2-M3-Full-Deep-SAR-Pathway-Migration", True, True, True
    ),
}


class FixedScharrMagnitude(nn.Module):
    """A deterministic display-domain structural view.

    BRIGHT currently supplies an 8-bit display SAR product, not calibrated
    sigma0/gamma0.  Consequently this module makes no claim of physical
    speckle modelling; it exposes only a stable local-gradient view.
    """

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]
        ) / 32.0
        kernel_y = kernel_x.t().contiguous()
        self.register_buffer("kernel_x", kernel_x.reshape(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.reshape(1, 1, 3, 3))

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError(f"Scharr view expects [B,1,H,W], got {tuple(sar.shape)}")
        gx = F.conv2d(sar.float(), self.kernel_x.float(), padding=1)
        gy = F.conv2d(sar.float(), self.kernel_y.float(), padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        scale = magnitude.mean((2, 3), keepdim=True).clamp_min(1e-4)
        return (magnitude / scale).to(sar.dtype)


class StructuralSARAdapter(nn.Module):
    """Inject one fixed structural view through an exactly-safe residual.

    The candidate branch is non-zero while ``transition`` starts at zero.
    Therefore M2/M3 are exactly M0 at construction, but the transition still
    receives a first-step gradient.
    """

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.structure = FixedScharrMagnitude()
        self.candidate = nn.Sequential(
            ConvGNAct(2, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 3, padding=1, bias=True),
        )
        nn.init.kaiming_normal_(
            self.candidate[-1].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.candidate[-1].bias)
        self.transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, sar: torch.Tensor, return_diagnostics: bool = False
    ):
        structure = self.structure(sar)
        residual = self.candidate(torch.cat([sar, structure], dim=1))
        gain = torch.tanh(self.transition)
        output = sar + gain * residual
        if not return_diagnostics:
            return output
        return output, {
            "dcsm_structural_gain": gain.reshape(1, 1, 1, 1),
            "dcsm_structure_abs": structure.abs().mean(1, keepdim=True),
            "dcsm_structural_delta_abs": (gain * residual).abs().mean(
                1, keepdim=True
            ),
        }


def unchanged_mask_from_damage(
    damage: torch.Tensor, ignore_index: int = 255
) -> torch.Tensor:
    """Return the train-label-only background/intact anchor mask."""

    if damage.ndim != 3:
        raise ValueError(f"damage target must be [B,H,W], got {tuple(damage.shape)}")
    valid = damage.ne(ignore_index)
    unchanged = (damage.eq(0) | damage.eq(1)) & valid
    return unchanged[:, None].float()


def damage_conditioned_target(
    sar_target: torch.Tensor,
    rgb_anchor: torch.Tensor,
    unchanged: torch.Tensor,
    anchor_strength: float,
) -> torch.Tensor:
    """Blend RGB semantics only where the train label declares no damage."""

    if sar_target.shape != rgb_anchor.shape:
        raise ValueError("SAR target and RGB anchor must share a shape")
    weight = F.interpolate(
        unchanged.float(),
        size=sar_target.shape[-2:],
        mode="nearest",
    ).to(sar_target.dtype)
    strength = float(anchor_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("anchor_strength must lie in [0, 1]")
    return sar_target + strength * weight * (rgb_anchor - sar_target)


def class_balanced_region_weights(
    damage: torch.Tensor,
    size: Tuple[int, int],
    ignore_index: int = 255,
    class_weights: Sequence[float] = (0.25, 1.0, 2.0, 2.0),
) -> torch.Tensor:
    """Weight buildings/classes without introducing another objective."""

    if len(class_weights) != 4:
        raise ValueError("class_weights must contain background/intact/damaged/destroyed")
    resized = F.interpolate(
        damage[:, None].float(), size=size, mode="nearest"
    )[:, 0].long()
    valid = resized.ne(ignore_index)
    clamped = resized.clamp(0, 3)
    table = torch.as_tensor(class_weights, device=damage.device, dtype=torch.float32)
    return table[clamped] * valid.float()


class DCSMFeaturePredictor(nn.Module):
    """Small pretraining-only predictor for common+unique SAR tokens."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(2 * channels, 2 * channels, 1),
            nn.Conv2d(2 * channels, 2 * channels, 1, bias=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def masked_cosine_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    patch_mask: torch.Tensor,
    region_weights: torch.Tensor,
) -> torch.Tensor:
    """The single DCSM pretraining objective."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    mask = F.interpolate(
        patch_mask.float(), size=prediction.shape[-2:], mode="nearest"
    )[:, 0]
    if region_weights.shape != mask.shape:
        raise ValueError("region_weights must match the resized patch mask")
    distance = 1.0 - F.cosine_similarity(
        prediction.float(), target.detach().float(), dim=1, eps=1e-6
    )
    weight = mask * region_weights.float()
    return (distance * weight).sum() / weight.sum().clamp_min(1.0)


class DamageFormerRGBSARA2DCSM(DamageFormerRGBSARH6SDCR):
    """H6-A2 with a controlled, pretraining-consistent SAR input contract."""

    def __init__(
        self,
        variant: str = "m0",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
    ) -> None:
        variant = variant.lower()
        if variant not in VALID_A2_DCSM_VARIANTS:
            raise ValueError(f"unknown A2-DCSM variant {variant!r}")
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.dcsm_config = A2_DCSM_VARIANTS[variant]
        self.structural_adapter: nn.Module
        if self.dcsm_config.structural_adapter:
            self.structural_adapter = StructuralSARAdapter()
        else:
            self.structural_adapter = nn.Identity()

    def adapt_native_sar(
        self, native_sar: torch.Tensor, return_diagnostics: bool = False
    ):
        if isinstance(self.structural_adapter, StructuralSARAdapter):
            return self.structural_adapter(
                native_sar, return_diagnostics=return_diagnostics
            )
        if return_diagnostics:
            zero = native_sar.new_zeros((1, 1, 1, 1))
            return native_sar, {
                "dcsm_structural_gain": zero,
                "dcsm_structure_abs": native_sar.abs().mean(1, keepdim=True),
                "dcsm_structural_delta_abs": torch.zeros_like(native_sar),
            }
        return native_sar

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
            raise ValueError("A2-DCSM expects SAR [B,1,H,W] or H4 audit [B,2,H,W]")
        adapted, structural_diagnostics = self.adapt_native_sar(
            post_sar[:, :1], return_diagnostics=True
        )
        forwarded_sar = (
            torch.cat([adapted, post_sar[:, 1:]], dim=1)
            if post_sar.shape[1] == 2
            else adapted
        )
        outputs = super().forward(pre_rgb, forwarded_sar, return_dict=True)
        outputs.update(structural_diagnostics)
        outputs["dcsm_variant_id"] = outputs["output_dam"].new_full(
            (1, 1, 1, 1), float(VALID_A2_DCSM_VARIANTS.index(self.variant))
        )
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_a2_dcsm_damageformer(
    variant: str, pretrained_rgb: bool = True
) -> DamageFormerRGBSARA2DCSM:
    return DamageFormerRGBSARA2DCSM(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "A2_DCSM_VARIANTS",
    "VALID_A2_DCSM_VARIANTS",
    "A2DCSMVariant",
    "DCSMFeaturePredictor",
    "DamageFormerRGBSARA2DCSM",
    "FixedScharrMagnitude",
    "StructuralSARAdapter",
    "build_a2_dcsm_damageformer",
    "class_balanced_region_weights",
    "damage_conditioned_target",
    "masked_cosine_prediction_loss",
    "unchanged_mask_from_damage",
]
