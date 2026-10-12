"""Function-preserving P3 task-structure transfer for ChangeMamba-BRIGHT.

``cm_p2`` adds a detached RGB building prior inside the damage decoder.  A
zero-initialised scalar makes the initial function exactly the accepted B0.

``cm_p3`` additionally factorises the existing four-class decision into
Intact-vs-Affected and Damaged-vs-Destroyed conditional decisions.  The two
new residual heads are zero-initialised, so their initial probabilities are
exactly those of the original four-class head.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from changedetection.checkpoints import extract_model_state_dict
from changedetection.experiments.native_sar.model import (
    ChangeMambaMMBDANativeSAR,
)


VALID_P3_TRANSFER_VARIANTS = ("cm_p2", "cm_p3")


def apply_soft_building_roi(
    feature: torch.Tensor,
    building_probability: torch.Tensor,
    transition: torch.Tensor,
    *,
    roi_floor: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a bounded, identity-initialised P3-style soft building prior."""
    if not 0.0 <= float(roi_floor) <= 1.0:
        raise ValueError("roi_floor must be in [0,1]")
    if building_probability.shape[-2:] != feature.shape[-2:]:
        building_probability = F.interpolate(
            building_probability,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    target_weight = float(roi_floor) + (1.0 - float(roi_floor)) * (
        building_probability.detach().float()
    )
    strength = torch.tanh(transition.float())
    weight = 1.0 + strength * (target_weight - 1.0)
    return feature * weight.to(feature.dtype), weight


def factorized_damage_log_probs(
    base_logits: torch.Tensor,
    occurrence_delta: torch.Tensor,
    severity_delta: torch.Tensor,
) -> torch.Tensor:
    """Reparameterise four classes without changing them at zero residual.

    Class order is Background, Intact, Damaged, Destroyed.  Background mass is
    retained from the mature four-class head.  Only the conditional building
    distribution is expressed as Intact-vs-Affected and, conditional on being
    affected, Damaged-vs-Destroyed.
    """
    if base_logits.ndim != 4 or base_logits.shape[1] != 4:
        raise ValueError("base_logits must have shape [B,4,H,W]")
    for name, delta in (
        ("occurrence_delta", occurrence_delta),
        ("severity_delta", severity_delta),
    ):
        if delta.shape != base_logits[:, :1].shape:
            raise ValueError(
                f"{name} must have shape {tuple(base_logits[:, :1].shape)}, "
                f"got {tuple(delta.shape)}"
            )

    log_probability = F.log_softmax(base_logits.float(), dim=1)
    background = log_probability[:, :1].exp()
    building_mass = 1.0 - background

    occurrence_base = torch.logsumexp(base_logits[:, 2:4].float(), dim=1, keepdim=True)
    occurrence_base = occurrence_base - base_logits[:, 1:2].float()
    severity_base = base_logits[:, 3:4].float() - base_logits[:, 2:3].float()

    affected = torch.sigmoid(occurrence_base + occurrence_delta.float())
    destroyed_given_affected = torch.sigmoid(
        severity_base + severity_delta.float()
    )
    probability = torch.cat(
        (
            background,
            building_mass * (1.0 - affected),
            building_mass * affected * (1.0 - destroyed_given_affected),
            building_mass * affected * destroyed_given_affected,
        ),
        dim=1,
    )
    probability = probability.clamp_min(1.0e-8)
    probability = probability / probability.sum(1, keepdim=True).clamp_min(1.0e-8)
    return probability.log().to(base_logits.dtype)


class ChangeMambaP3Transfer(ChangeMambaMMBDANativeSAR):
    """NativeSAR ChangeMamba with controlled P3 structural transfers."""

    def __init__(self, *args, variant: str, roi_floor: float = 0.10, **kwargs):
        variant = str(variant).lower()
        if variant not in VALID_P3_TRANSFER_VARIANTS:
            raise ValueError(f"unknown P3-transfer variant: {variant!r}")
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.roi_floor = float(roi_floor)
        self.building_transition = nn.Parameter(torch.zeros(()))
        decoder_channels = int(self.main_clf.in_channels)
        if variant == "cm_p3":
            self.occurrence_delta = nn.Conv2d(decoder_channels, 1, 1)
            self.severity_delta = nn.Conv2d(decoder_channels, 1, 1)
            nn.init.zeros_(self.occurrence_delta.weight)
            nn.init.zeros_(self.occurrence_delta.bias)
            nn.init.zeros_(self.severity_delta.weight)
            nn.init.zeros_(self.severity_delta.bias)

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor):
        if post_data.ndim != 4 or int(post_data.shape[1]) != 1:
            raise ValueError(
                "native SAR input must have shape [B,1,H,W], got "
                f"{tuple(post_data.shape)}"
            )
        adapted_sar = self.sar_adapter(post_data)
        pre_features = self.encoder_1(pre_data)
        post_features = self.encoder_2(adapted_sar)

        building_feature = self.decoder_building(pre_features)
        building_logits_low = self.aux_clf(building_feature)
        building_probability = building_logits_low.float().softmax(1)[:, 1:2]

        damage_feature = self.decoder_damage(pre_features, post_features)
        guided_damage, _weight = apply_soft_building_roi(
            damage_feature,
            building_probability,
            self.building_transition,
            roi_floor=self.roi_floor,
        )
        damage_logits_low = self.main_clf(guided_damage)

        output_building = F.interpolate(
            building_logits_low,
            size=pre_data.shape[-2:],
            mode="bilinear",
        )
        output_damage = F.interpolate(
            damage_logits_low,
            size=post_data.shape[-2:],
            mode="bilinear",
        )
        if self.variant == "cm_p3":
            occurrence_delta = F.interpolate(
                self.occurrence_delta(guided_damage),
                size=post_data.shape[-2:],
                mode="bilinear",
            )
            severity_delta = F.interpolate(
                self.severity_delta(guided_damage),
                size=post_data.shape[-2:],
                mode="bilinear",
            )
            output_damage = factorized_damage_log_probs(
                output_damage, occurrence_delta, severity_delta
            )
        return output_building, output_damage

    def experiment_metadata(self) -> Dict[str, object]:
        return {
            "family": "ChangeMamba NativeSAR + P3 structural transfer",
            "variant": self.variant,
            "building_prior": "detached RGB probability",
            "roi_floor": self.roi_floor,
            "roi_transition": "tanh scalar, zero initialised",
            "hierarchical_damage": self.variant == "cm_p3",
            "test_selection": False,
        }


def load_parent_checkpoint(
    model: ChangeMambaP3Transfer, checkpoint_path: str
) -> Dict[str, object]:
    """Load a trusted local B0/NativeSAR1 checkpoint with a strict state audit.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    The historical ChangeMamba checkpoint also stores NumPy-backed training
    metadata, so the restricted loader rejects it before we can extract the
    model tensors.  This path is supplied explicitly by the experiment owner;
    after loading, the namespace audit below still rejects undeclared model
    keys.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = extract_model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_new = {
        "sar_adapter.weight",
        "sar_adapter.bias",
        "building_transition",
    }
    if model.variant == "cm_p3":
        allowed_new.update(
            {
                "occurrence_delta.weight",
                "occurrence_delta.bias",
                "severity_delta.weight",
                "severity_delta.bias",
            }
        )
    missing = set(incompatible.missing_keys)
    if not missing.issubset(allowed_new) or incompatible.unexpected_keys:
        raise RuntimeError(
            "parent checkpoint changed undeclared namespaces: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if "sar_adapter.weight" in missing:
        model.reset_native_adapter(sar_mean=model.sar_mean, sar_std=model.sar_std)
    with torch.no_grad():
        if "building_transition" in missing:
            model.building_transition.zero_()
        if model.variant == "cm_p3":
            if "occurrence_delta.weight" in missing:
                model.occurrence_delta.weight.zero_()
            if "occurrence_delta.bias" in missing:
                model.occurrence_delta.bias.zero_()
            if "severity_delta.weight" in missing:
                model.severity_delta.weight.zero_()
            if "severity_delta.bias" in missing:
                model.severity_delta.bias.zero_()
    return {
        "checkpoint": checkpoint_path,
        "loaded_keys": len(state),
        "missing_keys": sorted(missing),
        "unexpected_keys": [],
        "function_preserving_new_modules": True,
    }


def freeze_for_p3_transfer(model: ChangeMambaP3Transfer) -> Dict[str, object]:
    """Freeze the accepted body and expose only the declared experiment knobs."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules: Iterable[nn.Module] = (model.sar_adapter,)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.building_transition.requires_grad_(True)
    if model.variant == "cm_p3":
        for module in (model.occurrence_delta, model.severity_delta):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    return {
        "total": sum(value.numel() for value in model.parameters()),
        "trainable": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "names": names,
    }


__all__ = [
    "VALID_P3_TRANSFER_VARIANTS",
    "ChangeMambaP3Transfer",
    "apply_soft_building_roi",
    "factorized_damage_log_probs",
    "load_parent_checkpoint",
    "freeze_for_p3_transfer",
]
