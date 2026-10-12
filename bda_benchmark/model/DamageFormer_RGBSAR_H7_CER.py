# -*- coding: utf-8 -*-
"""H7-CER: conservative counterfactual routing between frozen P3 and H6-A2."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_Heterogeneous import ConvGNAct


class DamagedCounterfactualRouter(nn.Module):
    """Allow A2 to propose *damaged* corrections in P3/A2 disagreement areas.

    The validation oracle showed a positive A2 correction balance only for the
    damaged ground-truth class.  Background, intact and destroyed proposals are
    therefore disabled in this first controlled H7 experiment.  The event
    context consists of image-level native-SAR statistics, never an event ID.
    """

    def __init__(
        self,
        hidden_channels: int = 24,
        gate_bias: float = -4.0,
        maximum_transition: float = 0.50,
    ) -> None:
        super().__init__()
        if not 0.0 < maximum_transition <= 1.0:
            raise ValueError("maximum_transition must be in (0, 1]")
        # P3 prob(4), A2 prob(4), abs disagreement(4), entropy(2), margin(2),
        # native-SAR global mean/std(2), P3 building support(1) = 19 channels.
        self.route_predictor = nn.Sequential(
            ConvGNAct(19, hidden_channels, 3),
            ConvGNAct(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.route_predictor[-1].weight)
        nn.init.constant_(self.route_predictor[-1].bias, float(gate_bias))
        self.transition = nn.Parameter(torch.zeros(()))
        self.maximum_transition = float(maximum_transition)

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        return -(probability * probability.clamp_min(1e-7).log()).sum(
            1, keepdim=True
        )

    @staticmethod
    def _margin(probability: torch.Tensor) -> torch.Tensor:
        top2 = probability.topk(2, dim=1).values
        return top2[:, :1] - top2[:, 1:2]

    def forward(
        self,
        p3_logits: torch.Tensor,
        native_logits: torch.Tensor,
        native_sar: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if native_logits.shape[-2:] != p3_logits.shape[-2:]:
            native_logits = F.interpolate(
                native_logits,
                size=p3_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p3_probability = p3_logits.float().softmax(1).to(p3_logits.dtype)
        native_probability = native_logits.float().softmax(1).to(native_logits.dtype)
        p3_prediction = p3_probability.argmax(1, keepdim=True)
        native_prediction = native_probability.argmax(1, keepdim=True)
        building_support = (1.0 - p3_probability[:, :1]).detach()

        if native_sar is None:
            native_sar = getattr(self, "_native_sar_context", None)
        if native_sar is None:
            sar_mean = p3_probability.new_zeros((p3_probability.shape[0], 1, 1, 1))
            sar_std = torch.zeros_like(sar_mean)
        else:
            native_sar = native_sar[:, :1].float()
            sar_mean = native_sar.mean((2, 3), keepdim=True).to(p3_logits.dtype)
            # Population std remains finite for a 1x1 crop.  The default
            # unbiased estimator has zero degrees of freedom in that case and
            # returns NaN, which can contaminate an otherwise zero-safe route.
            sar_std = native_sar.std(
                (2, 3), keepdim=True, unbiased=False
            ).to(p3_logits.dtype)
        spatial_size = p3_logits.shape[-2:]
        sar_mean = sar_mean.expand(-1, -1, *spatial_size)
        sar_std = sar_std.expand(-1, -1, *spatial_size)

        features = torch.cat(
            [
                p3_probability,
                native_probability,
                (p3_probability - native_probability).abs(),
                self._entropy(p3_probability),
                self._entropy(native_probability),
                self._margin(p3_probability),
                self._margin(native_probability),
                sar_mean,
                sar_std,
                building_support,
            ],
            dim=1,
        )
        route_logit = self.route_predictor(features)
        # Only A2-damaged proposals that disagree with P3 are eligible.
        eligibility = (
            (native_prediction == 2)
            & (p3_prediction != native_prediction)
            & (building_support > 0.05)
        ).to(route_logit.dtype)
        route_gate = torch.sigmoid(route_logit) * eligibility
        # A counterfactual *acceptance* router may move only from P3 towards
        # A2.  A negative transition would silently reverse A2's evidence.
        # This straight-through positive projection keeps the forward value
        # exactly zero at initialisation while retaining a usable gradient.
        positive_transition = self.transition + (
            self.transition.clamp_min(0.0) - self.transition
        ).detach()
        transition = self.maximum_transition * torch.tanh(positive_transition)
        routed_delta = transition * route_gate * (native_logits - p3_logits)
        output = p3_logits + routed_delta
        return output, {
            "cer_route_logit": route_logit,
            "cer_eligible_mask": eligibility,
            "cer_route_gate": route_gate,
            "cer_transition": transition.reshape(1, 1, 1, 1),
            "cer_residual_delta_abs": routed_delta.abs().mean(1, keepdim=True),
            "cer_p3_prediction": p3_prediction.detach(),
            "cer_native_prediction": native_prediction.detach(),
            "safe_residual_sar_gate": route_gate,
            "safe_class_transition": torch.cat(
                [transition.new_zeros(2), transition.reshape(1), transition.new_zeros(1)]
            ).reshape(1, 4, 1, 1),
            "safe_residual_delta_abs": routed_delta.abs().mean(1, keepdim=True),
            "native_p3_delta_abs": (native_logits - p3_logits).abs().mean(
                1, keepdim=True
            ),
        }


class DamageFormerRGBSARH7CER(DamageFormerRGBSARH6SDCR):
    def __init__(self, pretrained_rgb: bool = True) -> None:
        super().__init__(variant="b1r", pretrained_rgb=pretrained_rgb)
        self.variant = "cer_damaged"
        self.safe_residual_router = DamagedCounterfactualRouter()

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        # The H6 parent calls the router with two logits tensors.  Keep only a
        # short-lived tensor context so the router can compute scene statistics
        # without changing the validated H6 forward contract.
        self.safe_residual_router._native_sar_context = post_sar[:, :1]
        try:
            return super().forward(pre_rgb, post_sar, return_dict=return_dict)
        finally:
            del self.safe_residual_router._native_sar_context


def build_h7_cer_damageformer(pretrained_rgb: bool = True) -> DamageFormerRGBSARH7CER:
    return DamageFormerRGBSARH7CER(pretrained_rgb=pretrained_rgb)


__all__ = [
    "DamagedCounterfactualRouter",
    "DamageFormerRGBSARH7CER",
    "build_h7_cer_damageformer",
]
