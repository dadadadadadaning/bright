# -*- coding: utf-8 -*-
"""H7-CER-V2: rejectable, pair-conditional selection of frozen experts.

V2-A deliberately opens only two validation-audited proposals:
P3 damaged -> A2 background and P3 destroyed -> A2 background.  P3 and A2
remain frozen; this module learns whether a particular disagreement is useful.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_Heterogeneous import ConvGNAct


KEEP_P3 = 0
USE_A2 = 1
REJECT_BOTH = 2
PAIR_DAMAGED_TO_BACKGROUND = 2 * 4 + 0
PAIR_DESTROYED_TO_BACKGROUND = 3 * 4 + 0
V2A_ALLOWED_PAIR_IDS = (
    PAIR_DAMAGED_TO_BACKGROUND,
    PAIR_DESTROYED_TO_BACKGROUND,
)


class PairConditionalRejectableSelector(nn.Module):
    """Three-way selector with fixed conservative inference constraints."""

    def __init__(
        self,
        hidden_channels: int = 32,
        pair_embedding_channels: int = 8,
        advantage_threshold: float = 0.25,
        reject_threshold: float = 0.30,
        consistency_threshold: float = 0.60,
        route_budget: float = 0.10,
        building_threshold: float = 0.75,
        boundary_threshold: float = 0.08,
    ) -> None:
        super().__init__()
        if not 0.0 < route_budget <= 1.0:
            raise ValueError("route_budget must be in (0, 1]")
        self.pair_embedding = nn.Embedding(16, pair_embedding_channels)
        # 12 expert probabilities/difference + 4 uncertainty/margin +
        # pair embedding + 6 spatial consistency + 5 SAR statistics.
        input_channels = 12 + 4 + pair_embedding_channels + 6 + 5
        self.selector = nn.Sequential(
            ConvGNAct(input_channels, hidden_channels, 3),
            ConvGNAct(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 3, 1, bias=True),
        )
        nn.init.zeros_(self.selector[-1].weight)
        # Exact P3 safety at initialisation: KEEP dominates every pixel.
        with torch.no_grad():
            self.selector[-1].bias.copy_(torch.tensor([4.0, -4.0, -4.0]))
        self.advantage_threshold = float(advantage_threshold)
        self.reject_threshold = float(reject_threshold)
        self.consistency_threshold = float(consistency_threshold)
        self.route_budget = float(route_budget)
        self.building_threshold = float(building_threshold)
        self.boundary_threshold = float(boundary_threshold)
        self._native_sar_context: torch.Tensor | None = None
        self._localization_context: torch.Tensor | None = None

    def set_context(
        self,
        native_sar: torch.Tensor,
        localization_logits: torch.Tensor,
    ) -> None:
        self._native_sar_context = native_sar[:, :1]
        self._localization_context = localization_logits

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        return -(probability * probability.clamp_min(1e-7).log()).sum(
            1, keepdim=True
        )

    @staticmethod
    def _margin(probability: torch.Tensor) -> torch.Tensor:
        top2 = probability.topk(2, dim=1).values
        return top2[:, :1] - top2[:, 1:2]

    @staticmethod
    def _prediction_consistency(
        prediction: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        one_hot = F.one_hot(prediction[:, 0], num_classes=4).permute(0, 3, 1, 2)
        coverage = F.avg_pool2d(
            one_hot.float(), kernel_size, stride=1, padding=kernel_size // 2
        )
        return coverage.gather(1, prediction.long())

    @staticmethod
    def _budget_mask(
        score: torch.Tensor,
        proposed: torch.Tensor,
        disagreement: torch.Tensor,
        fraction: float,
    ) -> torch.Tensor:
        selected = torch.zeros_like(proposed)
        for index in range(score.shape[0]):
            proposal_indices = proposed[index, 0].flatten().nonzero().flatten()
            disagreement_count = int(disagreement[index, 0].sum().item())
            budget = int(disagreement_count * fraction)
            if proposal_indices.numel() == 0 or budget <= 0:
                continue
            count = min(int(proposal_indices.numel()), budget)
            values = score[index, 0].flatten()[proposal_indices]
            top = proposal_indices[values.topk(count, sorted=False).indices]
            selected[index, 0].view(-1)[top] = True
        return selected

    def _context_features(
        self,
        p3_probability: torch.Tensor,
        a2_probability: torch.Tensor,
        p3_prediction: torch.Tensor,
        a2_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        size = p3_probability.shape[-2:]
        pair_id = p3_prediction * 4 + a2_prediction
        pair_feature = self.pair_embedding(pair_id[:, 0]).permute(0, 3, 1, 2)

        localization = self._localization_context
        if localization is None:
            building = (1.0 - p3_probability[:, :1]).detach()
        else:
            if localization.shape[-2:] != size:
                localization = F.interpolate(
                    localization, size=size, mode="bilinear", align_corners=False
                )
            building = localization.float().softmax(1)[:, 1:2].to(
                p3_probability.dtype
            )
        boundary = (
            F.max_pool2d(building, 3, 1, 1)
            + F.max_pool2d(-building, 3, 1, 1)
        ).clamp_min(0.0)

        p3_consistency3 = self._prediction_consistency(p3_prediction, 3)
        p3_consistency7 = self._prediction_consistency(p3_prediction, 7)
        a2_consistency3 = self._prediction_consistency(a2_prediction, 3)
        a2_consistency7 = self._prediction_consistency(a2_prediction, 7)

        sar = self._native_sar_context
        if sar is None:
            sar = building.new_zeros((building.shape[0], 1, *size))
        else:
            sar = sar.float()
            if sar.shape[-2:] != size:
                sar = F.interpolate(sar, size=size, mode="bilinear", align_corners=False)
            sar = sar.to(building.dtype)
        sar_mean3 = F.avg_pool2d(sar, 3, 1, 1)
        sar_square3 = F.avg_pool2d(sar.square(), 3, 1, 1)
        sar_std3 = (sar_square3 - sar_mean3.square()).clamp_min(0.0).sqrt()
        sar_gradient = (
            F.pad((sar[:, :, :, 1:] - sar[:, :, :, :-1]).abs(), (0, 1, 0, 0))
            + F.pad((sar[:, :, 1:, :] - sar[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        )
        sar_global_mean = sar.mean((2, 3), keepdim=True).expand(-1, -1, *size)
        sar_global_std = sar.std(
            (2, 3), keepdim=True, unbiased=False
        ).expand(-1, -1, *size)

        features = torch.cat(
            [
                p3_probability,
                a2_probability,
                (p3_probability - a2_probability).abs(),
                self._entropy(p3_probability),
                self._entropy(a2_probability),
                self._margin(p3_probability),
                self._margin(a2_probability),
                pair_feature,
                building,
                boundary,
                p3_consistency3,
                p3_consistency7,
                a2_consistency3,
                a2_consistency7,
                sar_mean3,
                sar_std3,
                sar_gradient,
                sar_global_mean,
                sar_global_std,
            ],
            dim=1,
        )
        return features, {
            "cer_v2_pair_id": pair_id,
            "cer_v2_building_probability": building,
            "cer_v2_boundary_strength": boundary,
            "cer_v2_a2_consistency3": a2_consistency3,
            "cer_v2_a2_consistency7": a2_consistency7,
        }

    def forward(
        self,
        p3_logits: torch.Tensor,
        native_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if native_logits.shape[-2:] != p3_logits.shape[-2:]:
            native_logits = F.interpolate(
                native_logits,
                size=p3_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p3_probability = p3_logits.float().softmax(1).to(p3_logits.dtype)
        a2_probability = native_logits.float().softmax(1).to(native_logits.dtype)
        p3_prediction = p3_probability.argmax(1, keepdim=True)
        a2_prediction = a2_probability.argmax(1, keepdim=True)
        features, diagnostics = self._context_features(
            p3_probability, a2_probability, p3_prediction, a2_prediction
        )
        selector_logits = self.selector(features)
        selector_probability = selector_logits.float().softmax(1).to(
            selector_logits.dtype
        )
        pair_id = diagnostics["cer_v2_pair_id"]
        pair_allowed = torch.zeros_like(pair_id, dtype=torch.bool)
        for allowed in V2A_ALLOWED_PAIR_IDS:
            pair_allowed |= pair_id == allowed
        spatial_permission = (
            diagnostics["cer_v2_building_probability"] < self.building_threshold
        ) | (diagnostics["cer_v2_boundary_strength"] > self.boundary_threshold)
        candidate = pair_allowed & spatial_permission
        advantage = selector_probability[:, 1:2] - selector_probability[:, :1]
        reject_probability = selector_probability[:, 2:3]
        local_consistency = diagnostics["cer_v2_a2_consistency7"]
        proposed = (
            candidate
            & (advantage > self.advantage_threshold)
            & (reject_probability < self.reject_threshold)
            & (local_consistency > self.consistency_threshold)
        )
        disagreement = p3_prediction != a2_prediction
        route_open = self._budget_mask(
            advantage, proposed, disagreement, self.route_budget
        )
        output = torch.where(route_open.expand_as(p3_logits), native_logits, p3_logits)

        diagnostics.update(
            {
                "cer_v2_selector_logits": selector_logits,
                "cer_v2_selector_probability": selector_probability,
                "cer_v2_pair_allowed": pair_allowed.to(p3_logits.dtype),
                "cer_v2_spatial_permission": spatial_permission.to(p3_logits.dtype),
                "cer_v2_candidate_mask": candidate.to(p3_logits.dtype),
                "cer_v2_route_open_mask": route_open.to(p3_logits.dtype),
                "cer_v2_advantage": advantage,
                "cer_v2_reject_probability": reject_probability,
                "cer_v2_local_consistency": local_consistency,
                "cer_v2_p3_prediction": p3_prediction.detach(),
                "cer_v2_a2_prediction": a2_prediction.detach(),
                "safe_residual_sar_gate": route_open.to(p3_logits.dtype),
                "safe_class_transition": p3_logits.new_zeros((1, 4, 1, 1)),
                "safe_residual_delta_abs": (
                    output - p3_logits
                ).abs().mean(1, keepdim=True),
                "native_p3_delta_abs": (
                    native_logits - p3_logits
                ).abs().mean(1, keepdim=True),
            }
        )
        self._native_sar_context = None
        self._localization_context = None
        return output, diagnostics


class DamageFormerRGBSARH7CERV2(DamageFormerRGBSARH6SDCR):
    def __init__(self, pretrained_rgb: bool = True) -> None:
        super().__init__(variant="b1r", pretrained_rgb=pretrained_rgb)
        self.variant = "cer_v2a"
        self.safe_residual_router = PairConditionalRejectableSelector()


def build_h7_cer_v2_damageformer(
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH7CERV2:
    return DamageFormerRGBSARH7CERV2(pretrained_rgb=pretrained_rgb)


__all__ = [
    "KEEP_P3",
    "USE_A2",
    "REJECT_BOTH",
    "V2A_ALLOWED_PAIR_IDS",
    "PairConditionalRejectableSelector",
    "DamageFormerRGBSARH7CERV2",
    "build_h7_cer_v2_damageformer",
]
