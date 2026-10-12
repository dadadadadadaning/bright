"""Q2 building-region residual fusion with an exact frozen-P3 safe state.

Q2 never chooses between experts.  A frozen P3 produces the official dense
prediction and building proposals, while a frozen Q0 readout supplies native
SAR damage probabilities.  One small region MLP predicts a three-building-
class logit residual which is broadcast back only inside each P3 component.
The scalar transition starts at exactly zero, so the initial dense output is
bitwise the P3 output while the transition still receives a first-step
gradient from the non-zero, randomly initialised proposal head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .DamageFormer_RGBSAR_Q0_Readout import Q0BuildingConditionedSARReadout
from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL

try:  # OpenCV is the production path on the training server.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


Q2_VARIANTS = ("q2_b0", "q2_b1")
Q2_FEATURE_DIMENSION = 18


@dataclass(frozen=True)
class RegionRecord:
    batch_index: int
    instance_id: int
    area: int


def _components_numpy(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        count, labels = cv2.connectedComponents(mask, connectivity=4)
    else:  # Small dependency-free fallback used by unit tests.
        from .region_ops import _connected_components_4

        labels = _connected_components_4(mask)
        count = int(labels.max()) + 1
    output = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for old_id in range(1, int(count)):
        selected = labels == old_id
        if int(selected.sum()) < minimum_area:
            continue
        output[selected] = next_id
        next_id += 1
    return output


def build_p3_instance_map(
    localization_logits: torch.Tensor,
    threshold: float = 0.5,
    minimum_area: int = 16,
) -> torch.Tensor:
    """Create inference proposals from frozen P3 only; no GT is accepted."""
    if localization_logits.ndim != 4 or localization_logits.shape[1] != 2:
        raise ValueError("P3 localization logits must have shape [B,2,H,W]")
    probability = localization_logits.detach().float().softmax(1)[:, 1]
    binary = (probability >= float(threshold)).cpu().numpy()
    maps = [_components_numpy(value, minimum_area) for value in binary]
    return torch.from_numpy(np.stack(maps)).to(
        localization_logits.device, dtype=torch.long
    )


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability.clamp_min(1e-7).log() * probability).sum()


def _margin(probability: torch.Tensor) -> torch.Tensor:
    top = probability.topk(2).values
    return top[0] - top[1]


class Q2BuildingResidualFusion(nn.Module):
    def __init__(
        self,
        p3: DamageFormer_S8_MICB_DSMIL,
        q0: Q0BuildingConditionedSARReadout,
        variant: str,
        *,
        proposal_threshold: float = 0.5,
        minimum_area: int = 16,
        maximum_transition: float = 0.25,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in Q2_VARIANTS:
            raise ValueError(f"Q2 variant must be one of {Q2_VARIANTS}")
        if maximum_transition <= 0:
            raise ValueError("maximum_transition must be positive")
        self.variant = variant
        self.p3 = p3
        self.q0 = q0
        self.proposal_threshold = float(proposal_threshold)
        self.minimum_area = int(minimum_area)
        self.maximum_transition = float(maximum_transition)
        self.region_head = nn.Sequential(
            nn.Linear(Q2_FEATURE_DIMENSION, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        # Keep the proposal non-zero so alpha gets a gradient at the exact
        # zero-transition state.  Zeroing both would dead-lock the module.
        nn.init.normal_(self.region_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.region_head[-1].bias)
        self.transition_raw = nn.Parameter(torch.zeros(()))
        self.freeze_experts()

    def freeze_experts(self) -> None:
        self.p3.eval()
        self.q0.eval()
        for module in (self.p3, self.q0):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.p3.eval()
        self.q0.eval()
        return self

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, value in self.named_parameters() if value.requires_grad]

    @staticmethod
    def p3_sar_view(post_sar: torch.Tensor) -> torch.Tensor:
        if post_sar.ndim != 4 or post_sar.shape[1] != 2:
            raise RuntimeError("Q2 requires native and display SAR channels")
        display = post_sar[:, 1:2].float()
        mean = display.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        std = display.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        return ((display.repeat(1, 3, 1, 1) - mean) / std).to(post_sar.dtype)

    def _region_features(
        self,
        p3_damage_logits: torch.Tensor,
        q0_logits: torch.Tensor,
        localization_logits: torch.Tensor,
        instance_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Sequence[RegionRecord]]:
        p3_full_probability = p3_damage_logits.detach().float().softmax(1)
        p3_probability = p3_full_probability[:, 1:4]
        p3_probability = p3_probability / p3_probability.sum(1, keepdim=True).clamp_min(1e-7)
        q0_probability = q0_logits.detach().float().softmax(1)
        loc_probability = localization_logits.detach().float().softmax(1)[:, 1]
        height, width = instance_map.shape[-2:]
        image_area = float(height * width)
        features: List[torch.Tensor] = []
        base_logits: List[torch.Tensor] = []
        records: List[RegionRecord] = []
        for batch_index in range(instance_map.shape[0]):
            for instance_id in range(1, int(instance_map[batch_index].max()) + 1):
                mask = instance_map[batch_index] == instance_id
                area = int(mask.sum())
                if area <= 0:
                    continue
                p3_mean = p3_probability[batch_index, :, mask].mean(1)
                q0_mean = q0_probability[batch_index, :, mask].mean(1)
                ys, xs = torch.where(mask)
                box_height = (ys.max() - ys.min() + 1).float()
                box_width = (xs.max() - xs.min() + 1).float()
                geometry = p3_mean.new_tensor([
                    np.log1p(area) / np.log1p(image_area),
                    float((box_width / box_height.clamp_min(1)).item()),
                    float(loc_probability[batch_index, mask].mean().item()),
                ])
                feature = torch.cat([
                    p3_mean,
                    q0_mean,
                    _entropy(p3_mean).view(1),
                    _entropy(q0_mean).view(1),
                    _margin(p3_mean).view(1),
                    _margin(q0_mean).view(1),
                    (q0_mean - p3_mean).abs(),
                    (q0_mean - p3_mean)[1:3],
                    geometry,
                ])
                if feature.numel() != Q2_FEATURE_DIMENSION:
                    raise RuntimeError("Q2 region feature contract changed")
                features.append(feature)
                # Keep P3 background in the supervised region decision.  Q2
                # cannot alter its background logit, but its three building
                # residuals must learn to compete with it; a conditional-only
                # 3-class CE would be blind to Damaged -> Background errors.
                base_logits.append(p3_damage_logits[batch_index, :, mask].mean(1))
                records.append(RegionRecord(batch_index, instance_id, area))
        if not features:
            empty = p3_damage_logits.new_empty((0, Q2_FEATURE_DIMENSION))
            return empty, p3_damage_logits.new_empty((0, 4)), records
        return torch.stack(features), torch.stack(base_logits), records

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        with torch.no_grad():
            localization_logits, p3_damage_logits = self.p3(
                pre_rgb, self.p3_sar_view(post_sar)
            )
            stage4 = self.q0.source.sar_encoder(post_sar[:, :1])[-1]
            q0_low = self.q0.readout(stage4)
            q0_logits = torch.nn.functional.interpolate(
                q0_low,
                size=p3_damage_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        instance_map = build_p3_instance_map(
            localization_logits,
            threshold=self.proposal_threshold,
            minimum_area=self.minimum_area,
        )
        features, base_region_logits, records = self._region_features(
            p3_damage_logits, q0_logits, localization_logits, instance_map
        )
        region_delta = self.region_head(features)
        transition = self.maximum_transition * torch.tanh(self.transition_raw)
        padded_region_delta = torch.cat(
            [region_delta.new_zeros((region_delta.shape[0], 1)), region_delta], dim=1
        )
        region_logits = base_region_logits + transition * padded_region_delta
        output_damage = p3_damage_logits.clone()
        for index, record in enumerate(records):
            mask = instance_map[record.batch_index] == record.instance_id
            output_damage[record.batch_index, 1:4, mask] = (
                output_damage[record.batch_index, 1:4, mask]
                + transition * region_delta[index].view(3, 1)
            )
        outputs: Dict[str, object] = {
            "output_loc": localization_logits,
            "output_dam": output_damage,
            "p3_output_dam": p3_damage_logits,
            "q0_output_dam": q0_logits,
            "q2_instance_map": instance_map,
            "q2_region_features": features,
            "q2_base_region_logits": base_region_logits,
            "q2_region_logits": region_logits,
            "q2_region_delta": region_delta,
            "q2_records": records,
            "q2_transition": transition.view(1),
            "q2_query_count": output_damage.new_tensor(float(len(records))),
            "q2_delta_abs": (
                (transition * region_delta).abs().mean()
                if region_delta.numel() else output_damage.new_zeros(())
            ),
        }
        if return_dict:
            return outputs
        return localization_logits, output_damage


def assert_q2_freeze_contract(model: Q2BuildingResidualFusion) -> None:
    allowed = ("transition_raw", "region_head.")
    trainable = model.trainable_parameter_names()
    unexpected = [
        name for name in trainable
        if name != allowed[0] and not name.startswith(allowed[1])
    ]
    if not trainable or unexpected:
        raise RuntimeError(f"Q2 trainable-parameter contract violated: {trainable}")
    leaked = [
        name for prefix, module in (("p3", model.p3), ("q0", model.q0))
        for name, value in module.named_parameters() if value.requires_grad
    ]
    if leaked:
        raise RuntimeError(f"Q2 frozen expert leak: {leaked[:8]}")


__all__ = [
    "Q2_VARIANTS", "Q2_FEATURE_DIMENSION", "RegionRecord",
    "Q2BuildingResidualFusion", "assert_q2_freeze_contract",
    "build_p3_instance_map",
]
