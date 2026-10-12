# -*- coding: utf-8 -*-
"""H5 building-instance-query upgrades for heterogeneous RGB/SAR DamageFormer.

H5 changes the prediction unit without adding a new supervised objective:

* Q0 is an exact H4-A4 checkpoint view.
* Q1 pools multi-scale RGB/SAR evidence once per predicted building instance
  and broadcasts one three-way severity residual back to that instance.
* Q2 additionally lets each instance query select a few SAR sampling offsets
  at stages 3 and 4.  The resulting residual is restricted to building
  instances and cannot alter the frozen RGB localization branch.

All added trainable output projections are zero initialized.  Consequently
Q1/Q2 ``output_dam`` remain exact H4-A4 functions immediately after checkpoint
migration, while ``instance_query_output_dam`` exposes the strict
one-label-per-building decoder used for H5 validation and inference.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H4 import (
    H4_VARIANTS,
    DamageFormerRGBSARH4,
    ReliabilityGatedCommonUniqueStage,
    hierarchical_probabilities,
)
from .DamageFormer_RGBSAR_Heterogeneous import STAGE_CHANNELS, ConvGNAct


VALID_H5_VARIANTS = ("q0", "q1", "q2")


@dataclass(frozen=True)
class H5Variant:
    name: str
    instance_query: bool
    deep_sparse_deformable: bool
    extra_supervised_losses: bool = False


H5_VARIANTS: Mapping[str, H5Variant] = {
    "q0": H5Variant("H5-Q0-H4A4-Control", False, False),
    "q1": H5Variant("H5-Q1-Building-Instance-Query", True, False),
    "q2": H5Variant(
        "H5-Q2-BIQ-Deep-Sparse-Deformable-Fusion",
        True,
        True,
    ),
}


def _connected_components_4(mask: np.ndarray) -> List[np.ndarray]:
    """Dependency-free four-connected components for a small feature map."""

    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    height, width = mask.shape
    visited = np.zeros_like(mask)
    components: List[np.ndarray] = []
    for y0, x0 in np.argwhere(mask):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        queue = deque([(int(y0), int(x0))])
        points: List[Tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for yy, xx in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= yy < height
                    and 0 <= xx < width
                    and mask[yy, xx]
                    and not visited[yy, xx]
                ):
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        component = np.zeros_like(mask)
        yy, xx = zip(*points)
        component[np.asarray(yy), np.asarray(xx)] = True
        components.append(component)
    return components


@torch.no_grad()
def connected_building_instances(
    building: torch.Tensor,
    *,
    threshold: float = 0.5,
    min_area: int = 4,
    max_instances: int = 256,
) -> torch.Tensor:
    """Convert a frozen building probability/mask to contiguous instance ids.

    Four-connected components are ordered by decreasing area, with the source
    component id as a deterministic tie break.  The operation is intentionally
    detached: RGB localization is frozen and is not optimized through instance
    extraction.
    """

    if building.ndim == 4 and building.shape[1] == 1:
        building = building[:, 0]
    if building.ndim != 3:
        raise ValueError(
            f"building tensor must be [B,H,W] or [B,1,H,W], got {building.shape}"
        )
    if min_area < 1 or max_instances < 1:
        raise ValueError("min_area and max_instances must be positive")
    binary = (
        building.detach().float().to(device="cpu").numpy() >= float(threshold)
    )
    result = np.zeros(binary.shape, dtype=np.int32)
    for batch_index, sample in enumerate(binary):
        candidates = [
            (source_id, component, int(component.sum()))
            for source_id, component in enumerate(
                _connected_components_4(sample),
                start=1,
            )
            if int(component.sum()) >= min_area
        ]
        candidates.sort(key=lambda item: (-item[2], item[0]))
        for new_id, (_, component, _) in enumerate(
            candidates[:max_instances],
            start=1,
        ):
            result[batch_index][component] = new_id
    return torch.from_numpy(result).to(device=building.device, dtype=torch.long)


def resize_instance_map(
    instance_map: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    if instance_map.ndim != 3:
        raise ValueError("instance_map must be [B,H,W]")
    if tuple(instance_map.shape[-2:]) == tuple(size):
        return instance_map
    return F.interpolate(
        instance_map.unsqueeze(1).float(),
        size=size,
        mode="nearest",
    )[:, 0].long()


@torch.no_grad()
def compact_instance_map(
    instance_map: torch.Tensor,
    *,
    max_instances: int = 256,
) -> torch.Tensor:
    """Reindex arbitrary external instance ids to contiguous area-ranked ids."""

    if instance_map.ndim != 3:
        raise ValueError("instance_map must be [B,H,W]")
    source = instance_map.detach().to(device="cpu", dtype=torch.long).numpy()
    result = np.zeros(source.shape, dtype=np.int32)
    for batch_index, sample in enumerate(source):
        ids, counts = np.unique(sample[sample > 0], return_counts=True)
        ranked = sorted(
            zip(ids.tolist(), counts.tolist()),
            key=lambda item: (-int(item[1]), int(item[0])),
        )
        for new_id, (old_id, _) in enumerate(
            ranked[:max_instances],
            start=1,
        ):
            result[batch_index][sample == int(old_id)] = new_id
    return torch.from_numpy(result).to(
        device=instance_map.device,
        dtype=torch.long,
    )


def _masked_mean_table(
    feature: torch.Tensor,
    instance_map: torch.Tensor,
    instance_count: int,
) -> torch.Tensor:
    """Return rows 0..N of differentiable per-instance masked means."""

    if feature.ndim != 4 or instance_map.ndim != 3:
        raise ValueError("feature/instance map must be NCHW/BHW")
    if feature.shape[0] != 1 or instance_map.shape[0] != 1:
        raise ValueError("_masked_mean_table operates on one sample")
    labels = resize_instance_map(instance_map, feature.shape[-2:])[0].reshape(-1)
    values = feature[0].permute(1, 2, 0).reshape(-1, feature.shape[1])
    table = values.new_zeros((instance_count + 1, feature.shape[1]))
    table.index_add_(0, labels, values)
    counts = values.new_zeros((instance_count + 1, 1))
    counts.index_add_(0, labels, values.new_ones((values.shape[0], 1)))
    return table / counts.clamp_min(1.0)


class BuildingInstanceQueryHead(nn.Module):
    """One severity query per frozen RGB building component."""

    def __init__(
        self,
        query_dim: int = 256,
        projection_dim: int = 64,
        max_instances: int = 256,
    ) -> None:
        super().__init__()
        self.max_instances = int(max_instances)
        self.damage_projection = ConvGNAct(256, 2 * projection_dim, 1)
        self.rgb_projections = nn.ModuleList(
            [
                ConvGNAct(STAGE_CHANNELS[2], projection_dim, 1),
                ConvGNAct(STAGE_CHANNELS[3], projection_dim, 1),
            ]
        )
        self.sar_projections = nn.ModuleList(
            [
                ConvGNAct(STAGE_CHANNELS[2], projection_dim, 1),
                ConvGNAct(STAGE_CHANNELS[3], projection_dim, 1),
            ]
        )
        descriptor_dim = 2 * projection_dim + 4 * projection_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
        )
        self.self_attention = nn.MultiheadAttention(
            query_dim,
            num_heads=8,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(query_dim)
        self.severity_delta = nn.Linear(query_dim, 3)
        nn.init.zeros_(self.severity_delta.weight)
        nn.init.zeros_(self.severity_delta.bias)

    def forward(
        self,
        damage_feature: torch.Tensor,
        base_damage_logits: torch.Tensor,
        rgb_stages: Sequence[torch.Tensor],
        sar_stages: Sequence[torch.Tensor],
        instance_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if len(rgb_stages) != 4 or len(sar_stages) != 4:
            raise ValueError("BIQ expects four RGB and four SAR stages")
        target_size = damage_feature.shape[-2:]
        instances = resize_instance_map(instance_map, target_size)
        damage = self.damage_projection(damage_feature)
        rgb = [
            projection(feature)
            for projection, feature in zip(self.rgb_projections, rgb_stages[2:])
        ]
        sar = [
            projection(feature)
            for projection, feature in zip(self.sar_projections, sar_stages[2:])
        ]
        dense_delta = damage_feature.new_zeros(
            (damage_feature.shape[0], 4, *target_size)
        )
        instance_logits = base_damage_logits.clone()
        query_counts: List[int] = []
        query_delta_abs: List[torch.Tensor] = []
        for batch_index in range(damage_feature.shape[0]):
            sample_instances = instances[batch_index : batch_index + 1]
            instance_count = min(
                int(sample_instances.max().item()),
                self.max_instances,
            )
            query_counts.append(instance_count)
            if instance_count == 0:
                query_delta_abs.append(damage_feature.new_zeros(()))
                continue
            descriptors = [
                _masked_mean_table(
                    damage[batch_index : batch_index + 1],
                    sample_instances,
                    instance_count,
                )[1:]
            ]
            for feature in (*rgb, *sar):
                descriptors.append(
                    _masked_mean_table(
                        feature[batch_index : batch_index + 1],
                        sample_instances,
                        instance_count,
                    )[1:]
                )
            queries = self.query_encoder(torch.cat(descriptors, dim=1))
            attended, _ = self.self_attention(
                queries.unsqueeze(0),
                queries.unsqueeze(0),
                queries.unsqueeze(0),
                need_weights=False,
            )
            queries = self.query_norm(queries + attended[0])
            deltas = self.severity_delta(queries)
            pooled_base = _masked_mean_table(
                base_damage_logits[batch_index : batch_index + 1],
                sample_instances,
                instance_count,
            )[1:, 1:4]
            severity_logits = pooled_base + deltas
            table = deltas.new_zeros((instance_count + 1, 3))
            table[1:] = deltas
            # hierarchy_probabilities intentionally promotes the base damage
            # path to FP32 for numerical stability, while the query MLP runs
            # in BF16 under autocast.  Advanced index assignment does not
            # perform implicit dtype conversion, so construct the strict
            # instance table in the destination logits dtype.
            strict_table = base_damage_logits.new_zeros(
                (instance_count + 1, 4)
            )
            strict_table[1:, 0] = -20.0
            strict_table[1:, 1:4] = severity_logits
            labels = sample_instances[0].clamp_max(instance_count)
            dense_delta[batch_index, 1:4] = table[labels].permute(2, 0, 1)
            building = labels > 0
            instance_logits[batch_index, :, building] = strict_table[
                labels[building]
            ].transpose(0, 1)
            query_delta_abs.append(deltas.abs().mean())
        count_tensor = damage_feature.new_tensor(query_counts, dtype=torch.float32)
        delta_tensor = torch.stack(query_delta_abs)
        return dense_delta, instance_logits, {
            "instance_query_count": count_tensor,
            "instance_query_delta_abs": delta_tensor,
        }


class InstanceSparseDeformableFusion(nn.Module):
    """Instance-conditioned K-point SAR sampling for one deep feature stage."""

    def __init__(
        self,
        source_channels: int,
        latent_channels: int,
        *,
        points: int = 4,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__()
        if points < 1:
            raise ValueError("points must be positive")
        self.points = int(points)
        self.max_offset = float(max_offset)
        self.rgb_projection = ConvGNAct(source_channels, latent_channels, 1)
        self.sar_projection = ConvGNAct(source_channels, latent_channels, 1)
        hidden = max(64, latent_channels)
        self.instance_encoder = nn.Sequential(
            nn.Linear(3 * latent_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.offset_head = nn.Linear(hidden, 2 * self.points)
        self.weight_head = nn.Linear(hidden, self.points)
        self.output_projection = nn.Conv2d(
            4 * latent_channels,
            latent_channels,
            1,
            bias=False,
        )
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        nn.init.zeros_(self.output_projection.weight)

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
        self,
        joint: torch.Tensor,
        rgb_source: torch.Tensor,
        sar_source: torch.Tensor,
        instance_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb = self.rgb_projection(rgb_source)
        sar = self.sar_projection(sar_source)
        instances = resize_instance_map(instance_map, joint.shape[-2:])
        batch, _, height, width = joint.shape
        offset_map = joint.new_zeros((batch, height, width, self.points, 2))
        weight_map = joint.new_zeros((batch, height, width, self.points))
        building_mask = (instances > 0).unsqueeze(1).to(joint.dtype)
        for batch_index in range(batch):
            sample_instances = instances[batch_index : batch_index + 1]
            instance_count = int(sample_instances.max().item())
            if instance_count == 0:
                continue
            rgb_table = _masked_mean_table(
                rgb[batch_index : batch_index + 1],
                sample_instances,
                instance_count,
            )
            sar_table = _masked_mean_table(
                sar[batch_index : batch_index + 1],
                sample_instances,
                instance_count,
            )
            descriptor = torch.cat(
                [rgb_table, sar_table, (rgb_table - sar_table).abs()],
                dim=1,
            )
            encoded = self.instance_encoder(descriptor[1:])
            offsets = (
                torch.tanh(self.offset_head(encoded))
                .reshape(instance_count, self.points, 2)
                * self.max_offset
            )
            weights = self.weight_head(encoded).softmax(dim=1)
            offset_table = offsets.new_zeros((instance_count + 1, self.points, 2))
            weight_table = weights.new_zeros((instance_count + 1, self.points))
            offset_table[1:] = offsets
            weight_table[1:] = weights
            labels = sample_instances[0]
            offset_map[batch_index] = offset_table[labels]
            weight_map[batch_index] = weight_table[labels]

        base_grid = self._base_grid(
            batch,
            height,
            width,
            joint.device,
            joint.dtype,
        )
        sampled = torch.zeros_like(sar)
        for point in range(self.points):
            offset = offset_map[:, :, :, point]
            normalized = torch.empty_like(offset)
            normalized[..., 0] = offset[..., 0] * (2.0 / max(width - 1, 1))
            normalized[..., 1] = offset[..., 1] * (2.0 / max(height - 1, 1))
            value = F.grid_sample(
                sar,
                base_grid + normalized,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled = sampled + weight_map[..., point].unsqueeze(1) * value
        descriptor = torch.cat(
            [joint, rgb, sampled, (rgb - sampled).abs()],
            dim=1,
        )
        residual = self.output_projection(descriptor) * building_mask
        offset_magnitude = offset_map.float().square().sum(-1).sqrt()
        valid = instances > 0
        if valid.any():
            offset_mean = offset_magnitude[valid].mean()
            residual_mean = residual.float().abs()[
                building_mask.expand_as(residual) > 0
            ].mean()
        else:
            offset_mean = joint.float().sum() * 0.0
            residual_mean = joint.float().sum() * 0.0
        return joint + residual, {
            "instance_offset_mean": offset_mean.reshape(1),
            "instance_residual_abs": residual_mean.reshape(1),
        }


class DamageFormerRGBSARH5BIQ(DamageFormerRGBSARH4):
    """H5-A/H5-B model with exact H4-A4 initialization."""

    def __init__(
        self,
        variant: str = "q1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        instance_threshold: float = 0.5,
        min_instance_area: int = 4,
        max_instances: int = 256,
    ) -> None:
        variant = variant.lower()
        if variant not in VALID_H5_VARIANTS:
            raise ValueError(f"unknown H5 variant {variant!r}")
        super().__init__(
            variant="a4",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h5_config = H5_VARIANTS[variant]
        self.instance_threshold = float(instance_threshold)
        self.min_instance_area = int(min_instance_area)
        self.max_instances = int(max_instances)
        if self.h5_config.instance_query:
            self.instance_query_head = BuildingInstanceQueryHead(
                max_instances=max_instances
            )
        if self.h5_config.deep_sparse_deformable:
            self.instance_sparse_fusions = nn.ModuleDict(
                {
                    "stage3": InstanceSparseDeformableFusion(
                        STAGE_CHANNELS[2],
                        latent_channels[2],
                    ),
                    "stage4": InstanceSparseDeformableFusion(
                        STAGE_CHANNELS[3],
                        latent_channels[3],
                    ),
                }
            )

    def _resolve_instances(
        self,
        localization_probability: torch.Tensor,
        instance_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if instance_map is None:
            return connected_building_instances(
                localization_probability,
                threshold=self.instance_threshold,
                min_area=self.min_instance_area,
                max_instances=self.max_instances,
            )
        if instance_map.ndim == 4 and instance_map.shape[1] == 1:
            instance_map = instance_map[:, 0]
        if instance_map.ndim != 3:
            raise ValueError("external instance_map must be [B,H,W]")
        resized = resize_instance_map(
            instance_map.detach().long().to(localization_probability.device),
            localization_probability.shape[-2:],
        )
        return compact_instance_map(
            resized,
            max_instances=self.max_instances,
        )

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
        instance_map: torch.Tensor | None = None,
    ):
        if self.variant == "q0":
            return super().forward(pre_rgb, post_sar, return_dict=return_dict)

        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)

        localization_feature = self.localization_decoder(rgb_stages)
        localization_logits_low = self.localization_head(localization_feature)
        localization_probability = localization_logits_low.softmax(1)[:, 1:2]
        instances = self._resolve_instances(
            localization_probability,
            instance_map,
        )
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
            sparse_key = f"stage{index}"
            if (
                self.h5_config.deep_sparse_deformable
                and sparse_key in self.instance_sparse_fusions
            ):
                joint, sparse_diagnostics = self.instance_sparse_fusions[
                    sparse_key
                ](joint, rgb, sar, instances)
                stage_diagnostics.update(sparse_diagnostics)
            joint_stages.append(joint)
            for key, value in stage_diagnostics.items():
                diagnostics[f"stage{index}_{key}"] = value

        damage_feature = self.damage_decoder(joint_stages)
        damage_feature = damage_feature * (
            0.5 + 0.5 * localization_probability.detach()
        )
        base_damage_logits_low = self.damage_head(damage_feature)

        building_low = self.hierarchy_building_head(damage_feature)
        affected_low = self.hierarchy_affected_head(damage_feature)
        destroyed_low = self.hierarchy_destroyed_head(damage_feature)
        hierarchy_probability = hierarchical_probabilities(
            building_low,
            affected_low,
            destroyed_low,
        )
        hierarchy_logits_low = hierarchy_probability.clamp_min(1e-7).log()
        hierarchy_transition = torch.tanh(self.hierarchy_transition)
        damage_logits_low = base_damage_logits_low + hierarchy_transition * (
            hierarchy_logits_low - base_damage_logits_low
        )

        query_delta, instance_damage_logits_low, query_diagnostics = (
            self.instance_query_head(
            damage_feature,
            damage_logits_low,
            rgb_stages,
            sar_stages,
            instances,
            )
        )
        damage_logits_low = damage_logits_low + query_delta
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
            "instance_query_output_dam": F.interpolate(
                instance_damage_logits_low,
                size=input_size,
                mode="nearest",
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
            "instance_map_low": instances,
            **query_diagnostics,
            **diagnostics,
        }
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_h5_biq_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH5BIQ:
    return DamageFormerRGBSARH5BIQ(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H5_VARIANTS",
    "H5_VARIANTS",
    "connected_building_instances",
    "compact_instance_map",
    "resize_instance_map",
    "BuildingInstanceQueryHead",
    "InstanceSparseDeformableFusion",
    "DamageFormerRGBSARH5BIQ",
    "build_h5_biq_damageformer",
]
