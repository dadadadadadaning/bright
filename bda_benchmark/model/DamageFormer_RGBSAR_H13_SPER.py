# -*- coding: utf-8 -*-
"""H13-SPER: superpixel evidence reasoning for weak RGB--SAR damage.

The model is a controlled descendant of the locked H6-A2 expert.  Superpixels
cover the complete scene and are used only as *evidence cells*; they are not
building instances and they never impose a hard final damage label.  A frozen
RGB localisation mask limits the final residual to buildings, while regional
pooling itself may use surrounding context.

Variants form a causal chain:

``s0``  exact H6-A2 control;
``s1``  regular grid-region pooling (parameter/control baseline);
``s2``  RGB-guided soft superpixels inspired by SSN/SpixelFCN;
``s3``  ``s2`` plus one boundary-gated four-neighbour graph step.

All new heads are zero-initialised, so every trainable variant is exactly
H6-A2 before optimisation.  The residual edits only Intact/Affected odds and
preserves the Damaged/Destroyed severity odds exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H10_H3D import (
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from .DamageFormer_RGBSAR_H12_DGB import (
    OCCURRENCE_CANDIDATE_BOUND,
    ConvGNAct,
    bound_occurrence_candidate,
    bounded_occurrence_transition,
    erode_building_support,
)
from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR


@dataclass(frozen=True)
class H13Variant:
    name: str
    assignment_mode: str
    graph_refinement: bool


H13_VARIANTS: Mapping[str, H13Variant] = {
    "s0": H13Variant("H13-S0-Locked-H6A2", "none", False),
    "s1": H13Variant("H13-S1-Regular-Region-Control", "grid", False),
    "s2": H13Variant("H13-S2-RGB-Guided-Soft-Superpixels", "rgb", False),
    "s3": H13Variant("H13-S3-Superpixel-Adjacency-Reasoning", "rgb", True),
}
VALID_H13_VARIANTS = tuple(H13_VARIANTS)


@dataclass
class SuperpixelAssignment:
    """Sparse pixel-to-region association over at most nine grid seeds."""

    weights: torch.Tensor  # B,K,P
    indices: torch.Tensor  # K,P, shared by the batch
    grid_height: int
    grid_width: int
    entropy: torch.Tensor  # B,1,H,W

    @property
    def node_count(self) -> int:
        return int(self.grid_height * self.grid_width)


class RGBGuidedSoftSuperpixels(nn.Module):
    """Dependency-free SSN-style local soft superpixel assignment.

    Grid seeds guarantee compactness.  In ``rgb`` mode, each pixel chooses
    softly among the surrounding 3x3 seeds using detached pre-event RGB and
    spatial distance.  The same assignment is then used for RGB and SAR
    evidence, avoiding two incompatible modality-specific partitions.
    """

    def __init__(
        self,
        cell_size: int = 2,
        color_temperature: float = 0.50,
        compactness: float = 0.75,
    ) -> None:
        super().__init__()
        if cell_size < 1:
            raise ValueError("superpixel cell size must be positive")
        if color_temperature <= 0.0:
            raise ValueError("superpixel color temperature must be positive")
        if compactness < 0.0:
            raise ValueError("superpixel compactness must be non-negative")
        self.cell_size = int(cell_size)
        self.color_temperature = float(color_temperature)
        self.compactness = float(compactness)

    @staticmethod
    def _normalise_rgb(rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.detach().float()
        mean = rgb.mean((2, 3), keepdim=True)
        std = rgb.std((2, 3), keepdim=True, unbiased=False).clamp_min(1.0e-4)
        return ((rgb - mean) / std).clamp(-5.0, 5.0)

    def _candidate_geometry(
        self, height: int, width: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        cell = self.cell_size
        grid_h = (height + cell - 1) // cell
        grid_w = (width + cell - 1) // cell
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        base_y = yy // cell
        base_x = xx // cell
        ids = []
        valid = []
        spatial = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                gy = base_y + dy
                gx = base_x + dx
                ok = (gy >= 0) & (gy < grid_h) & (gx >= 0) & (gx < grid_w)
                safe_y = gy.clamp(0, grid_h - 1)
                safe_x = gx.clamp(0, grid_w - 1)
                ids.append((safe_y * grid_w + safe_x).reshape(-1))
                valid.append(ok.reshape(-1))
                centre_y = (safe_y.float() + 0.5) * cell - 0.5
                centre_x = (safe_x.float() + 0.5) * cell - 0.5
                distance = (
                    ((yy.float() - centre_y) / cell).square()
                    + ((xx.float() - centre_x) / cell).square()
                )
                spatial.append(distance.reshape(-1))
        return (
            torch.stack(ids),
            torch.stack(valid),
            torch.stack(spatial),
            grid_h,
            grid_w,
        )

    def forward(self, rgb: torch.Tensor, mode: str) -> SuperpixelAssignment:
        if mode not in ("grid", "rgb"):
            raise ValueError(f"unknown superpixel assignment mode: {mode!r}")
        batch, channels, height, width = rgb.shape
        ids, valid, spatial, grid_h, grid_w = self._candidate_geometry(
            height, width, rgb.device
        )
        pixels = height * width
        if mode == "grid":
            centre_slot = 4
            weights = rgb.new_zeros((batch, 9, pixels), dtype=torch.float32)
            weights[:, centre_slot] = 1.0
        else:
            normalised = self._normalise_rgb(rgb)
            pad_h = grid_h * self.cell_size - height
            pad_w = grid_w * self.cell_size - width
            padded = F.pad(normalised, (0, pad_w, 0, pad_h), mode="replicate")
            centres = F.avg_pool2d(
                padded,
                kernel_size=self.cell_size,
                stride=self.cell_size,
            ).flatten(2)
            gather_index = ids.reshape(1, 1, -1).expand(batch, channels, -1)
            candidate_colour = torch.gather(centres, 2, gather_index)
            candidate_colour = candidate_colour.reshape(batch, channels, 9, pixels)
            candidate_colour = candidate_colour.permute(0, 2, 1, 3)
            pixel_colour = normalised.flatten(2).unsqueeze(1)
            colour_distance = (pixel_colour - candidate_colour).square().mean(2)
            logits = -colour_distance / self.color_temperature
            logits = logits - self.compactness * spatial.unsqueeze(0)
            logits = logits.masked_fill(~valid.unsqueeze(0), -1.0e4)
            weights = logits.softmax(1)
        entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(1)
        entropy = entropy / math.log(9.0)
        entropy = entropy.reshape(batch, 1, height, width)
        return SuperpixelAssignment(
            weights=weights,
            indices=ids,
            grid_height=grid_h,
            grid_width=grid_w,
            entropy=entropy,
        )


def pool_superpixel_statistics(
    feature: torch.Tensor,
    assignment: SuperpixelAssignment,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft mean/std, hard-member max, and region mass."""

    batch, channels, height, width = feature.shape
    pixels = height * width
    nodes = assignment.node_count
    values = feature.float().flatten(2).transpose(1, 2)  # B,P,C
    total = values.new_zeros((batch, nodes, channels))
    second = values.new_zeros((batch, nodes, channels))
    mass = values.new_zeros((batch, nodes, 1))
    for slot in range(assignment.weights.shape[1]):
        index = assignment.indices[slot].reshape(1, pixels, 1)
        index_c = index.expand(batch, pixels, channels)
        index_m = index.expand(batch, pixels, 1)
        weight = assignment.weights[:, slot].reshape(batch, pixels, 1).float()
        total.scatter_add_(1, index_c, values * weight)
        second.scatter_add_(1, index_c, values.square() * weight)
        mass.scatter_add_(1, index_m, weight)
    denominator = mass.clamp_min(1.0e-6)
    mean = total / denominator
    variance = (second / denominator - mean.square()).clamp_min(0.0)
    std = torch.sqrt(variance + 1.0e-6)

    # Max uses the dominant association only.  It preserves a small strong
    # damage response that a regional mean would otherwise erase.
    winner = assignment.weights.argmax(1)
    all_ids = assignment.indices.unsqueeze(0).expand(batch, -1, -1)
    hard_ids = torch.gather(all_ids, 1, winner.unsqueeze(1)).squeeze(1)
    maxima = values.new_full((batch, nodes, channels), float("-inf"))
    maxima.scatter_reduce_(
        1,
        hard_ids.unsqueeze(-1).expand(batch, pixels, channels),
        values,
        reduce="amax",
        include_self=True,
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return mean, std, maxima, mass


def unpool_superpixel_values(
    nodes: torch.Tensor,
    assignment: SuperpixelAssignment,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Softly broadcast node values to their associated pixels."""

    batch, _node_count, channels = nodes.shape
    height, width = output_size
    pixels = height * width
    output = nodes.new_zeros((batch, pixels, channels))
    for slot in range(assignment.weights.shape[1]):
        index = assignment.indices[slot].reshape(1, pixels, 1)
        index = index.expand(batch, pixels, channels)
        selected = torch.gather(nodes, 1, index)
        weight = assignment.weights[:, slot].reshape(batch, pixels, 1).float()
        output = output + selected * weight
    return output.transpose(1, 2).reshape(batch, channels, height, width)


class BoundaryGatedRegionGraph(nn.Module):
    """One conservative four-neighbour message-passing step."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.message = nn.Linear(channels, channels, bias=False)
        self.gate = nn.Linear(2 * channels, channels)
        self.transition = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _shift(value: torch.Tensor, dy: int, dx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros_like(value)
        valid = value.new_zeros((value.shape[0], 1, value.shape[2], value.shape[3]))
        src_y = slice(max(-dy, 0), value.shape[2] - max(dy, 0))
        dst_y = slice(max(dy, 0), value.shape[2] - max(-dy, 0))
        src_x = slice(max(-dx, 0), value.shape[3] - max(dx, 0))
        dst_x = slice(max(dx, 0), value.shape[3] - max(-dx, 0))
        output[:, :, dst_y, dst_x] = value[:, :, src_y, src_x]
        valid[:, :, dst_y, dst_x] = 1.0
        return output, valid

    def forward(
        self,
        nodes: torch.Tensor,
        rgb_nodes: torch.Tensor,
        support_nodes: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _count, channels = nodes.shape
        grid = nodes.transpose(1, 2).reshape(batch, channels, grid_height, grid_width)
        rgb = rgb_nodes.transpose(1, 2).reshape(batch, 3, grid_height, grid_width)
        support = support_nodes.transpose(1, 2).reshape(batch, 1, grid_height, grid_width)
        message_sum = torch.zeros_like(grid)
        weight_sum = grid.new_zeros((batch, 1, grid_height, grid_width))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour, valid = self._shift(grid, dy, dx)
            neighbour_rgb, _ = self._shift(rgb, dy, dx)
            neighbour_support, _ = self._shift(support, dy, dx)
            colour_distance = (rgb - neighbour_rgb).square().mean(1, keepdim=True)
            support_compatibility = 1.0 - (support - neighbour_support).abs().clamp(0.0, 1.0)
            weight = torch.exp(-colour_distance) * (0.25 + 0.75 * support_compatibility)
            weight = weight * valid
            message_sum = message_sum + neighbour * weight
            weight_sum = weight_sum + weight
        message = message_sum / weight_sum.clamp_min(1.0e-6)
        message_nodes = message.flatten(2).transpose(1, 2)
        transformed = self.message(message_nodes)
        gate = torch.sigmoid(self.gate(torch.cat([nodes, message_nodes], dim=-1)))
        update = torch.tanh(self.transition) * gate * transformed
        return nodes + update, update.abs().mean(-1, keepdim=True)


class SuperpixelEvidenceStage(nn.Module):
    """Stage2 RGB--SAR evidence pooling and optional region reasoning."""

    def __init__(
        self,
        source_channels: int,
        hidden_channels: int = 32,
        assignment_mode: str = "rgb",
        graph_refinement: bool = False,
        cell_size: int = 2,
        color_temperature: float = 0.50,
        compactness: float = 0.75,
    ) -> None:
        super().__init__()
        self.assignment_mode = assignment_mode
        self.rgb_projection = ConvGNAct(source_channels, hidden_channels, 1)
        self.sar_projection = ConvGNAct(source_channels, hidden_channels, 1)
        self.relation = nn.Sequential(
            ConvGNAct(6 * hidden_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels),
        )
        self.superpixels = RGBGuidedSoftSuperpixels(
            cell_size=cell_size,
            color_temperature=color_temperature,
            compactness=compactness,
        )
        node_input_channels = 3 * hidden_channels + 2
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_channels, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
        )
        self.graph = BoundaryGatedRegionGraph(hidden_channels) if graph_refinement else None
        self.output = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
        pre_rgb: torch.Tensor,
        building_support: torch.Tensor,
        base_occurrence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb = self.rgb_projection(rgb_feature.float())
        sar = self.sar_projection(sar_feature.float())
        difference = (rgb - sar).abs()
        product = rgb * sar
        local_mean = F.avg_pool2d(difference, 3, stride=1, padding=1)
        local_second = F.avg_pool2d(difference.square(), 3, stride=1, padding=1)
        local_std = torch.sqrt((local_second - local_mean.square()).clamp_min(0.0) + 1.0e-6)
        relation = self.relation(
            torch.cat([rgb, sar, difference, product, local_mean, local_std], dim=1)
        )

        size = relation.shape[-2:]
        rgb_low = F.interpolate(pre_rgb.float(), size=size, mode="bilinear", align_corners=False)
        support_low = F.interpolate(building_support.float(), size=size, mode="bilinear", align_corners=False)
        occurrence_low = F.interpolate(base_occurrence.float(), size=size, mode="bilinear", align_corners=False)
        assignment = self.superpixels(rgb_low, self.assignment_mode)
        mean, std, maximum, mass = pool_superpixel_statistics(relation, assignment)
        support_mean, _support_std, _support_max, _ = pool_superpixel_statistics(
            support_low, assignment
        )
        occurrence_mean, _occ_std, _occ_max, _ = pool_superpixel_statistics(
            occurrence_low, assignment
        )
        rgb_nodes, _rgb_std, _rgb_max, _ = pool_superpixel_statistics(
            self.superpixels._normalise_rgb(rgb_low), assignment
        )
        node = self.node_encoder(
            torch.cat([mean, std, maximum, support_mean, occurrence_mean], dim=-1)
        )
        graph_update = node.new_zeros((node.shape[0], node.shape[1], 1))
        if self.graph is not None:
            node, graph_update = self.graph(
                node,
                rgb_nodes,
                support_mean,
                assignment.grid_height,
                assignment.grid_width,
            )
        node_score = self.output(node)
        residual = unpool_superpixel_values(node_score, assignment, size)
        residual = residual * support_low
        diagnostics = {
            "h13_assignment_entropy": assignment.entropy,
            "h13_superpixel_mass_mean": mass.mean(1).reshape(-1, 1, 1, 1),
            "h13_superpixel_feature_std": std.mean(-1).reshape(
                std.shape[0], 1, assignment.grid_height, assignment.grid_width
            ),
            "h13_superpixel_building_overlap": support_mean.transpose(1, 2).reshape(
                support_mean.shape[0], 1, assignment.grid_height, assignment.grid_width
            ),
            "h13_graph_update_abs": graph_update.transpose(1, 2).reshape(
                graph_update.shape[0], 1, assignment.grid_height, assignment.grid_width
            ),
            "h13_superpixel_count": mass.new_full(
                (mass.shape[0], 1, 1, 1), float(assignment.node_count)
            ),
        }
        return residual, diagnostics


class DamageFormerRGBSARH13SPER(DamageFormerRGBSARH6SDCR):
    """Locked H6-A2 plus a safe regional occurrence residual."""

    def __init__(
        self,
        variant: str = "s2",
        pretrained_rgb: bool = True,
        latent_channels: Tuple[int, ...] = (64, 96, 128, 160),
        hidden_channels: int = 32,
        cell_size: int = 2,
        color_temperature: float = 0.50,
        compactness: float = 0.75,
        erosion_radius: int = 1,
    ) -> None:
        variant = str(variant).lower()
        if variant not in VALID_H13_VARIANTS:
            raise ValueError(f"unknown H13 variant {variant!r}")
        super().__init__(variant="a2", pretrained_rgb=pretrained_rgb, latent_channels=latent_channels)
        self.variant = variant
        self.h13_config = H13_VARIANTS[variant]
        self.cell_size = int(cell_size)
        self.color_temperature = float(color_temperature)
        self.compactness = float(compactness)
        self.erosion_radius = int(erosion_radius)
        if self.h13_config.assignment_mode != "none":
            self.region_head = SuperpixelEvidenceStage(
                source_channels=int(latent_channels[1]),
                hidden_channels=hidden_channels,
                assignment_mode=self.h13_config.assignment_mode,
                graph_refinement=self.h13_config.graph_refinement,
                cell_size=self.cell_size,
                color_temperature=self.color_temperature,
                compactness=self.compactness,
            )
            self.affected_transition = nn.Parameter(torch.tensor(0.0))

    def forward(self, pre_rgb: torch.Tensor, post_sar: torch.Tensor, return_dict: bool = False):
        with torch.no_grad():
            outputs = super().forward(pre_rgb, post_sar, return_dict=True)
        base_logits = outputs["output_dam"]
        base_occurrence, base_severity = factorize_building_logits(base_logits.float())
        outputs.update(
            {
                "h10_base_affected_logits": base_occurrence,
                "h10_base_severity_logits": base_severity,
                "h10_affected_logits": base_occurrence,
                "h10_severity_logits": base_severity,
            }
        )
        if self.h13_config.assignment_mode == "none":
            if return_dict:
                return outputs
            return outputs["output_loc"], outputs["output_dam"]

        device_type = "cuda" if pre_rgb.is_cuda else pre_rgb.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            building_probability = outputs["output_loc"].float().softmax(1)[:, 1:2]
            building_support = erode_building_support(
                building_probability, radius=self.erosion_radius
            )
            candidate_low, diagnostics = self.region_head(
                outputs["stage2_rgb_common"].float(),
                outputs["stage2_sar_common"].float(),
                pre_rgb.float(),
                building_support,
                base_occurrence,
            )
            candidate = F.interpolate(
                candidate_low,
                size=base_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if not torch.isfinite(candidate).all():
                raise FloatingPointError("non-finite H13 superpixel candidate")
            bounded = bound_occurrence_candidate(
                candidate, limit=OCCURRENCE_CANDIDATE_BOUND
            )
            transition = bounded_occurrence_transition(self.affected_transition)
            occurrence_residual = transition * bounded
            if not torch.isfinite(occurrence_residual).all():
                raise FloatingPointError("non-finite H13 occurrence residual")
            updated_logits = apply_hierarchical_residuals(
                base_logits.float(), occurrence_residual=occurrence_residual
            )
            if not torch.isfinite(updated_logits).all():
                raise FloatingPointError("non-finite H13 updated damage logits")
            affected, severity = factorize_building_logits(updated_logits)
        outputs.update(diagnostics)
        outputs.update(
            {
                "h13_occurrence_transition": transition.reshape(1, 1, 1, 1),
                "h13_occurrence_candidate_abs": candidate.detach().abs(),
                "h13_occurrence_residual_abs": occurrence_residual.abs(),
                "h13_building_support": building_support,
                "h10_affected_logits": affected,
                "h10_severity_logits": severity,
                "output_dam": updated_logits,
            }
        )
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_h13_sper_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
    cell_size: int = 2,
    color_temperature: float = 0.50,
    compactness: float = 0.75,
) -> DamageFormerRGBSARH13SPER:
    return DamageFormerRGBSARH13SPER(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
        cell_size=cell_size,
        color_temperature=color_temperature,
        compactness=compactness,
    )


__all__ = [
    "H13_VARIANTS",
    "VALID_H13_VARIANTS",
    "SuperpixelAssignment",
    "RGBGuidedSoftSuperpixels",
    "pool_superpixel_statistics",
    "unpool_superpixel_values",
    "BoundaryGatedRegionGraph",
    "SuperpixelEvidenceStage",
    "DamageFormerRGBSARH13SPER",
    "build_h13_sper_damageformer",
]
