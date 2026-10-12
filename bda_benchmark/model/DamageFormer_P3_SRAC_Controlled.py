# -*- coding: utf-8 -*-
"""Controlled structure/retrieval replacements for P3's pixel damage head.

This module is *inspired* by the public SRAC overview, SSN, and region-memory
segmentation work.  It is not a reproduction of SRAC: the paper's full text
and official implementation were not publicly available when this controlled
experiment was written.

The experimental contract is deliberately narrow:

``sr0``
    Exact frozen P3 guard.  No structure classifier is allowed to affect the
    output.
``sr1``
    P3 encoders, RGB localisation, and soft building prior are frozen.  The
    final pixel damage head is replaced by a differentiable RGB-guided
    superpixel classifier.  Region predictions are the primary output, not a
    residual on P3.
``sr1p``
    Keeps P3's mature damage classifier and inserts a differentiable
    superpixel projection in feature space immediately before that classifier.
    A zero-initialised bounded transition makes the initial class
    probabilities exactly equal to P3.  P3 retains exact authority over
    Background-versus-Building; the structured feature may change only the
    conditional Intact/Damaged/Destroyed distribution.
``sr1pa``
    Partition-only pretraining.  A boundary-aware partition reads frozen RGB
    features, frozen P3 building probability, and pre-RGB Sobel edges.  Direct
    building reconstruction/boundary/entropy losses train the partition while
    the prediction remains exactly P3 (the projection transition is frozen).
``sr1pb``
    Loads a completed ``sr1pa`` partition and opens the same bounded,
    non-negative safe feature projection used by ``sr1p``.  This separates
    partition quality from downstream damage utility.
``sr1pai`` / ``sr1pib``
    Controlled iterative counterparts of ``sr1pa`` / ``sr1pb``.  They keep
    all inputs, losses, and permissions unchanged, but replace the one-shot
    assignment with three differentiable assignment--seed-update rounds.
``sr1paa``
    Stops seed competition entirely.  Frozen P3 RGB geometry and frozen H6
    native-SAR Stage2/3 features train an eight-neighbour same-damage affinity
    head inside labelled buildings.  P3 outputs remain bitwise unchanged and
    no affinity propagation is allowed before a validation-only evidence gate.
``sr2``
    Adds a class-balanced, train-split-only region memory to ``sr1``.  The
    memory is updated only by the training objective, never by forward/eval.

No test sample, test feature, or validation label can enter the memory bank.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.configs.experiment_registry import (
    P3_SRAC_VARIANTS as REGISTERED_P3_SRAC_VARIANTS,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H13_SPER import (
    SuperpixelAssignment,
    pool_superpixel_statistics,
    unpool_superpixel_values,
)
from bda_benchmark.model.DamageFormer_RGBSAR_Heterogeneous import (
    NativeSARPyramid,
)
from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
    P3PseudoSARSource,
)


VALID_P3_SRAC_VARIANTS = REGISTERED_P3_SRAC_VARIANTS


@dataclass(frozen=True)
class P3SRACVariant:
    name: str
    primary_region_classifier: bool
    retrieval_memory: bool
    preserve_p3_occurrence: bool
    feature_projection: bool
    boundary_aware_partition: bool = False
    partition_pretrain: bool = False
    assignment_iterations: int = 1
    affinity_pretrain: bool = False


P3_SRAC_VARIANTS = {
    "sr0": P3SRACVariant("P3-SR0-Exact-P3-Guard", False, False, False, False),
    "sr1": P3SRACVariant("P3-SR1-Structure-Primary", True, False, False, False),
    "sr1f": P3SRACVariant(
        "P3-SR1F-P3-Occurrence-Region-Severity", True, False, True, False
    ),
    "sr1p": P3SRACVariant(
        "P3-SR1P-Safe-Feature-Structure-Projection", False, False, True, True
    ),
    "sr1pa": P3SRACVariant(
        "P3-SR1PA-Building-Partition-Pretraining",
        False, False, True, True, True, True,
    ),
    "sr1pb": P3SRACVariant(
        "P3-SR1PB-Boundary-Aware-Feature-Projection",
        False, False, True, True, True, False,
    ),
    "sr1pai": P3SRACVariant(
        "P3-SR1PAI-Iterative-Building-Partition-Pretraining",
        False, False, True, True, True, True, 3,
    ),
    "sr1pib": P3SRACVariant(
        "P3-SR1PIB-Iterative-Boundary-Aware-Feature-Projection",
        False, False, True, True, True, False, 3,
    ),
    "sr1paa": P3SRACVariant(
        "P3-SR1PAA-Building-Conditioned-Damage-Affinity",
        False, False, True, False, False, False, 1, True,
    ),
    "sr2": P3SRACVariant(
        "P3-SR2-Train-Memory-Retrieval", True, True, False, False
    ),
}

if tuple(P3_SRAC_VARIANTS) != VALID_P3_SRAC_VARIANTS:
    raise RuntimeError(
        "P3-SRAC implementation registry diverged from "
        "configs.experiment_registry"
    )


AFFINITY_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _shift_with_validity(
    value: torch.Tensor,
    dy: int,
    dx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift NCHW values without wrap-around and expose valid pair pixels."""
    if value.ndim != 4:
        raise ValueError("affinity shifts require an NCHW tensor")
    height, width = value.shape[-2:]
    shifted = torch.zeros_like(value)
    valid = torch.zeros(
        value.shape[0], 1, height, width,
        dtype=torch.bool,
        device=value.device,
    )
    source_y = slice(max(0, dy), min(height, height + dy))
    target_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, dx), min(width, width + dx))
    target_x = slice(max(0, -dx), min(width, width - dx))
    shifted[:, :, target_y, target_x] = value[:, :, source_y, source_x]
    valid[:, :, target_y, target_x] = True
    return shifted, valid


class BuildingConditionedDamageAffinity(nn.Module):
    """Predict local damage-region affinity inside an RGB building support.

    P3 RGB features and its building probability provide stable geometry.
    Frozen native-SAR Stage2/3 features provide post-event interior evidence.
    A shared pair head scores the eight neighbouring relations from explicit
    [centre, neighbour, absolute difference, product] descriptors.  It never
    changes P3 logits; this module is an evidence audit until a separate gate
    explicitly admits a downstream region operator.
    """

    offsets = AFFINITY_OFFSETS

    def __init__(self, embedding_channels: int = 32) -> None:
        super().__init__()
        channels = int(embedding_channels)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(256, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.sar2_projection = nn.Sequential(
            nn.Conv2d(192, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.sar3_projection = nn.Sequential(
            nn.Conv2d(384, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(3 * channels + 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.pair_head = nn.Sequential(
            nn.Conv2d(4 * channels + 4, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, bias=True),
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        sar_stage2: torch.Tensor,
        sar_stage3: torch.Tensor,
        building_probability: torch.Tensor,
        rgb_edge: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        size = rgb_feature.shape[-2:]
        sar2 = F.interpolate(
            self.sar2_projection(sar_stage2),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        sar3 = F.interpolate(
            self.sar3_projection(sar_stage3),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        building = F.interpolate(
            building_probability.float(), size=size, mode="bilinear",
            align_corners=False,
        )
        edge = F.interpolate(
            rgb_edge.float(), size=size, mode="bilinear", align_corners=False
        )
        context = self.context(
            torch.cat(
                (self.rgb_projection(rgb_feature), sar2, sar3, building, edge),
                dim=1,
            )
        )
        logits, validity = [], []
        for dy, dx in self.offsets:
            neighbour, valid = _shift_with_validity(context, dy, dx)
            neighbour_building, _ = _shift_with_validity(building, dy, dx)
            neighbour_edge, _ = _shift_with_validity(edge, dy, dx)
            pair = torch.cat(
                (
                    context,
                    neighbour,
                    (context - neighbour).abs(),
                    context * neighbour,
                    building,
                    neighbour_building,
                    edge,
                    neighbour_edge,
                ),
                dim=1,
            )
            logits.append(self.pair_head(pair))
            validity.append(valid)
        return torch.cat(logits, dim=1), torch.cat(validity, dim=1)


class LearnableLocalSuperpixels(nn.Module):
    """Differentiable local SSN-style assignments from frozen RGB features.

    Every feature pixel considers only the surrounding 3x3 grid seeds.  The
    candidate restriction keeps regions compact and bounds memory to nine
    associations per pixel.  Unlike H13, the appearance embedding and the
    assignment are differentiable; gradients from the primary region
    classifier therefore reach the partition network.
    """

    def __init__(
        self,
        in_channels: int = 256,
        embedding_channels: int = 8,
        cell_size: int = 4,
        temperature: float = 0.20,
        compactness: float = 0.50,
        boundary_aware: bool = False,
        assignment_iterations: int = 1,
    ) -> None:
        super().__init__()
        if cell_size < 1:
            raise ValueError("cell_size must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if compactness < 0.0:
            raise ValueError("compactness must be non-negative")
        if assignment_iterations < 1:
            raise ValueError("assignment_iterations must be positive")
        self.cell_size = int(cell_size)
        self.boundary_aware = bool(boundary_aware)
        self.assignment_iterations = int(assignment_iterations)
        self.temperature = float(temperature)
        self.compactness = float(compactness)
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        if self.boundary_aware:
            hidden = 32
            self.embedding = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.GroupNorm(4, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                nn.GroupNorm(4, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, embedding_channels, 1, bias=False),
            )
            self.register_buffer(
                "scheduled_temperature", torch.tensor(0.50), persistent=True
            )
        else:
            self.embedding = nn.Sequential(
                nn.Conv2d(in_channels, embedding_channels, 1, bias=False),
                nn.GroupNorm(1, embedding_channels),
                nn.GELU(),
                nn.Conv2d(embedding_channels, embedding_channels, 1, bias=False),
            )
            self.scheduled_temperature = None

    def set_progress(self, progress: float) -> None:
        if self.scheduled_temperature is None:
            return
        progress = min(max(float(progress), 0.0), 1.0)
        value = 0.50 + progress * (0.15 - 0.50)
        self.scheduled_temperature.fill_(value)

    @property
    def effective_temperature(self) -> float:
        if self.scheduled_temperature is None:
            return self.temperature
        return float(self.scheduled_temperature.item())

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
        base_y, base_x = yy // cell, xx // cell
        ids, valid, distance = [], [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                gy, gx = base_y + dy, base_x + dx
                ok = (gy >= 0) & (gy < grid_h) & (gx >= 0) & (gx < grid_w)
                safe_y = gy.clamp(0, grid_h - 1)
                safe_x = gx.clamp(0, grid_w - 1)
                ids.append((safe_y * grid_w + safe_x).reshape(-1))
                centre_y = (safe_y.float() + 0.5) * cell - 0.5
                centre_x = (safe_x.float() + 0.5) * cell - 0.5
                distance.append(
                    (
                        ((yy.float() - centre_y) / cell).square()
                        + ((xx.float() - centre_x) / cell).square()
                    ).reshape(-1)
                )
                valid.append(ok.reshape(-1))
        return (
            torch.stack(ids),
            torch.stack(valid),
            torch.stack(distance),
            grid_h,
            grid_w,
        )

    @staticmethod
    def _update_seeds(
        feature: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        """Differentiable SSN-style weighted seed recomputation."""
        batch, channels, height, width = feature.shape
        pixels = height * width
        values = feature.flatten(2).transpose(1, 2)  # B,P,C
        total = values.new_zeros((batch, node_count, channels))
        mass = values.new_zeros((batch, node_count, 1))
        for slot in range(weights.shape[1]):
            index = ids[slot].reshape(1, pixels, 1)
            index_c = index.expand(batch, pixels, channels)
            index_m = index.expand(batch, pixels, 1)
            weight = weights[:, slot].reshape(batch, pixels, 1).float()
            total = total.scatter_add(1, index_c, values * weight)
            mass = mass.scatter_add(1, index_m, weight)
        seeds = total / mass.clamp_min(1.0e-6)
        return F.normalize(seeds, dim=-1).transpose(1, 2)

    def _assign(
        self,
        feature: torch.Tensor,
        seeds: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        spatial: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        pixels = height * width
        gather_index = ids.reshape(1, 1, -1).expand(batch, channels, -1)
        candidates = torch.gather(seeds, 2, gather_index)
        candidates = candidates.reshape(batch, channels, 9, pixels).permute(0, 2, 1, 3)
        pixels_feature = feature.flatten(2).unsqueeze(1)
        appearance = (pixels_feature - candidates).square().mean(2)
        logits = -appearance / self.effective_temperature
        logits = logits - self.compactness * spatial.unsqueeze(0)
        logits = logits.masked_fill(~valid.unsqueeze(0), -1.0e4)
        return logits.softmax(1)

    def forward(self, rgb_feature: torch.Tensor) -> SuperpixelAssignment:
        feature = F.normalize(self.embedding(rgb_feature).float(), dim=1)
        batch, _channels, height, width = feature.shape
        ids, valid, spatial, grid_h, grid_w = self._candidate_geometry(
            height, width, feature.device
        )
        pad_h = grid_h * self.cell_size - height
        pad_w = grid_w * self.cell_size - width
        padded = F.pad(feature, (0, pad_w, 0, pad_h), mode="replicate")
        seeds = F.avg_pool2d(
            padded, kernel_size=self.cell_size, stride=self.cell_size
        ).flatten(2)
        seeds = F.normalize(seeds, dim=1)
        node_count = grid_h * grid_w
        weights = None
        weight_history = []
        seed_movements = []
        for iteration in range(self.assignment_iterations):
            weights = self._assign(feature, seeds, ids, valid, spatial)
            weight_history.append(weights)
            if iteration + 1 < self.assignment_iterations:
                updated_seeds = self._update_seeds(
                    feature, weights, ids, node_count
                )
                seed_movements.append(
                    (updated_seeds.float() - seeds.float()).abs().mean()
                )
                seeds = updated_seeds
        assert weights is not None
        top2 = weights.topk(k=2, dim=1).values
        zero = weights.new_zeros(())
        self.last_diagnostics = {
            "srac_assignment_top1_probability": top2[:, 0].mean().detach(),
            "srac_assignment_top12_margin": (
                top2[:, 0] - top2[:, 1]
            ).mean().detach(),
            "srac_q0_q1_l1": (
                (weight_history[1] - weight_history[0]).abs().mean().detach()
                if len(weight_history) > 1 else zero
            ),
            "srac_q1_q2_l1": (
                (weight_history[2] - weight_history[1]).abs().mean().detach()
                if len(weight_history) > 2 else zero
            ),
            "srac_seed_move_0_1": (
                seed_movements[0].detach() if seed_movements else zero
            ),
            "srac_seed_move_1_2": (
                seed_movements[1].detach() if len(seed_movements) > 1 else zero
            ),
        }
        entropy = -(
            weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()
        ).sum(1) / math.log(9.0)
        return SuperpixelAssignment(
            weights=weights,
            indices=ids,
            grid_height=grid_h,
            grid_width=grid_w,
            entropy=entropy.reshape(batch, 1, height, width),
        )


class ClassBalancedRegionMemory(nn.Module):
    """Fixed-capacity FIFO memory with the same capacity for each class."""

    def __init__(self, channels: int, capacity_per_class: int = 512) -> None:
        super().__init__()
        if capacity_per_class < 1:
            raise ValueError("capacity_per_class must be positive")
        self.channels = int(channels)
        self.capacity_per_class = int(capacity_per_class)
        self.register_buffer(
            "keys", torch.zeros(3, capacity_per_class, channels), persistent=True
        )
        self.register_buffer(
            "valid", torch.zeros(3, capacity_per_class, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "cursor", torch.zeros(3, dtype=torch.long), persistent=True
        )

    @torch.no_grad()
    def enqueue(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> None:
        features = F.normalize(features.detach().float(), dim=-1)
        labels = labels.detach().long()
        if keep is not None:
            keep = keep.detach().bool()
            features, labels = features[keep], labels[keep]
        for class_index in range(3):
            selected = features[labels == class_index]
            if selected.numel() == 0:
                continue
            for feature in selected:
                slot = int(self.cursor[class_index].item()) % self.capacity_per_class
                self.keys[class_index, slot].copy_(feature)
                self.valid[class_index, slot] = True
                self.cursor[class_index] += 1

    def retrieve(
        self, query: torch.Tensor, topk_per_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return supports, class ids, and validity for every query node."""
        query = F.normalize(query.float(), dim=-1)
        supports, labels, validity = [], [], []
        for class_index in range(3):
            mask = self.valid[class_index]
            keys = self.keys[class_index]
            score = torch.einsum("bnc,mc->bnm", query, keys)
            score = score.masked_fill(~mask.reshape(1, 1, -1), -1.0e4)
            count = min(int(topk_per_class), self.capacity_per_class)
            top_score, top_index = score.topk(count, dim=-1)
            expanded = keys.reshape(1, 1, self.capacity_per_class, self.channels)
            expanded = expanded.expand(query.shape[0], query.shape[1], -1, -1)
            selected = torch.gather(
                expanded,
                2,
                top_index.unsqueeze(-1).expand(-1, -1, -1, self.channels),
            )
            supports.append(selected)
            labels.append(
                torch.full_like(top_index, class_index, dtype=torch.long)
            )
            validity.append(top_score > -9.0e3)
        return (
            torch.cat(supports, dim=2),
            torch.cat(labels, dim=2),
            torch.cat(validity, dim=2),
        )

    @property
    def size(self) -> int:
        return int(self.valid.sum().item())


class RegionRetrievalClassifier(nn.Module):
    def __init__(
        self,
        relation_channels: int = 128,
        embedding_channels: int = 128,
        memory_capacity: int = 512,
        retrieval_topk: int = 2,
        use_memory: bool = False,
    ) -> None:
        super().__init__()
        self.use_memory = bool(use_memory)
        self.retrieval_topk = int(retrieval_topk)
        # mean/std/max relation statistics plus raw SAR mean/std.
        input_channels = relation_channels * 3 + 2
        self.main_projection = nn.Sequential(
            nn.Linear(input_channels, embedding_channels),
            nn.LayerNorm(embedding_channels),
            nn.GELU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.raw_skip = nn.Sequential(
            nn.Linear(2, embedding_channels),
            nn.LayerNorm(embedding_channels),
        )
        self.skip_scale = nn.Parameter(torch.tensor(0.0))
        self.label_embedding = nn.Embedding(3, embedding_channels)
        self.cross_attention = nn.MultiheadAttention(
            embedding_channels, num_heads=4, batch_first=True
        )
        self.retrieval_scale = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_channels),
            nn.Linear(embedding_channels, embedding_channels),
            nn.GELU(),
            nn.Linear(embedding_channels, 3),
        )
        self.memory = ClassBalancedRegionMemory(
            embedding_channels, capacity_per_class=memory_capacity
        )

    def forward(
        self,
        relation_mean: torch.Tensor,
        relation_std: torch.Tensor,
        relation_max: torch.Tensor,
        raw_statistics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        node_input = torch.cat(
            (relation_mean, relation_std, relation_max, raw_statistics), dim=-1
        )
        node = self.main_projection(node_input)
        node = node + torch.tanh(self.skip_scale) * self.raw_skip(raw_statistics)
        retrieval_valid_fraction = node.new_zeros(())
        if self.use_memory and self.memory.size > 0:
            support, labels, valid = self.memory.retrieve(node, self.retrieval_topk)
            support = support + self.label_embedding(labels)
            flat_query = node.reshape(-1, 1, node.shape[-1])
            flat_support = support.reshape(-1, support.shape[2], support.shape[3])
            flat_valid = valid.reshape(-1, valid.shape[2])
            # MultiheadAttention requires at least one unmasked key per query.
            none_valid = ~flat_valid.any(1)
            if none_valid.any():
                flat_valid = flat_valid.clone()
                flat_valid[none_valid, 0] = True
                flat_support = flat_support.clone()
                flat_support[none_valid, 0] = 0.0
            retrieved, _ = self.cross_attention(
                flat_query,
                flat_support,
                flat_support,
                key_padding_mask=~flat_valid,
                need_weights=False,
            )
            retrieved = retrieved.reshape_as(node)
            node = node + torch.tanh(self.retrieval_scale) * retrieved
            retrieval_valid_fraction = valid.float().mean()
        logits = self.classifier(node)
        diagnostics = {
            "srac_memory_size": logits.new_tensor(float(self.memory.size)),
            "srac_retrieval_valid_fraction": retrieval_valid_fraction,
            "srac_skip_scale": torch.tanh(self.skip_scale).detach(),
            "srac_retrieval_scale": torch.tanh(self.retrieval_scale).detach(),
        }
        return logits, node, diagnostics


class P3SRACControlled(nn.Module):
    """P3 with a controlled replacement of only the final damage decision."""

    resume_protocol = "P3_SRAC_controlled_resume_v1"
    best_weights_protocol = "P3_SRAC_controlled_best_v1"
    experiment_name = "P3_SRAC_Controlled"

    def __init__(
        self,
        p3: DamageFormer_S8_MICB_DSMIL,
        variant: str,
        cell_size: int = 4,
        relation_channels: int = 128,
        memory_capacity: int = 512,
        retrieval_topk: int = 2,
        native_sar_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in VALID_P3_SRAC_VARIANTS:
            raise ValueError(f"unknown P3-SRAC variant: {variant!r}")
        self.variant = variant
        self.config = P3_SRAC_VARIANTS[variant]
        self.p3 = p3
        for parameter in self.p3.parameters():
            parameter.requires_grad_(False)
        partition_channels = 258 if self.config.boundary_aware_partition else 256
        self.partition = LearnableLocalSuperpixels(
            in_channels=partition_channels,
            cell_size=cell_size,
            boundary_aware=self.config.boundary_aware_partition,
            assignment_iterations=self.config.assignment_iterations,
        )
        self.native_sar_encoder = (
            native_sar_encoder
            if native_sar_encoder is not None
            else (NativeSARPyramid() if self.config.affinity_pretrain else None)
        )
        self.damage_affinity = (
            BuildingConditionedDamageAffinity()
            if self.config.affinity_pretrain
            else None
        )
        if self.native_sar_encoder is not None:
            self.native_sar_encoder.requires_grad_(False)
        self.relation_projection = nn.Sequential(
            nn.Conv2d(256 * 5, relation_channels, 1, bias=False),
            nn.GroupNorm(8, relation_channels),
            nn.GELU(),
            nn.Conv2d(relation_channels, relation_channels, 3, padding=1, bias=False),
        )
        self.region_classifier = RegionRetrievalClassifier(
            relation_channels=relation_channels,
            memory_capacity=memory_capacity,
            retrieval_topk=retrieval_topk,
            use_memory=self.config.retrieval_memory,
        )
        # SR1P changes representation before P3's original classifier rather
        # than learning a replacement classifier.  At zero, the structured
        # path is an exact probability-level P3 guard.  The scalar receives a
        # first-step gradient; after it opens, gradients also reach the soft
        # partition through the pool/unpool projection.
        self.structure_transition = nn.Parameter(torch.zeros(()))
        if self.config.feature_projection:
            self.relation_projection.requires_grad_(False)
            self.region_classifier.requires_grad_(False)
            if self.config.partition_pretrain:
                self.structure_transition.requires_grad_(False)
        else:
            self.structure_transition.requires_grad_(False)
        if variant == "sr0":
            self.partition.requires_grad_(False)
            self.relation_projection.requires_grad_(False)
            self.region_classifier.requires_grad_(False)
        if self.config.affinity_pretrain:
            self.partition.requires_grad_(False)
            self.relation_projection.requires_grad_(False)
            self.region_classifier.requires_grad_(False)
            self.structure_transition.requires_grad_(False)
        # Keeps SR0 constructible by the common optimizer without affecting P3.
        self.sr0_guard_parameter = nn.Parameter(
            torch.zeros(()), requires_grad=(variant == "sr0")
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.p3.eval()
        if self.native_sar_encoder is not None:
            self.native_sar_encoder.eval()
        return self

    def set_optimizer_progress(self, step: int, total_steps: int) -> None:
        denominator = max(int(total_steps), 1)
        self.partition.set_progress(float(step) / float(denominator))

    @staticmethod
    def _rgb_edge(
        pre_rgb: torch.Tensor, output_size: Tuple[int, int]
    ) -> torch.Tensor:
        gray = pre_rgb.float().mean(1, keepdim=True)
        kernel_x = gray.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(-1, -2)
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
        edge = F.interpolate(edge, size=output_size, mode="bilinear", align_corners=False)
        scale = edge.flatten(2).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1.0e-6)
        return (edge / scale).clamp(0.0, 5.0)

    def checkpoint_contract(self) -> Dict[str, object]:
        return {
            "protocol": "P3_SRAC_controlled_model_v1",
            "variant": self.variant,
            "p3_frozen": True,
            "prediction_unit": (
                "pixel_p3"
                if self.variant == "sr0" or self.config.affinity_pretrain
                else (
                    "superpixel_feature_projection_to_p3_classifier"
                    if self.config.feature_projection
                    else "learned_superpixel_primary"
                )
            ),
            "occurrence_probability": (
                "p3_damage_foreground"
                if self.config.preserve_p3_occurrence
                else "rgb_localisation_building"
            ),
            "retrieval_memory": bool(self.config.retrieval_memory),
            "feature_projection": bool(self.config.feature_projection),
            "boundary_aware_partition": bool(
                self.config.boundary_aware_partition
            ),
            "partition_pretrain": bool(self.config.partition_pretrain),
            "assignment_iterations": int(self.config.assignment_iterations),
            "affinity_pretrain": bool(self.config.affinity_pretrain),
            "native_sar_encoder_frozen": bool(
                self.native_sar_encoder is not None
                and all(
                    not parameter.requires_grad
                    for parameter in self.native_sar_encoder.parameters()
                )
            ),
            "p3_damage_classifier_reused": bool(self.config.feature_projection),
            "safe_probability_initialisation": bool(self.config.feature_projection),
            "memory_scope": "train_only_class_balanced",
            "test_memory_updates": False,
        }

    def forward(
        self, pre_rgb: torch.Tensor, post_sar: torch.Tensor, return_dict: bool = False
    ):
        if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
            raise ValueError("P3-SRAC expects native/display SAR [B,1|2,H,W]")
        p3_post = P3PseudoSARSource.to_p3_input(post_sar)
        with torch.no_grad():
            output_loc, base_damage, aux = self.p3(
                pre_rgb, p3_post, return_aux=True
            )
        if self.variant == "sr0":
            result = {
                "output_loc": output_loc,
                "output_dam": base_damage,
                "srac_exact_p3_guard": base_damage.new_ones(()),
                "srac_memory_size": base_damage.new_zeros(()),
            }
            return result if return_dict else (output_loc, base_damage)

        rgb = aux["rgb_feature"]
        sar = aux["sar_feature"]
        fused = aux["fused_roi"]
        if self.config.affinity_pretrain:
            if self.native_sar_encoder is None or self.damage_affinity is None:
                raise RuntimeError("SR1PAA native-SAR affinity modules are missing")
            building_prior = aux["p_building"].detach().float()
            rgb_edge = self._rgb_edge(pre_rgb, rgb.shape[-2:])
            with torch.no_grad():
                native_stages = self.native_sar_encoder(post_sar[:, :1])
            affinity_logits, affinity_valid = self.damage_affinity(
                rgb,
                native_stages[1],
                native_stages[2],
                building_prior,
                rgb_edge,
            )
            result = {
                "output_loc": output_loc,
                "output_dam": base_damage,
                "srac_affinity_logits": affinity_logits,
                "srac_affinity_valid": affinity_valid,
                "srac_affinity_offsets": AFFINITY_OFFSETS,
                "srac_affinity_pretrain": base_damage.new_ones(()),
                "srac_exact_p3_guard": base_damage.new_ones(()),
                "srac_memory_size": base_damage.new_zeros(()),
            }
            return result if return_dict else (output_loc, base_damage)
        if self.config.boundary_aware_partition:
            building_prior = aux["p_building"].detach().float()
            if building_prior.shape[-2:] != rgb.shape[-2:]:
                building_prior = F.interpolate(
                    building_prior,
                    size=rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            rgb_edge = self._rgb_edge(pre_rgb, rgb.shape[-2:])
            partition_input = torch.cat((rgb, building_prior, rgb_edge), dim=1)
        else:
            partition_input = rgb
        assignment = self.partition(partition_input)
        assignment_diagnostics = dict(self.partition.last_diagnostics)

        if self.config.feature_projection:
            node_feature, _node_std, _node_max, mass = pool_superpixel_statistics(
                fused, assignment
            )
            projected_feature = unpool_superpixel_values(
                node_feature, assignment, fused.shape[-2:]
            )
            # A structure projection may move only from the original feature
            # towards its region-pooled reconstruction.  A negative value
            # would be an uncontrolled high-pass extrapolation, not a
            # superpixel projection.  The straight-through positive clamp
            # keeps the forward value exactly zero at initialisation while
            # retaining a usable first-step gradient.
            positive_transition = self.structure_transition + (
                self.structure_transition.clamp_min(0.0)
                - self.structure_transition
            ).detach()
            transition = torch.tanh(positive_transition)
            structured_feature = fused + transition * (projected_feature - fused)

            # P3 is frozen, but autograd must traverse clf_2 to train the
            # structured feature path.  Frozen parameters do not prevent
            # gradients with respect to structured_feature.
            structured_logits_low = self.p3.clf_2(structured_feature)
            structured_logits = F.interpolate(
                structured_logits_low,
                size=pre_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Preserve P3's Background/Building marginal exactly.  Only the
            # conditional I/D/X distribution is supplied by the structured
            # feature and P3's original classifier.
            base_probability = base_damage.float().softmax(1)
            base_background = base_probability[:, :1]
            base_foreground = 1.0 - base_background
            conditional = structured_logits[:, 1:4].float().softmax(1)
            probability = torch.cat(
                (base_background, base_foreground * conditional), dim=1
            )
            probability = probability.clamp_min(1.0e-7)
            probability = probability / probability.sum(
                1, keepdim=True
            ).clamp_min(1.0e-7)
            output_damage = probability.log()
            result = {
                "output_loc": output_loc,
                "output_dam": output_damage,
                "srac_assignment": assignment,
                "srac_node_mass": mass,
                "srac_partition_entropy": assignment.entropy.mean(),
                "srac_partition_temperature": output_damage.new_tensor(
                    self.partition.effective_temperature
                ),
                "srac_base_damage_logits": base_damage,
                "srac_feature_projection": output_damage.new_ones(()),
                "srac_boundary_partition": output_damage.new_tensor(
                    float(self.config.boundary_aware_partition)
                ),
                "srac_partition_pretrain": output_damage.new_tensor(
                    float(self.config.partition_pretrain)
                ),
                "srac_structure_transition": transition.detach(),
                "srac_projection_delta_abs": (
                    projected_feature.float() - fused.float()
                ).abs().mean().detach(),
                "srac_memory_size": output_damage.new_zeros(()),
                "srac_retrieval_valid_fraction": output_damage.new_zeros(()),
                **assignment_diagnostics,
            }
            return result if return_dict else (output_loc, output_damage)

        relation = self.relation_projection(
            torch.cat((rgb, sar, (rgb - sar).abs(), rgb * sar, fused), dim=1)
        )
        relation_mean, relation_std, relation_max, mass = pool_superpixel_statistics(
            relation, assignment
        )
        raw_display = post_sar[:, -1:].float()
        raw_low = F.interpolate(
            raw_display, size=relation.shape[-2:], mode="bilinear", align_corners=False
        )
        raw_mean, raw_std, _raw_max, _raw_mass = pool_superpixel_statistics(
            raw_low, assignment
        )
        raw_statistics = torch.cat((raw_mean, raw_std), dim=-1)
        node_logits, node_embedding, diagnostics = self.region_classifier(
            relation_mean, relation_std, relation_max, raw_statistics
        )
        region_logits = unpool_superpixel_values(
            node_logits, assignment, relation.shape[-2:]
        )
        region_probability = region_logits.softmax(1)
        if self.config.preserve_p3_occurrence:
            # SR1F changes only the conditional I/D/X decision.  P3's mature
            # four-class head retains exact authority over Background versus
            # Building, avoiding SR1's second (confounded) occurrence change.
            region_probability_full = F.interpolate(
                region_probability,
                size=pre_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            region_probability_full = region_probability_full / (
                region_probability_full.sum(1, keepdim=True).clamp_min(1.0e-7)
            )
            base_probability = base_damage.float().softmax(1)
            base_background = base_probability[:, :1]
            base_foreground = 1.0 - base_background
            probability = torch.cat(
                (base_background, base_foreground * region_probability_full), dim=1
            )
        else:
            building_probability = aux["p_building"].detach().float()
            probability_low = torch.cat(
                (
                    1.0 - building_probability,
                    building_probability * region_probability,
                ),
                dim=1,
            )
            probability = F.interpolate(
                probability_low,
                size=pre_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probability = probability.clamp_min(1.0e-7)
        probability = probability / probability.sum(1, keepdim=True).clamp_min(1.0e-7)
        output_damage = probability.log()
        result = {
            "output_loc": output_loc,
            "output_dam": output_damage,
            "srac_assignment": assignment,
            "srac_node_logits": node_logits,
            "srac_node_embedding": node_embedding,
            "srac_node_mass": mass,
            "srac_partition_entropy": assignment.entropy.mean(),
            "srac_base_damage_logits": base_damage,
            **assignment_diagnostics,
            **diagnostics,
        }
        return result if return_dict else (output_loc, output_damage)


def load_frozen_p3(
    checkpoint_path: str | Path,
) -> Tuple[DamageFormer_S8_MICB_DSMIL, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("P3 checkpoint does not contain a model state mapping")
    p3 = DamageFormer_S8_MICB_DSMIL(mil_residual_scale=0.0)
    incompatible = p3.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval()
    return p3, {
        "path": str(Path(checkpoint_path).resolve()),
        "step": int(checkpoint.get("step", 0)),
        "protocol": checkpoint.get("protocol", "unreported"),
        "strict_load": True,
    }


def load_frozen_native_sar_encoder(
    checkpoint_path: str | Path,
) -> Tuple[NativeSARPyramid, Dict[str, object]]:
    """Strictly recover only H6's native one-channel SAR encoder."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("H6 checkpoint does not contain a model state mapping")
    normalised = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    prefix = "sar_encoder."
    encoder_state = {
        key[len(prefix):]: value
        for key, value in normalised.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError("H6 checkpoint contains no sar_encoder.* parameters")
    encoder = NativeSARPyramid()
    incompatible = encoder.load_state_dict(encoder_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict native-SAR encoder load failed: {incompatible}")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder, {
        "path": str(Path(checkpoint_path).resolve()),
        "step": int(checkpoint.get("step", 0)),
        "protocol": checkpoint.get("protocol", "unreported"),
        "variant": checkpoint.get("variant", "unreported"),
        "namespace": prefix,
        "strict_load": True,
    }


def build_p3_srac_controlled(
    variant: str,
    p3_checkpoint: str | Path,
    native_sar_checkpoint: str | Path | None = None,
    **kwargs,
) -> P3SRACControlled:
    p3, provenance = load_frozen_p3(p3_checkpoint)
    native_provenance: Dict[str, object] = {}
    if P3_SRAC_VARIANTS[variant.lower()].affinity_pretrain:
        if native_sar_checkpoint is None:
            raise RuntimeError("SR1PAA requires a frozen H6 native-SAR checkpoint")
        encoder, native_provenance = load_frozen_native_sar_encoder(
            native_sar_checkpoint
        )
        kwargs["native_sar_encoder"] = encoder
    model = P3SRACControlled(p3, variant, **kwargs)
    model.p3_provenance = provenance
    model.native_sar_provenance = native_provenance
    return model


__all__ = [
    "VALID_P3_SRAC_VARIANTS",
    "P3_SRAC_VARIANTS",
    "LearnableLocalSuperpixels",
    "BuildingConditionedDamageAffinity",
    "AFFINITY_OFFSETS",
    "ClassBalancedRegionMemory",
    "RegionRetrievalClassifier",
    "P3SRACControlled",
    "build_p3_srac_controlled",
    "load_frozen_native_sar_encoder",
]
