# -*- coding: utf-8 -*-
"""Change-conditioned cross-modal unification for RGB--SAR damage mapping.

CCU-BDA is an independent, teacher-free experimental line.  It keeps the
parts of the BRIGHT pipeline that are properties of the task (RGB-only
building localisation and hierarchical severity), while replacing the old
pseudo-RGB SAR contract and post-hoc expert routing with three explicit
operations:

1. modality-specific RGB and native one-channel SAR encoders;
2. building-local correlation matching at semantic stages 3/4;
3. change-conditioned unification: align reliable unchanged content, retain
   reliable changed evidence, and reject unreliable SAR evidence.

The variants form a controlled ablation rather than a stack of unrelated
modules:

``c0`` independent encoders + late fusion;
``c1`` unconditional/global feature unification;
``c2`` building-local correlation at stages 3/4;
``c3`` c2 + change-conditioned unification;
``c4`` c3 + hierarchical Building/Affected/Destroyed decoder.
``e1`` locked H6-A2 + same-position Stage4 difference control;
``e2`` locked H6-A2 + multi-scale 3D spatio-temporal difference;
``e3`` locked H6-A2 + parameter-free Stage4 soft local matching.

No teacher, knowledge distillation, prototype memory, image translation, or
extra supervised loss is declared by this model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H4 import hierarchical_probabilities
from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_Heterogeneous import (
    STAGE_CHANNELS,
    ConvGNAct,
    MultiScaleDecoder,
    NativeSARPyramid,
    SwinRGBPyramid,
)


VALID_CCU_VARIANTS = (
    "c0", "c1", "c2", "c3", "c4", "e1", "e2", "e3",
)


@dataclass(frozen=True)
class CCUVariant:
    name: str
    global_unification: bool
    local_correlation: bool
    change_conditioning: bool
    hierarchical_severity: bool


CCU_VARIANTS: Mapping[str, CCUVariant] = {
    "c0": CCUVariant(
        "CCU-C0-ModalitySpecific-LateFusion", False, False, False, False
    ),
    "c1": CCUVariant(
        "CCU-C1-Unconditional-GlobalUnification", True, False, False, False
    ),
    "c2": CCUVariant(
        "CCU-C2-BuildingLocal-Correlation", False, True, False, False
    ),
    "c3": CCUVariant(
        "CCU-C3-ChangeConditioned-Unification", False, True, True, False
    ),
    "c4": CCUVariant(
        "CCU-C4-Hierarchical-DamageAssessment", False, True, True, True
    ),
    "e1": CCUVariant(
        "CCU-E1-H6A2-Stage4-SamePosition-Control", False, False, False, False
    ),
    "e2": CCUVariant(
        "CCU-E2-H6A2-Stage4-MultiScale-3D-Difference",
        False,
        False,
        False,
        False,
    ),
    "e3": CCUVariant(
        "CCU-E3-H6A2-Stage4-SoftLocalMatching", False, True, False, False
    ),
}


class SpatioTemporalDifference3D(nn.Module):
    """Small HeteCD-style multi-scale difference operator for E2.

    RGB and SAR features are stacked as a two-frame sequence.  Three
    depth-wise 3D convolutions measure same-position, local-context and
    dilated-context change.  Their deterministic initialisation computes
    ``SAR - RGB`` at the corresponding spatial scale, avoiding a random
    difference representation before the zero-gated residual opens.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.same = nn.Conv3d(
            channels,
            channels,
            kernel_size=(2, 1, 1),
            groups=channels,
            bias=False,
        )
        self.local = nn.Conv3d(
            channels,
            channels,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.context = nn.Conv3d(
            channels,
            channels,
            kernel_size=(2, 3, 3),
            padding=(0, 2, 2),
            dilation=(1, 2, 2),
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.reset_parameters()

    @staticmethod
    def _initialize_signed_average(layer: nn.Conv3d) -> None:
        spatial_area = layer.kernel_size[1] * layer.kernel_size[2]
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:, 0, 0].fill_(-1.0 / spatial_area)
            layer.weight[:, 0, 1].fill_(1.0 / spatial_area)

    def reset_parameters(self) -> None:
        self._initialize_signed_average(self.same)
        self._initialize_signed_average(self.local)
        self._initialize_signed_average(self.context)
        with torch.no_grad():
            self.fuse.weight.zero_()
            for channel in range(self.channels):
                for branch in range(3):
                    self.fuse.weight[channel, branch * self.channels + channel, 0, 0] = 1.0 / 3.0

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor
    ) -> torch.Tensor:
        if rgb.shape != sar.shape:
            raise ValueError("3D difference requires equal RGB/SAR shapes")
        sequence = torch.stack((rgb, sar), dim=2)
        differences = (
            self.same(sequence).squeeze(2),
            self.local(sequence).squeeze(2),
            self.context(sequence).squeeze(2),
        )
        return self.fuse(torch.cat(differences, dim=1))


class IndependentLateFusionStage(nn.Module):
    """The clean non-shared baseline used by C0 and shallow C2--C4 stages."""

    def __init__(self, in_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.rgb_projection = ConvGNAct(in_channels, latent_channels, 1)
        self.sar_projection = ConvGNAct(in_channels, latent_channels, 1)
        self.fuse = nn.Sequential(
            ConvGNAct(4 * latent_channels, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del building_prior
        rgb_latent = self.rgb_projection(rgb)
        sar_latent = self.sar_projection(sar)
        signed = sar_latent - rgb_latent
        descriptor = torch.cat(
            [rgb_latent, sar_latent, signed.abs(), signed], dim=1
        )
        joint = self.fuse(descriptor)
        one = torch.ones_like(joint[:, :1])
        zero = torch.zeros_like(one)
        return joint, {
            "rgb_latent": rgb_latent,
            "sar_latent": sar_latent,
            "reliability": one,
            "change_probability": zero,
            "unchanged_gate": zero,
            "changed_gate": zero,
            "rejected_gate": zero,
            "offset_x": zero,
            "offset_y": zero,
            "offset_magnitude": zero,
            "correlation_entropy": zero,
            "correlation_peak": one,
            "correlation_margin": one,
        }


class UnconditionalGlobalUnificationStage(nn.Module):
    """C1 baseline: force both modalities through one global shared feature.

    This is intentionally simple.  It is not proposed as the final method; it
    provides the necessary experimental control for the hypothesis that
    unconditional unification can remove useful damage evidence.
    """

    def __init__(self, in_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.rgb_projection = ConvGNAct(in_channels, latent_channels, 1)
        self.sar_projection = ConvGNAct(in_channels, latent_channels, 1)
        self.shared_projection = ConvGNAct(latent_channels, latent_channels, 3)
        self.fuse = nn.Sequential(
            ConvGNAct(4 * latent_channels, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del building_prior
        rgb_latent = self.rgb_projection(rgb)
        sar_latent = self.sar_projection(sar)
        shared = self.shared_projection(0.5 * (rgb_latent + sar_latent))
        descriptor = torch.cat(
            [shared, rgb_latent - shared, sar_latent - shared, (rgb_latent - sar_latent).abs()],
            dim=1,
        )
        joint = self.fuse(descriptor)
        one = torch.ones_like(joint[:, :1])
        zero = torch.zeros_like(one)
        return joint, {
            "rgb_latent": rgb_latent,
            "sar_latent": sar_latent,
            "unified_content": shared,
            "reliability": one,
            "change_probability": zero,
            "unchanged_gate": one,
            "changed_gate": zero,
            "rejected_gate": zero,
            "offset_x": zero,
            "offset_y": zero,
            "offset_magnitude": zero,
            "correlation_entropy": zero,
            "correlation_peak": one,
            "correlation_margin": one,
        }


class BuildingLocalCorrelation(nn.Module):
    """Compute a bounded RGB-query/SAR-key local correlation volume.

    The module does not warp the full SAR image.  It returns the expectation of
    nearby SAR values plus interpretable matching geometry.  Reliability is
    based only on correlation concentration (entropy and top-1/top-2 margin),
    not on the final damage logits; this is the key distinction from H7-style
    post-hoc expert routers.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 5,
        temperature: float = 0.10,
        topk: int | None = 9,
    ) -> None:
        super().__init__()
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("window_size must be an odd integer >= 3")
        neighbours = window_size * window_size
        if topk is not None and not 1 <= topk <= neighbours:
            raise ValueError("topk must be inside the correlation window")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.neighbours = neighbours
        self.topk = topk
        self.temperature = float(temperature)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

        radius = window_size // 2
        offsets = [
            (float(dx), float(dy))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        self.register_buffer(
            "offset_table",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb.shape != sar.shape:
            raise ValueError("local correlation requires equal RGB/SAR shapes")
        batch, channels, height, width = rgb.shape
        locations = height * width

        query = F.normalize(self.query(rgb).float(), dim=1).to(rgb.dtype)
        key = F.normalize(self.key(sar).float(), dim=1).to(sar.dtype)
        value = self.value(sar)
        key_patches = F.unfold(
            key,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).reshape(batch, channels, self.neighbours, locations)
        value_patches = F.unfold(
            value,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).reshape(batch, channels, self.neighbours, locations)
        query_flat = query.reshape(batch, channels, locations)
        scores = torch.einsum("bcl,bckl->bkl", query_flat, key_patches)
        scores = scores / self.temperature

        if self.topk is not None and self.topk < self.neighbours:
            selected, indices = scores.topk(self.topk, dim=1)
            sparse = torch.full_like(scores, torch.finfo(scores.dtype).min)
            scores = sparse.scatter(1, indices, selected)
        attention = scores.softmax(dim=1)
        aligned = torch.einsum("bkl,bckl->bcl", attention, value_patches)
        aligned = self.output(aligned.reshape(batch, channels, height, width))

        active_neighbours = float(self.topk or self.neighbours)
        entropy = -(
            attention.float().clamp_min(1e-8).log() * attention.float()
        ).sum(1, keepdim=True)
        if active_neighbours > 1:
            entropy = entropy / math.log(active_neighbours)
        else:
            entropy = torch.zeros_like(entropy)

        top_values = attention.float().topk(min(2, self.neighbours), dim=1).values
        peak = top_values[:, 0:1]
        margin = (
            top_values[:, 0:1] - top_values[:, 1:2]
            if top_values.shape[1] > 1
            else top_values[:, 0:1]
        )
        concentration = (1.0 - entropy).clamp(0.0, 1.0)
        # Margin is at most one.  Averaging it with entropy concentration is
        # deliberately conservative: a single sharp but ambiguous peak does
        # not immediately receive full reliability.
        reliability = (0.5 * concentration + 0.5 * margin).clamp(0.0, 1.0)

        offsets = self.offset_table.to(attention.device, attention.dtype)
        offset_x = torch.einsum("bkl,k->bl", attention, offsets[:, 0])
        offset_y = torch.einsum("bkl,k->bl", attention, offsets[:, 1])
        offset_x = offset_x.reshape(batch, 1, height, width)
        offset_y = offset_y.reshape(batch, 1, height, width)
        radius = max(self.window_size // 2, 1)
        # A perfectly centred/symmetric match has exactly zero expected
        # offset.  sqrt(x^2 + y^2) has an undefined derivative at (0, 0),
        # which poisoned the first C2 optimizer update even though its forward
        # value was finite.  Evaluate the norm in FP32 and add a tiny epsilon
        # so both the value and gradient remain finite under AMP.
        offset_squared = offset_x.float().square() + offset_y.float().square()
        offset_magnitude = (
            torch.sqrt(offset_squared + 1e-8)
            / (math.sqrt(2.0) * radius)
        ).to(offset_x.dtype)

        # Reverse SAR->RGB matching supplies an explicit forward/backward
        # cycle check.  The reverse offset is sampled at the forward expected
        # destination; a reliable pair should approximately return to its
        # starting RGB position.
        reverse_key_patches = F.unfold(
            query,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).reshape(batch, channels, self.neighbours, locations)
        reverse_query = key.reshape(batch, channels, locations)
        reverse_scores = torch.einsum(
            "bcl,bckl->bkl", reverse_query, reverse_key_patches
        ) / self.temperature
        if self.topk is not None and self.topk < self.neighbours:
            selected, indices = reverse_scores.topk(self.topk, dim=1)
            sparse = torch.full_like(
                reverse_scores, torch.finfo(reverse_scores.dtype).min
            )
            reverse_scores = sparse.scatter(1, indices, selected)
        reverse_attention = reverse_scores.softmax(dim=1)
        reverse_x = torch.einsum(
            "bkl,k->bl", reverse_attention, offsets[:, 0]
        ).reshape(batch, 1, height, width)
        reverse_y = torch.einsum(
            "bkl,k->bl", reverse_attention, offsets[:, 1]
        ).reshape(batch, 1, height, width)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=rgb.device, dtype=offset_x.dtype),
            torch.arange(width, device=rgb.device, dtype=offset_x.dtype),
            indexing="ij",
        )
        destination_x = xx.unsqueeze(0).expand(batch, -1, -1) + offset_x[:, 0]
        destination_y = yy.unsqueeze(0).expand(batch, -1, -1) + offset_y[:, 0]
        grid_x = 2.0 * destination_x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * destination_y / max(height - 1, 1) - 1.0
        forward_grid = torch.stack([grid_x, grid_y], dim=-1)
        sampled_reverse_x = F.grid_sample(
            reverse_x,
            forward_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled_reverse_y = F.grid_sample(
            reverse_y,
            forward_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        cycle_error = torch.sqrt(
            (offset_x + sampled_reverse_x).square()
            + (offset_y + sampled_reverse_y).square()
            + 1e-8
        )
        cycle_consistency = torch.exp(-cycle_error / float(radius))
        reliability = reliability * cycle_consistency.reshape(
            batch, 1, locations
        )

        def spatial(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, 1, height, width).to(rgb.dtype)

        return aligned, {
            "reliability_raw": spatial(reliability),
            "correlation_entropy": spatial(entropy),
            "correlation_peak": spatial(peak),
            "correlation_margin": spatial(margin),
            "offset_x": offset_x.to(rgb.dtype),
            "offset_y": offset_y.to(rgb.dtype),
            "offset_magnitude": offset_magnitude.to(rgb.dtype),
            "cycle_error": cycle_error.to(rgb.dtype),
            "cycle_consistency": cycle_consistency.to(rgb.dtype),
            "correlation_attention": attention.reshape(
                batch, self.neighbours, height, width
            ).to(rgb.dtype),
        }


class BuildingLocalCorrelationStage(nn.Module):
    """C2 stage: local matching in a soft RGB building region."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        window_size: int = 5,
        topk: int | None = 9,
    ) -> None:
        super().__init__()
        self.rgb_projection = ConvGNAct(in_channels, latent_channels, 1)
        self.sar_projection = ConvGNAct(in_channels, latent_channels, 1)
        self.matcher = BuildingLocalCorrelation(
            latent_channels,
            window_size=window_size,
            topk=topk,
        )
        self.fuse = nn.Sequential(
            ConvGNAct(4 * latent_channels + 4, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )

    @staticmethod
    def _prior(
        building_prior: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if building_prior is None:
            return torch.ones_like(reference[:, :1])
        if building_prior.shape[-2:] != reference.shape[-2:]:
            building_prior = F.interpolate(
                building_prior,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return building_prior.detach().to(reference.dtype).clamp(0.0, 1.0)

    def _base_forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_prior: torch.Tensor | None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        rgb_latent = self.rgb_projection(rgb)
        sar_latent = self.sar_projection(sar)
        aligned_sar, matching = self.matcher(rgb_latent, sar_latent)
        prior = self._prior(building_prior, rgb_latent)
        reliability = matching["reliability_raw"] * prior
        difference = aligned_sar - rgb_latent
        geometry = torch.cat(
            [
                reliability,
                matching["correlation_entropy"],
                matching["offset_magnitude"],
                prior,
            ],
            dim=1,
        )
        descriptor = torch.cat(
            [rgb_latent, aligned_sar, difference.abs(), difference, geometry],
            dim=1,
        )
        joint = self.fuse(descriptor)
        diagnostics = {
            "rgb_latent": rgb_latent,
            "sar_latent": sar_latent,
            "aligned_sar": aligned_sar,
            "building_prior": prior,
            "reliability": reliability,
            "change_probability": torch.zeros_like(reliability),
            "unchanged_gate": torch.zeros_like(reliability),
            "changed_gate": reliability,
            "rejected_gate": 1.0 - reliability,
            **matching,
        }
        return joint, rgb_latent, aligned_sar, difference, diagnostics

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        joint, _, _, _, diagnostics = self._base_forward(
            rgb, sar, building_prior
        )
        return joint, diagnostics


class ChangeConditionedUnificationStage(BuildingLocalCorrelationStage):
    """C3/C4 stage implementing the paper hypothesis.

    ``conditional_transition`` is zero initialized.  After loading the C2
    base weights, C3 is therefore exactly C2 before optimisation, while the
    transition itself receives a non-zero gradient.  This enables a strict
    causal ablation between local matching and change-conditioned use of the
    same matches.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        window_size: int = 5,
        topk: int | None = 9,
    ) -> None:
        super().__init__(
            in_channels,
            latent_channels,
            window_size=window_size,
            topk=topk,
        )
        hidden = max(16, latent_channels // 4)
        self.change_predictor = nn.Sequential(
            ConvGNAct(7, hidden, 3),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.change_predictor[-1].weight)
        nn.init.zeros_(self.change_predictor[-1].bias)
        self.conditional_fuse = nn.Sequential(
            ConvGNAct(5 * latent_channels + 5, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )
        self.conditional_transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base, rgb_latent, aligned_sar, difference, diagnostics = self._base_forward(
            rgb, sar, building_prior
        )
        cosine_distance = 1.0 - F.cosine_similarity(
            rgb_latent.float(), aligned_sar.float(), dim=1
        ).unsqueeze(1).to(rgb_latent.dtype)
        change_input = torch.cat(
            [
                difference.abs().mean(1, keepdim=True),
                cosine_distance,
                diagnostics["correlation_entropy"],
                diagnostics["correlation_peak"],
                diagnostics["correlation_margin"],
                diagnostics["offset_magnitude"],
                diagnostics["building_prior"],
            ],
            dim=1,
        )
        change_probability = torch.sigmoid(self.change_predictor(change_input))
        reliability = diagnostics["reliability"]
        unchanged_gate = reliability * (1.0 - change_probability)
        changed_gate = reliability * change_probability
        rejected_gate = 1.0 - reliability

        # Only reliable unchanged content is moved toward a shared midpoint.
        shared_midpoint = 0.5 * (rgb_latent + aligned_sar)
        unified_content = rgb_latent + unchanged_gate * (
            shared_midpoint - rgb_latent
        )
        # Reliable changed evidence is preserved as a signed and absolute
        # residual.  Low-reliability SAR contributes neither term.
        changed_signed = changed_gate * difference
        changed_absolute = changed_gate * difference.abs()
        accepted_sar = reliability * aligned_sar + rejected_gate * rgb_latent
        geometry = torch.cat(
            [
                reliability,
                change_probability,
                unchanged_gate,
                changed_gate,
                diagnostics["offset_magnitude"],
            ],
            dim=1,
        )
        conditional_descriptor = torch.cat(
            [
                rgb_latent,
                unified_content,
                accepted_sar,
                changed_absolute,
                changed_signed,
                geometry,
            ],
            dim=1,
        )
        conditional = self.conditional_fuse(conditional_descriptor)
        transition = torch.tanh(self.conditional_transition)
        joint = base + transition * (conditional - base)
        diagnostics.update(
            {
                "unified_content": unified_content,
                "accepted_sar": accepted_sar,
                "change_probability": change_probability,
                "unchanged_gate": unchanged_gate,
                "changed_gate": changed_gate,
                "rejected_gate": rejected_gate,
                "conditional_transition": transition.reshape(1, 1, 1, 1),
            }
        )
        return joint, diagnostics


class ControlledStage4Difference(nn.Module):
    """E1/E2/E3 adapter over frozen H6-A2 stage-4 features.

    E1 and E3 have exactly the same learnable architecture and parameter
    count.  Their only difference is the SAR evidence supplied to the
    descriptor: E1 uses the same-position feature, whereas E3 uses a 5x5
    soft local match.  E0 showed that the absolute stage-4 response peak is
    informative, while stage-3, entropy, margin and offset are not reliable;
    therefore only the peak is exposed to this first controlled model.

    The residual transition is exactly zero at construction.  Consequently
    both variants reproduce the locked H6-A2 output before optimisation.
    """

    def __init__(
        self,
        channels: int = 160,
        *,
        local_matching: bool,
        spatiotemporal_difference: bool = False,
        window_size: int = 5,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("controlled window_size must be odd and >= 3")
        if temperature <= 0:
            raise ValueError("controlled temperature must be positive")
        self.channels = int(channels)
        self.local_matching = bool(local_matching)
        self.spatiotemporal_difference = bool(spatiotemporal_difference)
        if self.local_matching and self.spatiotemporal_difference:
            raise ValueError(
                "local matching and 3D difference are mutually exclusive"
            )
        self.window_size = int(window_size)
        self.temperature = float(temperature)
        self.adapter = nn.Sequential(
            ConvGNAct(4 * channels + 1, 2 * channels, 1),
            ConvGNAct(2 * channels, channels, 3),
        )
        self.difference_3d = (
            SpatioTemporalDifference3D(channels)
            if self.spatiotemporal_difference
            else None
        )
        # Do not zero the adapter: with both adapter and transition at zero,
        # the complete branch would be gradient dead.  The transition alone
        # supplies the exact safe initialisation and receives the first update.
        self.transition = nn.Parameter(torch.tensor(0.0))

    def _soft_local_match(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb.shape != sar.shape:
            raise ValueError("controlled local matching requires equal shapes")
        batch, channels, height, width = rgb.shape
        locations = height * width
        neighbours = self.window_size * self.window_size
        radius = self.window_size // 2

        # Correlation is deliberately evaluated in FP32 under AMP.  E3 has no
        # cycle geometry or square-root offset path, removing the two unstable
        # mechanisms found in the old C2 smoke experiment.
        rgb_unit = F.normalize(rgb.float(), dim=1, eps=1e-6)
        sar_unit = F.normalize(sar.float(), dim=1, eps=1e-6)
        key_patches = F.unfold(
            sar_unit,
            kernel_size=self.window_size,
            padding=radius,
        ).reshape(batch, channels, neighbours, locations)
        value_patches = F.unfold(
            sar.float(),
            kernel_size=self.window_size,
            padding=radius,
        ).reshape(batch, channels, neighbours, locations)
        scores = torch.einsum(
            "bcl,bckl->bkl",
            rgb_unit.reshape(batch, channels, locations),
            key_patches,
        ).clamp(-1.0, 1.0)

        # Zero padding must not become a candidate at feature-map boundaries.
        valid = F.unfold(
            torch.ones(
                (batch, 1, height, width),
                device=rgb.device,
                dtype=torch.float32,
            ),
            kernel_size=self.window_size,
            padding=radius,
        ).reshape(batch, neighbours, locations)
        scores = scores.masked_fill(valid < 0.5, -1e4)
        attention = (scores / self.temperature).softmax(dim=1)
        matched = torch.einsum(
            "bkl,bckl->bcl", attention, value_patches
        ).reshape(batch, channels, height, width)
        peak = scores.amax(dim=1, keepdim=True).reshape(
            batch, 1, height, width
        )
        return matched.to(sar.dtype), peak.to(rgb.dtype)

    def forward(
        self,
        base_joint: torch.Tensor,
        rgb_common: torch.Tensor,
        sar_common: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.local_matching:
            matched_sar, peak = self._soft_local_match(rgb_common, sar_common)
        else:
            matched_sar = sar_common
            peak = F.cosine_similarity(
                rgb_common.float(), sar_common.float(), dim=1, eps=1e-6
            ).unsqueeze(1).to(rgb_common.dtype)
        difference = (
            self.difference_3d(rgb_common, sar_common)
            if self.difference_3d is not None
            else matched_sar - rgb_common
        )
        descriptor = torch.cat(
            [
                rgb_common,
                matched_sar,
                difference,
                difference.abs(),
                peak,
            ],
            dim=1,
        )
        residual = self.adapter(descriptor)
        building = F.interpolate(
            building_probability.detach(),
            size=base_joint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(base_joint.dtype).clamp(0.0, 1.0)
        transition = torch.tanh(self.transition)
        controlled = base_joint + transition * building * residual
        return controlled, {
            "controlled_correlation_peak": peak.detach(),
            "controlled_transition": transition.reshape(1, 1, 1, 1),
            "controlled_residual_abs": residual.detach().abs().mean(
                1, keepdim=True
            ),
            "controlled_matched_difference_abs": difference.detach().abs().mean(
                1, keepdim=True
            ),
            "controlled_3d_difference_abs": (
                difference.detach().abs().mean(1, keepdim=True)
                if self.difference_3d is not None
                else difference.new_zeros((difference.shape[0], 1, *difference.shape[-2:]))
            ),
        }


class DamageFormerRGBSARCCUControlled(DamageFormerRGBSARH6SDCR):
    """Frozen H6-A2 with one zero-gated controlled stage-4 residual."""

    def __init__(
        self,
        variant: str,
        pretrained_rgb: bool = True,
        correlation_window: int = 5,
        correlation_temperature: float = 0.10,
    ) -> None:
        variant = str(variant).lower()
        if variant not in ("e1", "e2", "e3"):
            raise ValueError("controlled CCU variant must be e1, e2 or e3")
        super().__init__(variant="a2", pretrained_rgb=pretrained_rgb)
        self.variant = variant
        self.ccu_config = CCU_VARIANTS[variant]
        self.stage4_control = ControlledStage4Difference(
            160,
            local_matching=variant == "e3",
            spatiotemporal_difference=variant == "e2",
            window_size=correlation_window,
            temperature=correlation_temperature,
        )
        self.freeze_h6_a2_base()

    def freeze_h6_a2_base(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("stage4_control."))

    def train(self, mode: bool = True):
        super().train(mode)
        # H6-A2 is a locked feature source.  Keep every inherited module,
        # including normalisation state, in evaluation mode.
        frozen_modules = (
            self.rgb_encoder,
            self.sar_encoder,
            self.p3_spatial_anchor,
            self.joint_stages,
            self.localization_decoder,
            self.damage_decoder,
            self.localization_head,
            self.damage_head,
            self.sar_aux_head,
            self.hierarchy_building_head,
            self.hierarchy_affected_head,
            self.hierarchy_destroyed_head,
            self.sensor_damage_adapters,
        )
        for module in frozen_modules:
            module.eval()
        self.stage4_control.train(mode)
        return self

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        diagnostics: Dict[str, torch.Tensor] = {}

        def controlled_hook(module, inputs, output):
            del module
            base_joint, base_diagnostics = output
            if len(inputs) < 6:
                raise RuntimeError("unexpected H6-A2 stage4 adapter interface")
            rgb_common = inputs[1]
            sar_common = inputs[2]
            building_probability = inputs[5]
            controlled, values = self.stage4_control(
                base_joint,
                rgb_common,
                sar_common,
                building_probability,
            )
            diagnostics.update(values)
            merged = dict(base_diagnostics)
            merged.update(values)
            return controlled, merged

        handle = self.sensor_damage_adapters["stage4"].register_forward_hook(
            controlled_hook
        )
        try:
            outputs = super().forward(pre_rgb, post_sar, return_dict=True)
        finally:
            handle.remove()
        outputs.update(diagnostics)
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


class DamageFormerRGBSARCCUBDA(nn.Module):
    """Teacher-free RGB/SAR building damage model for the CCU experiments."""

    def __init__(
        self,
        variant: str = "c3",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        correlation_window: int = 5,
        correlation_topk: int | None = 9,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in VALID_CCU_VARIANTS:
            raise ValueError(f"unknown CCU-BDA variant {variant!r}")
        if variant in ("e1", "e2", "e3"):
            raise ValueError(
                "E1/E2/E3 require DamageFormerRGBSARCCUControlled or "
                "build_ccu_bda(); they are not cold-start CCU variants"
            )
        if len(latent_channels) != 4:
            raise ValueError("latent_channels must contain four stages")
        self.variant = variant
        self.ccu_config = CCU_VARIANTS[variant]
        self.rgb_encoder = SwinRGBPyramid(pretrained=pretrained_rgb)
        self.sar_encoder = NativeSARPyramid()

        stages: List[nn.Module] = []
        for stage_index, (source_channels, latent) in enumerate(
            zip(STAGE_CHANNELS, latent_channels), start=1
        ):
            is_deep = stage_index >= 3
            if self.ccu_config.change_conditioning and is_deep:
                stage: nn.Module = ChangeConditionedUnificationStage(
                    source_channels,
                    latent,
                    window_size=correlation_window,
                    topk=correlation_topk,
                )
            elif self.ccu_config.local_correlation and is_deep:
                stage = BuildingLocalCorrelationStage(
                    source_channels,
                    latent,
                    window_size=correlation_window,
                    topk=correlation_topk,
                )
            elif self.ccu_config.global_unification:
                stage = UnconditionalGlobalUnificationStage(
                    source_channels, latent
                )
            else:
                stage = IndependentLateFusionStage(source_channels, latent)
            stages.append(stage)
        self.joint_stages = nn.ModuleList(stages)

        self.localization_decoder = MultiScaleDecoder(STAGE_CHANNELS, 256)
        self.damage_decoder = MultiScaleDecoder(latent_channels, 256)
        self.localization_head = nn.Conv2d(256, 2, 1)
        self.damage_head = nn.Conv2d(256, 4, 1)
        self.sar_aux_head = nn.Sequential(
            ConvGNAct(STAGE_CHANNELS[-1], 256, 1),
            nn.Conv2d(256, 4, 1),
        )
        if self.ccu_config.hierarchical_severity:
            self.hierarchy_building_head = nn.Conv2d(256, 2, 1)
            self.hierarchy_affected_head = nn.Conv2d(256, 2, 1)
            self.hierarchy_destroyed_head = nn.Conv2d(256, 2, 1)
            self.hierarchy_transition = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        if pre_rgb.ndim != 4 or pre_rgb.shape[1] != 3:
            raise ValueError(
                f"CCU-BDA expects RGB [B,3,H,W], got {tuple(pre_rgb.shape)}"
            )
        if post_sar.ndim != 4 or post_sar.shape[1] != 1:
            raise ValueError(
                f"CCU-BDA expects native SAR [B,1,H,W], got {tuple(post_sar.shape)}"
            )
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)

        localization_feature = self.localization_decoder(rgb_stages)
        localization_logits_low = self.localization_head(localization_feature)
        localization_probability = localization_logits_low.softmax(1)[:, 1:2]

        joint_stages: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for index, (stage, rgb, sar) in enumerate(
            zip(self.joint_stages, rgb_stages, sar_stages), start=1
        ):
            joint, stage_outputs = stage(
                rgb,
                sar,
                building_prior=localization_probability,
            )
            joint_stages.append(joint)
            for key, value in stage_outputs.items():
                diagnostics[f"stage{index}_{key}"] = value

        damage_feature = self.damage_decoder(joint_stages)
        # The optical branch defines where buildings are.  It is a soft prior,
        # not a hard mask, so localisation errors do not erase damage gradients.
        damage_feature = damage_feature * (
            0.5 + 0.5 * localization_probability.detach()
        )
        base_damage_low = self.damage_head(damage_feature)
        damage_low = base_damage_low
        hierarchy_outputs: Dict[str, torch.Tensor] = {}
        if self.ccu_config.hierarchical_severity:
            building_low = self.hierarchy_building_head(damage_feature)
            affected_low = self.hierarchy_affected_head(damage_feature)
            destroyed_low = self.hierarchy_destroyed_head(damage_feature)
            hierarchy_probability = hierarchical_probabilities(
                building_low, affected_low, destroyed_low
            )
            hierarchy_low = hierarchy_probability.clamp_min(1e-7).log()
            hierarchy_transition = torch.tanh(self.hierarchy_transition)
            damage_low = base_damage_low + hierarchy_transition * (
                hierarchy_low - base_damage_low
            )
            hierarchy_outputs = {
                "hierarchy_building_logits": F.interpolate(
                    building_low, size=input_size, mode="bilinear", align_corners=False
                ),
                "hierarchy_affected_logits": F.interpolate(
                    affected_low, size=input_size, mode="bilinear", align_corners=False
                ),
                "hierarchy_destroyed_logits": F.interpolate(
                    destroyed_low, size=input_size, mode="bilinear", align_corners=False
                ),
                "hierarchy_transition": hierarchy_transition.reshape(1, 1, 1, 1),
            }

        outputs: Dict[str, torch.Tensor] = {
            "output_loc": F.interpolate(
                localization_logits_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "output_dam": F.interpolate(
                damage_low, size=input_size, mode="bilinear", align_corners=False
            ),
            "base_damage_logits": F.interpolate(
                base_damage_low,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "sar_aux_logits": F.interpolate(
                self.sar_aux_head(sar_stages[-1]),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "localization_probability_low": localization_probability,
            **hierarchy_outputs,
            **diagnostics,
        }
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def load_registration_transfer(
    model: DamageFormerRGBSARCCUBDA,
    checkpoint_path: str,
) -> Dict[str, object]:
    """Load SAR/matching weights from a registration pretraining artifact.

    Accepted keys may be stored directly or below ``model``/``state_dict``.
    Only the SAR encoder and deep correlation modules are migrated.  RGB,
    localisation and damage heads are deliberately excluded so the transfer
    experiment has a clear causal interpretation.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("registration checkpoint has no state dictionary")
    current = model.state_dict()
    allowed_prefixes = (
        "sar_encoder.",
        "joint_stages.2.rgb_projection.",
        "joint_stages.2.sar_projection.",
        "joint_stages.2.matcher.",
        "joint_stages.3.rgb_projection.",
        "joint_stages.3.sar_projection.",
        "joint_stages.3.matcher.",
    )
    migrated: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for raw_key, value in state.items():
        key = str(raw_key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if not key.startswith(allowed_prefixes):
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            migrated[key] = value
        else:
            skipped.append(key)
    if not migrated:
        raise RuntimeError("registration checkpoint provided no compatible CCU weights")
    current.update(migrated)
    model.load_state_dict(current, strict=True)
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "migrated_key_count": len(migrated),
        "skipped_keys": sorted(skipped),
        "source_protocol": checkpoint.get("protocol"),
        "source_step": int(checkpoint.get("step", 0)),
    }


def load_controlled_variant_transfer(
    model: nn.Module,
    checkpoint_path: str,
) -> Dict[str, object]:
    """Migrate a preceding CCU ablation without changing shared weights.

    E1/E2/E3 require a strict H6-A2 source; the only missing target keys may be
    their new stage-4 controller.  The legacy C2 -> C3 -> C4 progression is
    retained for checkpoint compatibility.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_variant = str(checkpoint.get("variant", "")).lower()
    target_variant = str(getattr(model, "variant", "")).lower()
    if target_variant in ("e1", "e2", "e3"):
        if source_variant != "a2":
            raise RuntimeError(
                f"controlled {target_variant.upper()} requires H6-A2, "
                f"received {source_variant!r}"
            )
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, Mapping) or not state:
            raise TypeError("H6-A2 source checkpoint has no model state dictionary")
        state = {
            (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
            for key, value in state.items()
        }
        current = model.state_dict()
        unexpected = sorted(set(state) - set(current))
        shape_mismatch = sorted(
            key
            for key in state
            if key in current and tuple(state[key].shape) != tuple(current[key].shape)
        )
        if unexpected or shape_mismatch:
            raise RuntimeError(
                "H6-A2 controlled source is incompatible: "
                f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
            )
        migrated = {
            key: value
            for key, value in state.items()
            if key in current and key not in shape_mismatch
        }
        missing = sorted(set(current) - set(migrated))
        if not missing or any(
            not key.startswith("stage4_control.") for key in missing
        ):
            raise RuntimeError(
                "controlled H6-A2 migration may leave only stage4_control keys; "
                f"missing={missing[:8]}"
            )
        current.update(migrated)
        model.load_state_dict(current, strict=True)
        freeze = getattr(model, "freeze_h6_a2_base", None)
        if callable(freeze):
            freeze()
        return {
            "path": str(Path(checkpoint_path).resolve()),
            "source_protocol": checkpoint.get("protocol"),
            "source_variant": source_variant,
            "source_step": int(checkpoint.get("step", 0)),
            "migrated_key_count": len(migrated),
            "new_target_keys": missing,
            "strict_h6_a2_base": True,
        }

    allowed = {("c2", "c3"), ("c3", "c4")}
    if (source_variant, target_variant) not in allowed:
        raise RuntimeError(
            "controlled CCU transfer only supports C2->C3 or C3->C4; "
            f"received {source_variant!r}->{target_variant!r}"
        )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("CCU source checkpoint has no state dictionary")
    current = model.state_dict()
    migrated: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[str] = []
    unexpected: List[str] = []
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key not in current:
            unexpected.append(key)
        elif tuple(current[key].shape) != tuple(value.shape):
            shape_mismatch.append(key)
        else:
            migrated[key] = value
    if unexpected or shape_mismatch:
        raise RuntimeError(
            "CCU source contains incompatible shared state: "
            f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )
    if not migrated:
        raise RuntimeError("CCU source checkpoint did not provide compatible weights")
    current.update(migrated)
    model.load_state_dict(current, strict=True)
    new_keys = sorted(set(current) - set(migrated))
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "source_protocol": checkpoint.get("protocol"),
        "source_variant": source_variant,
        "source_step": int(checkpoint.get("step", 0)),
        "migrated_key_count": len(migrated),
        "new_target_keys": new_keys,
    }


def zero_extra_regularization(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    unchanged_weight: float = 1.0,
    orthogonality_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Declare that CCU-BDA adds no auxiliary representation loss.

    The arguments match the legacy trainer hook.  CCU is trained by the same
    localisation, damage, Lovasz and SAR-auxiliary objectives as the clean
    heterogeneous baseline; the novelty is architectural rather than a larger
    loss collection.
    """

    del labels, unchanged_weight, orthogonality_weight
    reference = outputs["output_dam"]
    zero = reference.sum() * 0.0
    return {
        "representation_loss": zero,
        "common_unchanged_loss": zero,
        "common_unique_orthogonality_loss": zero,
    }


def build_ccu_bda(
    variant: str,
    pretrained_rgb: bool = True,
) -> nn.Module:
    if str(variant).lower() in ("e1", "e2", "e3"):
        return DamageFormerRGBSARCCUControlled(
            variant=variant,
            pretrained_rgb=pretrained_rgb,
        )
    return DamageFormerRGBSARCCUBDA(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_CCU_VARIANTS",
    "CCU_VARIANTS",
    "IndependentLateFusionStage",
    "UnconditionalGlobalUnificationStage",
    "BuildingLocalCorrelation",
    "BuildingLocalCorrelationStage",
    "ChangeConditionedUnificationStage",
    "SpatioTemporalDifference3D",
    "ControlledStage4Difference",
    "DamageFormerRGBSARCCUControlled",
    "DamageFormerRGBSARCCUBDA",
    "build_ccu_bda",
    "load_controlled_variant_transfer",
    "load_registration_transfer",
    "zero_extra_regularization",
]
