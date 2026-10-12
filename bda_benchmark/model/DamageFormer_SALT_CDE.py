# -*- coding: utf-8 -*-
"""SALT-CDE: stable-anchor local transport for RGB--SAR damage assessment.

The family is deliberately isolated from P3/P5.  D0 and D0N are controls;
D1--D3 share one native-SAR architecture and change exactly one mechanism:

* D1: per-pixel local transport;
* D2: transport offsets propagated from the outer building ring;
* D3: D2 plus a learned reliability gate (trained counterfactually);
* D2R: reliability-weighted, bounded ring correction over D1;
* D3V2: D2R plus an explicitly supervised global pair verifier.

The transport branch is a residual over a colocated native-SAR fusion branch.
Consequently a rejected transport falls back to D0N-like fusion rather than
removing the post-event observation altogether.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_Heterogeneous import (
    STAGE_CHANNELS,
    ConvGNAct,
    DamageFormerRGBSARHeterogeneous,
    MultiScaleDecoder,
    NaiveHeterogeneousStage,
    NativeSARPyramid,
    OriginalDamageFormerControl,
    SwinRGBPyramid,
)
from .IntactAnchoredAlignment import warp_with_offsets


VALID_SALT_VARIANTS = ("d0", "d0n", "d1", "d2", "d3", "d2r", "d3v2")


@dataclass(frozen=True)
class SALTVariant:
    name: str
    local_transport: bool
    stable_ring: bool
    safe_ring_residual: bool
    reliability_gate: bool
    global_pair_gate: bool


SALT_VARIANTS: Mapping[str, SALTVariant] = {
    "d0": SALTVariant("D0-Original-DamageFormer", False, False, False, False, False),
    "d0n": SALTVariant("D0N-NativeSAR-Colocated-Control", False, False, False, False, False),
    "d1": SALTVariant("D1-Building-Local-Transport", True, False, False, False, False),
    "d2": SALTVariant("D2-Stable-Ring-Transport", True, True, False, False, False),
    "d3": SALTVariant("D3-Counterfactual-Stable-Ring-Transport", True, True, False, True, False),
    "d2r": SALTVariant("D2R-Safe-Residual-Ring-Transport", True, True, True, False, False),
    "d3v2": SALTVariant("D3V2-Explicit-Pair-Verified-Transport", True, True, True, True, True),
}


def _offset_table(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)],
        device=device,
        dtype=dtype,
    )


def outer_ring(probability: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Differentiable soft outer ring; no target label is used at inference."""
    if kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError("ring kernel must be odd and at least three")
    probability = probability.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(probability, kernel_size, stride=1, padding=kernel_size // 2)
    return (dilated - probability).clamp(0.0, 1.0)


def propagate_anchor_offsets(
    offsets: torch.Tensor,
    anchor: torch.Tensor,
    kernel_size: int = 9,
    minimum_support: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spread offsets estimated on stable anchors to their local interior."""
    if offsets.ndim != 4 or offsets.shape[1] != 2:
        raise ValueError("offsets must be [B,2,H,W]")
    if anchor.shape != offsets[:, :1].shape:
        raise ValueError("anchor/offset shape mismatch")
    pad = kernel_size // 2
    support = F.avg_pool2d(anchor.float(), kernel_size, stride=1, padding=pad)
    numerator = F.avg_pool2d(
        offsets.float() * anchor.float(), kernel_size, stride=1, padding=pad
    )
    propagated = numerator / support.clamp_min(float(minimum_support))
    available = support >= float(minimum_support)
    return torch.where(available, propagated, offsets.float()), support


class StableAnchorTransportStage(nn.Module):
    """Explicit local correlation followed by optional outer-ring propagation."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        *,
        radius: int = 2,
        temperature: float = 0.10,
        stable_ring: bool = False,
        safe_ring_residual: bool = False,
        reliability_gate: bool = False,
        global_pair_gate: bool = False,
        ring_kernel: int = 7,
        propagation_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.stable_ring = bool(stable_ring)
        self.safe_ring_residual = bool(safe_ring_residual)
        self.reliability_gate = bool(reliability_gate)
        self.global_pair_gate = bool(global_pair_gate)
        self.ring_kernel = int(ring_kernel)
        self.propagation_kernel = int(propagation_kernel)
        self.rgb = ConvGNAct(in_channels, latent_channels, 1)
        self.sar = ConvGNAct(in_channels, latent_channels, 1)
        self.query = ConvGNAct(latent_channels, 32, 1, activation=False)
        self.key = ConvGNAct(latent_channels, 32, 1, activation=False)
        self.transport_fuse = nn.Sequential(
            ConvGNAct(3 * latent_channels + 4, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )
        self.base_fuse = nn.Sequential(
            ConvGNAct(3 * latent_channels, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )
        gate_channels = 9 if self.global_pair_gate else 4
        self.gate_head = nn.Sequential(
            ConvGNAct(gate_channels, 16, 3),
            nn.Conv2d(16, 1, 1),
        )
        # Register this parameter only for the new variants.  Keeping it absent
        # elsewhere preserves strict loading of the completed D1/D2/D3 runs.
        self.ring_strength_logit = (
            nn.Parameter(torch.tensor(-4.0)) if self.safe_ring_residual else None
        )
        # D1/D2 use the complete transport residual.  D3 learns a spatial gate.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def _match(self, rgb: torch.Tensor, sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, height, width = rgb.shape
        neighbours = (2 * self.radius + 1) ** 2
        query = F.normalize(self.query(rgb).float(), dim=1, eps=1e-6)
        key = F.normalize(self.key(sar).float(), dim=1, eps=1e-6)
        patches = F.unfold(
            key, kernel_size=2 * self.radius + 1, padding=self.radius
        ).reshape(batch, key.shape[1], neighbours, height * width)
        correlation = torch.einsum(
            "bcl,bckl->bkl", query.flatten(2), patches
        ).reshape(batch, neighbours, height, width)
        valid = F.unfold(
            torch.ones((batch, 1, height, width), device=rgb.device),
            kernel_size=2 * self.radius + 1,
            padding=self.radius,
        ).reshape(batch, neighbours, height, width)
        correlation = correlation.masked_fill(valid < 0.5, -1e4)
        probability = (correlation / self.temperature).softmax(1)
        table = _offset_table(self.radius, rgb.device, probability.dtype)
        x = (probability * table[:, 0].view(1, -1, 1, 1)).sum(1)
        y = (probability * table[:, 1].view(1, -1, 1, 1)).sum(1)
        offsets = torch.stack((x, y), 1)
        entropy = -(
            probability.clamp_min(1e-8).log() * probability
        ).sum(1, keepdim=True) / math.log(float(neighbours))
        peaks = probability.topk(2, dim=1).values
        margin = peaks[:, :1] - peaks[:, 1:2]
        return {
            "correlation": correlation,
            "probability": probability,
            "offset": offsets,
            "entropy": entropy,
            "margin": margin,
        }

    def forward(
        self,
        rgb_source: torch.Tensor,
        sar_source: torch.Tensor,
        building_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb = self.rgb(rgb_source)
        sar = self.sar(sar_source)
        building = F.interpolate(
            building_probability.float(), rgb.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        ring = outer_ring(building, self.ring_kernel)
        matching = self._match(rgb, sar)
        direct_offset = matching["offset"]
        support = torch.ones_like(building)
        offset = direct_offset
        if self.stable_ring:
            if self.safe_ring_residual:
                match_reliability = (
                    (1.0 - matching["entropy"])
                    * (matching["margin"] * matching["probability"].shape[1]).clamp(0.0, 1.0)
                ).detach()
                anchor_weight = ring * match_reliability
                propagated, support = propagate_anchor_offsets(
                    direct_offset, anchor_weight, self.propagation_kernel
                )
                support_confidence = support / (support + 0.05)
                if self.ring_strength_logit is None:  # defensive; impossible by construction
                    raise RuntimeError("safe ring residual has no learnable strength")
                blend = torch.sigmoid(self.ring_strength_logit) * support_confidence
                offset = direct_offset + blend * (propagated - direct_offset)
            else:
                offset, support = propagate_anchor_offsets(
                    direct_offset, ring, self.propagation_kernel
                )
        aligned = warp_with_offsets(sar, offset.to(dtype=sar.dtype))
        base = self.base_fuse(torch.cat((rgb, sar, (rgb - sar).abs()), 1))
        # D3V2 must be able to reject the post-event observation itself.  In
        # old D3, gate=0 still selected ``base``, which already contained the
        # possibly shuffled SAR and therefore was not a genuine safe fallback.
        if self.global_pair_gate:
            absent_sar = torch.zeros_like(sar)
            fallback = self.base_fuse(
                torch.cat((rgb, absent_sar, rgb.abs()), 1)
            )
        else:
            fallback = base
        if self.global_pair_gate:
            aligned_similarity = (
                1.0 + F.cosine_similarity(rgb.float(), aligned.float(), dim=1).unsqueeze(1)
            ) * 0.5
            weight = (building + ring).clamp(0.0, 1.0)
            pair_statistics = torch.cat(
                (1.0 - matching["entropy"], matching["margin"], aligned_similarity), 1
            )
            pooled = (pair_statistics * weight).flatten(2).sum(2) / weight.flatten(2).sum(2).clamp_min(1.0)
            pooled = pooled[:, :, None, None].expand(-1, -1, rgb.shape[-2], rgb.shape[-1])
            gate_input = torch.cat(
                (
                    matching["entropy"], matching["margin"], building,
                    support.clamp(0, 1), aligned_similarity,
                    (rgb.float() - aligned.float()).abs().mean(1, keepdim=True),
                    pooled,
                ),
                1,
            )
        else:
            gate_input = torch.cat(
                (matching["entropy"], matching["margin"], building, support.clamp(0, 1)), 1
            )
        gate = torch.sigmoid(self.gate_head(gate_input)) if self.reliability_gate else torch.ones_like(building)
        transported = self.transport_fuse(
            torch.cat(
                (
                    rgb,
                    aligned,
                    (rgb - aligned).abs(),
                    matching["entropy"],
                    matching["margin"],
                    building,
                    ring,
                ),
                1,
            )
        )
        fused = fallback + gate * (transported - fallback)
        diagnostics = {
            "entropy": matching["entropy"],
            "margin": matching["margin"],
            "offset_x": offset[:, 0:1],
            "offset_y": offset[:, 1:2],
            "direct_offset_x": direct_offset[:, 0:1],
            "direct_offset_y": direct_offset[:, 1:2],
            "anchor_ring": ring,
            "anchor_support": support,
            "sar_gate": gate,
            "base_feature": fallback,
        }
        if self.ring_strength_logit is not None:
            diagnostics["ring_strength"] = torch.sigmoid(
                self.ring_strength_logit
            ).reshape(1, 1, 1, 1)
        return fused, diagnostics


class DamageFormerSALT(nn.Module):
    def __init__(
        self,
        variant: str,
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        radius: int = 2,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in ("d1", "d2", "d3", "d2r", "d3v2"):
            raise ValueError("DamageFormerSALT requires d1/d2/d3/d2r/d3v2")
        self.variant = variant
        self.variant_config = SALT_VARIANTS[variant]
        self.rgb_encoder = SwinRGBPyramid(pretrained=pretrained_rgb)
        self.sar_encoder = NativeSARPyramid()
        self.localization_decoder = MultiScaleDecoder(STAGE_CHANNELS, 256)
        self.localization_head = nn.Conv2d(256, 2, 1)
        # Matching only at semantic stages 2/3 keeps the memory bounded and
        # avoids asking the texture-heavy first or tiny fourth stage to solve
        # cross-modal correspondence.
        self.joint_stages = nn.ModuleList()
        for index, (source, latent) in enumerate(zip(STAGE_CHANNELS, latent_channels)):
            if index in (1, 2):
                module: nn.Module = StableAnchorTransportStage(
                    source,
                    latent,
                    radius=radius,
                    stable_ring=self.variant_config.stable_ring,
                    safe_ring_residual=self.variant_config.safe_ring_residual,
                    reliability_gate=self.variant_config.reliability_gate,
                    global_pair_gate=self.variant_config.global_pair_gate,
                )
            else:
                module = NaiveHeterogeneousStage(source, latent)
            self.joint_stages.append(module)
        self.damage_decoder = MultiScaleDecoder(latent_channels, 256)
        self.damage_head = nn.Conv2d(256, 4, 1)
        self.sar_aux_head = nn.Sequential(
            ConvGNAct(STAGE_CHANNELS[-1], 256, 1), nn.Conv2d(256, 4, 1)
        )

    def _decode(
        self,
        rgb_stages: Sequence[torch.Tensor],
        sar_stages: Sequence[torch.Tensor],
        size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        loc_feature = self.localization_decoder(rgb_stages)
        loc_low = self.localization_head(loc_feature)
        building = loc_low.softmax(1)[:, 1:2]
        joint: List[torch.Tensor] = []
        bases: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for index, (module, rgb, sar) in enumerate(
            zip(self.joint_stages, rgb_stages, sar_stages), start=1
        ):
            if isinstance(module, StableAnchorTransportStage):
                value, info = module(rgb, sar, building.detach())
                base = info.pop("base_feature")
            else:
                value, info = module(rgb, sar)
                base = value
            joint.append(value)
            bases.append(base)
            diagnostics.update({f"stage{index}_{key}": item for key, item in info.items()})
        # The same decoder reads transported and colocated features.  This is
        # important for the D1-D0N attribution: the transport experiment does
        # not gain a second damage decoder merely to create its fallback.
        damage_low = self.damage_head(self.damage_decoder(joint))
        base_low = self.damage_head(self.damage_decoder(bases))
        outputs = {
            "output_loc": F.interpolate(loc_low, size=size, mode="bilinear", align_corners=False),
            "output_dam": F.interpolate(damage_low, size=size, mode="bilinear", align_corners=False),
            "base_damage_logits": F.interpolate(base_low, size=size, mode="bilinear", align_corners=False),
            "sar_aux_logits": F.interpolate(
                self.sar_aux_head(sar_stages[-1]), size=size, mode="bilinear", align_corners=False
            ),
            "localization_probability_low": building,
            **diagnostics,
        }
        return outputs

    def forward(self, pre_rgb: torch.Tensor, post_sar: torch.Tensor, return_dict: bool = False):
        if post_sar.ndim != 4 or post_sar.shape[1] != 1:
            raise ValueError("SALT D1-D3 require native SAR [B,1,H,W]")
        size = pre_rgb.shape[-2:]
        outputs = self._decode(
            self.rgb_encoder(pre_rgb), self.sar_encoder(post_sar), size
        )
        return outputs if return_dict else (outputs["output_loc"], outputs["output_dam"])

    def forward_counterfactual(
        self,
        pre_rgb: torch.Tensor,
        correct_sar: torch.Tensor,
        wrong_sar: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Reuse the expensive RGB encoding for a correct/wrong SAR pair."""
        size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        correct = self._decode(rgb_stages, self.sar_encoder(correct_sar), size)
        wrong = self._decode(rgb_stages, self.sar_encoder(wrong_sar), size)
        return correct, wrong

    def checkpoint_contract(self) -> Dict[str, object]:
        return {
            "family": "SALT_CDE_v1",
            "variant": self.variant,
            "stable_ring": self.variant_config.stable_ring,
            "safe_ring_residual": self.variant_config.safe_ring_residual,
            "counterfactual_gate": self.variant_config.reliability_gate,
            "global_pair_gate": self.variant_config.global_pair_gate,
        }


def build_salt_damageformer(variant: str, pretrained_rgb: bool = True) -> nn.Module:
    variant = variant.lower()
    if variant == "d0":
        model = OriginalDamageFormerControl(pretrained=pretrained_rgb)
        model.variant = "d0"
        return model
    if variant == "d0n":
        model = DamageFormerRGBSARHeterogeneous(variant="h1", pretrained_rgb=pretrained_rgb)
        model.variant = "d0n"
        return model
    return DamageFormerSALT(variant, pretrained_rgb=pretrained_rgb)


def make_counterfactual_sar(sar: torch.Tensor, mode: str = "batch_shuffle") -> torch.Tensor:
    """Create a training-only wrong SAR without wrap-around leakage."""
    if mode == "zero":
        return torch.zeros_like(sar)
    if mode == "batch_shuffle" and sar.shape[0] > 1:
        return sar.roll(1, dims=0)
    if mode in ("batch_shuffle", "shift_x8"):
        amount = max(1, min(8, sar.shape[-1] // 4))
        result = torch.zeros_like(sar)
        result[..., amount:] = sar[..., :-amount]
        return result
    raise ValueError(f"unknown counterfactual mode: {mode}")


def counterfactual_transport_loss(
    correct: Mapping[str, torch.Tensor],
    wrong: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    gate_margin: float = 0.10,
    explicit_gate_bce: bool = False,
) -> Dict[str, torch.Tensor]:
    """Prefer useful correct transport and reject transport for wrong SAR."""
    gate_pairs = []
    gate_bce_terms = []
    for stage in range(1, 5):
        key = f"stage{stage}_sar_gate"
        if key in correct and key in wrong:
            mask = F.interpolate(
                ((target >= 1) & (target <= 3)).float().unsqueeze(1),
                correct[key].shape[-2:],
                mode="area",
            )
            denom = mask.sum().clamp_min(1.0)
            good = (correct[key].float() * mask).sum() / denom
            bad = (wrong[key].float() * mask).sum() / denom
            gate_pairs.append(F.relu(float(gate_margin) + bad - good))
            if explicit_gate_bce:
                good_map = F.binary_cross_entropy(
                    correct[key].float().clamp(1e-5, 1.0 - 1e-5),
                    torch.ones_like(correct[key].float()),
                    reduction="none",
                )
                bad_map = F.binary_cross_entropy(
                    wrong[key].float().clamp(1e-5, 1.0 - 1e-5),
                    torch.zeros_like(wrong[key].float()),
                    reduction="none",
                )
                gate_bce_terms.append(((good_map + bad_map) * mask).sum() / denom)
    zero = correct["output_dam"].float().sum() * 0.0
    gate_rank = torch.stack(gate_pairs).mean() if gate_pairs else zero
    gate_bce = torch.stack(gate_bce_terms).mean() if gate_bce_terms else zero
    wrong_log = F.log_softmax(wrong["output_dam"].float(), dim=1)
    wrong_base = F.softmax(wrong["base_damage_logits"].detach().float(), dim=1)
    valid = (target != int(ignore_index)).float()
    fallback_map = F.kl_div(wrong_log, wrong_base, reduction="none").sum(1)
    fallback = (fallback_map * valid).sum() / valid.sum().clamp_min(1.0)
    correct_ce = F.cross_entropy(
        correct["output_dam"].float(), target, ignore_index=ignore_index, reduction="none"
    )
    base_ce = F.cross_entropy(
        correct["base_damage_logits"].detach().float(), target, ignore_index=ignore_index, reduction="none"
    )
    utility = (F.relu(correct_ce - base_ce) * valid).sum() / valid.sum().clamp_min(1.0)
    return {
        "counterfactual_loss": gate_rank + gate_bce + fallback + utility,
        "counterfactual_gate_rank": gate_rank,
        "counterfactual_gate_bce": gate_bce,
        "counterfactual_fallback_kl": fallback,
        "transport_nonharm_loss": utility,
    }


__all__ = [
    "VALID_SALT_VARIANTS",
    "SALT_VARIANTS",
    "DamageFormerSALT",
    "StableAnchorTransportStage",
    "build_salt_damageformer",
    "counterfactual_transport_loss",
    "make_counterfactual_sar",
    "outer_ring",
    "propagate_anchor_offsets",
]
