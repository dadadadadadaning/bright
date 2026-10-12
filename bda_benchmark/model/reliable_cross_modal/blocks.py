# -*- coding: utf-8 -*-
"""Conservative RGB-SAR modules used by the S8-R0...R5 experiments.

The modules in this file deliberately separate three operations:

1. measure *within-modality* local structure and compare the relations;
2. supervise that relation map as an auxiliary change cue;
3. only in later ablations, let a bounded spatial gate modulate the main path.

This separation prevents a noisy heterogeneous difference map from being
inserted directly into the four-class damage feature space.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _shift_neighbor(
    feature: torch.Tensor,
    dy: int,
    dx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read the (y+dy,x+dx) neighbour without circular wraparound."""

    if abs(dy) > 1 or abs(dx) > 1 or (dy == 0 and dx == 0):
        raise ValueError(f"unsupported local offset {(dy, dx)}")
    height, width = feature.shape[-2:]
    padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
    shifted = padded[
        :,
        :,
        1 + dy : 1 + dy + height,
        1 + dx : 1 + dx + width,
    ]
    valid = feature.new_ones((1, 1, height, width))
    if dy < 0:
        valid[..., : -dy, :] = 0
    elif dy > 0:
        valid[..., height - dy :, :] = 0
    if dx < 0:
        valid[..., :, : -dx] = 0
    elif dx > 0:
        valid[..., :, width - dx :] = 0
    return shifted, valid


class LocalStructuralRelationDifference(nn.Module):
    """Compare eight-neighbour self-relations instead of RGB-SAR values.

    The two 1x1 projectors are initialized identically.  Consequently, before
    learning modality-specific refinements, a positive rescaling of one
    modality changes direct feature values but not the cosine relation map.
    """

    def __init__(
        self,
        in_channels: int,
        projection_channels: int = 32,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        if min(in_channels, projection_channels, out_channels) <= 0:
            raise ValueError("all channel counts must be positive")
        self.rgb_projector = nn.Conv2d(
            in_channels, projection_channels, kernel_size=1, bias=False
        )
        self.sar_projector = nn.Conv2d(
            in_channels, projection_channels, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.sar_projector.weight.copy_(self.rgb_projector.weight)
        self.relation_encoder = nn.Sequential(
            nn.Conv2d(len(_OFFSETS), out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _relations(feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = F.normalize(feature, p=2.0, dim=1, eps=1e-6)
        relations = []
        validity = []
        for dy, dx in _OFFSETS:
            neighbour, valid = _shift_neighbor(feature, dy, dx)
            cosine = (feature * neighbour).sum(dim=1, keepdim=True)
            relations.append(cosine * valid)
            validity.append(valid)
        return torch.cat(relations, dim=1), torch.cat(validity, dim=1)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb_feature.shape != sar_feature.shape:
            raise ValueError("RGB and SAR stage features must have identical shapes")
        input_dtype = rgb_feature.dtype
        # Cosine normalization and subtraction stay in FP32 under bf16/fp16 AMP.
        with torch.autocast(device_type=rgb_feature.device.type, enabled=False):
            rgb = self.rgb_projector(rgb_feature.float())
            sar = self.sar_projector(sar_feature.float())
            rgb_relation, rgb_valid = self._relations(rgb)
            sar_relation, sar_valid = self._relations(sar)
            valid = rgb_valid * sar_valid
            raw_difference = (rgb_relation - sar_relation).abs() * valid
            encoded = self.relation_encoder(raw_difference)
            valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            energy = raw_difference.sum(dim=1, keepdim=True) / valid_count
        diagnostics = {
            "relation_energy_mean": energy.detach().mean(),
            "relation_energy_max": energy.detach().amax(),
            "relation_valid_fraction": valid.detach().mean(),
        }
        return (
            encoded.to(dtype=input_dtype),
            energy.to(dtype=input_dtype),
            diagnostics,
        )


class ChangeRelationPyramid(nn.Module):
    """Stage2/Stage3 relation difference and change-attention auxiliary head."""

    def __init__(
        self,
        stage_channels: Tuple[int, int] = (192, 384),
        projection_channels: int = 32,
        relation_channels: int = 64,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.stage2 = LocalStructuralRelationDifference(
            stage_channels[0], projection_channels, relation_channels
        )
        self.stage3 = LocalStructuralRelationDifference(
            stage_channels[1], projection_channels, relation_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(relation_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.change_head = nn.Conv2d(out_channels, 2, kernel_size=1)

    def forward(
        self,
        rgb_stage2: torch.Tensor,
        sar_stage2: torch.Tensor,
        rgb_stage3: torch.Tensor,
        sar_stage3: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        relation2, energy2, stats2 = self.stage2(rgb_stage2, sar_stage2)
        relation3, energy3, stats3 = self.stage3(rgb_stage3, sar_stage3)
        relation2 = F.interpolate(
            relation2, size=target_size, mode="bilinear", align_corners=False
        )
        relation3 = F.interpolate(
            relation3, size=target_size, mode="bilinear", align_corners=False
        )
        relation_feature = self.fuse(torch.cat((relation2, relation3), dim=1))
        change_logits = self.change_head(relation_feature)
        diagnostics = {
            "relation_s2_mean": stats2["relation_energy_mean"],
            "relation_s2_max": stats2["relation_energy_max"],
            "relation_s3_mean": stats3["relation_energy_mean"],
            "relation_s3_max": stats3["relation_energy_max"],
            "relation_valid_s2": stats2["relation_valid_fraction"],
            "relation_valid_s3": stats3["relation_valid_fraction"],
            "relation_energy_s2": energy2,
            "relation_energy_s3": energy3,
        }
        return relation_feature, change_logits, diagnostics


class ResidualSARAdapter(nn.Module):
    """Near-identity SAR input adapter with a zero-initialized residual."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        return sar + self.body(sar)


class _QualityEstimator(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(16, in_channels // 8)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feature))


class BoundedSpatialReliabilityGate(nn.Module):
    """Spatially modulate the base feature by ``1 + alpha*A_change``.

    ``alpha`` is non-negative, learnable, and strictly below ``alpha_max``.
    It starts close to zero, so loading an S8 baseline does not immediately
    perturb the main segmentation path.
    """

    def __init__(
        self,
        base_channels: int = 256,
        relation_channels: int = 64,
        alpha_max: float = 0.1,
        alpha_init_logit: float = -6.0,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_max <= 0.1:
            raise ValueError("alpha_max must be in (0, 0.1]")
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init_logit)))
        self.rgb_quality = _QualityEstimator(base_channels)
        self.sar_quality = _QualityEstimator(base_channels)
        hidden = max(32, relation_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(relation_channels + 3, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        base_feature: torch.Tensor,
        relation_feature: torch.Tensor,
        building_probability: torch.Tensor,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        size = base_feature.shape[-2:]
        relation = F.interpolate(
            relation_feature, size=size, mode="bilinear", align_corners=False
        )
        prior = F.interpolate(
            building_probability, size=size, mode="bilinear", align_corners=False
        ).detach()
        rgb_quality = self.rgb_quality(rgb_feature)
        sar_quality = self.sar_quality(sar_feature)
        attention_logits = self.gate(
            torch.cat((relation, prior, rgb_quality, sar_quality), dim=1)
        )
        attention = torch.sigmoid(attention_logits)
        alpha = self.alpha()
        multiplier = 1.0 + alpha * attention
        output = base_feature * multiplier
        return output, {
            # Keep logits attached: R2+ directly supervises the exact
            # A_change that modulates the main damage feature.
            "change_attention_logits": attention_logits,
            "change_attention": attention,
            "reliability_alpha": alpha,
            "reliability_multiplier_min": multiplier.detach().amin(),
            "reliability_multiplier_max": multiplier.detach().amax(),
            "rgb_quality_mean": rgb_quality.detach().mean(),
            "sar_quality_mean": sar_quality.detach().mean(),
        }


class PixelDamageHierarchyHead(nn.Module):
    """Pixel auxiliary heads: Intact/Damage and Damaged/Destroyed."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.exist_head = nn.Conv2d(hidden_channels, 2, 1)
        self.severity_head = nn.Conv2d(hidden_channels, 2, 1)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(feature)
        return self.exist_head(shared), self.severity_head(shared)


def _identity_conv1x1(conv: nn.Conv2d) -> None:
    if conv.kernel_size != (1, 1) or conv.in_channels != conv.out_channels:
        raise ValueError("identity initialization requires a square 1x1 convolution")
    nn.init.zeros_(conv.weight)
    with torch.no_grad():
        indices = torch.arange(conv.in_channels)
        conv.weight[indices, indices, 0, 0] = 1.0
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class ConservativeStage3Rectifier(nn.Module):
    """Bounded Stage3 cross-modal correction in reliable building regions.

    The correction is a small interpolation toward a projected peer feature.
    It is strongest where the detached building prior is high and the original
    Stage3 structural relation discrepancy is low (an intact-like condition).
    """

    def __init__(
        self,
        channels: int = 384,
        hidden_channels: int = 64,
        alpha_max: float = 0.05,
        alpha_init_logit: float = -6.0,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_max <= 0.1:
            raise ValueError("alpha_max must be in (0, 0.1]")
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init_logit)))
        self.rgb_embed = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.sar_embed = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.sar_to_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.rgb_to_sar = nn.Conv2d(channels, channels, 1, bias=False)
        _identity_conv1x1(self.sar_to_rgb)
        _identity_conv1x1(self.rgb_to_sar)
        gate_in = hidden_channels * 2 + 2
        self.rgb_gate = nn.Conv2d(gate_in, 1, 1)
        self.sar_gate = nn.Conv2d(gate_in, 1, 1)
        nn.init.zeros_(self.rgb_gate.weight)
        nn.init.zeros_(self.rgb_gate.bias)
        nn.init.zeros_(self.sar_gate.weight)
        nn.init.zeros_(self.sar_gate.bias)

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        rgb_stage3: torch.Tensor,
        sar_stage3: torch.Tensor,
        building_probability: torch.Tensor,
        relation_energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb_stage3.shape != sar_stage3.shape:
            raise ValueError("Stage3 RGB/SAR feature shapes must match")
        size = rgb_stage3.shape[-2:]
        prior = F.interpolate(
            building_probability, size=size, mode="bilinear", align_corners=False
        ).detach().clamp(0.0, 1.0)
        energy = F.interpolate(
            relation_energy, size=size, mode="bilinear", align_corners=False
        ).detach()
        consistency = torch.exp(-energy.float().clamp_min(0.0)).to(rgb_stage3.dtype)
        safe_region = prior * consistency
        gate_input = torch.cat(
            (
                self.rgb_embed(rgb_stage3),
                self.sar_embed(sar_stage3),
                prior,
                consistency,
            ),
            dim=1,
        )
        rgb_gate = torch.sigmoid(self.rgb_gate(gate_input)) * safe_region
        sar_gate = torch.sigmoid(self.sar_gate(gate_input)) * safe_region
        alpha = self.alpha()
        rgb_delta = self.sar_to_rgb(sar_stage3) - rgb_stage3
        sar_delta = self.rgb_to_sar(rgb_stage3) - sar_stage3
        rgb_out = rgb_stage3 + alpha * rgb_gate * rgb_delta
        sar_out = sar_stage3 + alpha * sar_gate * sar_delta
        return rgb_out, sar_out, {
            "rectifier_alpha": alpha,
            "rectifier_safe_fraction": safe_region.detach().mean(),
            "rectifier_rgb_gate_mean": rgb_gate.detach().mean(),
            "rectifier_sar_gate_mean": sar_gate.detach().mean(),
        }


def build_hierarchical_targets(
    damage_target: torch.Tensor,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map BRIGHT labels 1/2/3 to hierarchy targets.

    Existence: 1 -> Intact(0), 2/3 -> Damage(1); background is ignored.
    Severity: 2 -> Damaged(0), 3 -> Destroyed(1); all other labels are ignored.
    """

    if damage_target.ndim == 4 and damage_target.shape[1] == 1:
        damage_target = damage_target[:, 0]
    if damage_target.ndim != 3:
        raise ValueError("damage_target must have shape [B,H,W] or [B,1,H,W]")
    exist = torch.full_like(damage_target, int(ignore_index))
    severity = torch.full_like(damage_target, int(ignore_index))
    exist[damage_target == 1] = 0
    exist[(damage_target == 2) | (damage_target == 3)] = 1
    severity[damage_target == 2] = 0
    severity[damage_target == 3] = 1
    return exist, severity


def _safe_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=logits.shape[-2:],
            mode="nearest",
        )[:, 0].long()
    valid = target != int(ignore_index)
    if not bool(valid.any().item()):
        return logits.float().sum() * 0.0
    return F.cross_entropy(logits.float(), target, ignore_index=int(ignore_index))


def reliable_auxiliary_losses(
    outputs: Dict[str, torch.Tensor],
    damage_target: torch.Tensor,
    *,
    lambda_change: float = 0.1,
    lambda_exist: float = 0.1,
    lambda_severity: float = 0.1,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """Compute only the auxiliary losses enabled by the selected variant."""

    anchor = outputs["output_dam"].float().sum() * 0.0
    exist_target, severity_target = build_hierarchical_targets(
        damage_target, ignore_index=ignore_index
    )
    change_loss = anchor
    relation_change_loss = anchor
    attention_change_loss = anchor
    exist_loss = anchor
    severity_loss = anchor
    if "change_logits_feature" in outputs:
        relation_change_loss = _safe_cross_entropy(
            outputs["change_logits_feature"], exist_target, ignore_index
        )
    if "change_attention_logits" in outputs:
        attention_change_loss = _safe_cross_entropy(
            torch.cat(
                (
                    torch.zeros_like(outputs["change_attention_logits"]),
                    outputs["change_attention_logits"],
                ),
                dim=1,
            ),
            exist_target,
            ignore_index,
        )
    active_change_losses = []
    if "change_logits_feature" in outputs:
        active_change_losses.append(relation_change_loss)
    if "change_attention_logits" in outputs:
        active_change_losses.append(attention_change_loss)
    if active_change_losses:
        change_loss = torch.stack(active_change_losses).mean()
    if "pixel_exist_logits_feature" in outputs:
        exist_loss = _safe_cross_entropy(
            outputs["pixel_exist_logits_feature"], exist_target, ignore_index
        )
    if "pixel_severity_logits_feature" in outputs:
        severity_loss = _safe_cross_entropy(
            outputs["pixel_severity_logits_feature"], severity_target, ignore_index
        )
    total = (
        float(lambda_change) * change_loss
        + float(lambda_exist) * exist_loss
        + float(lambda_severity) * severity_loss
    )
    return {
        "loss_aux_total": total,
        "loss_change": change_loss,
        "loss_change_relation": relation_change_loss,
        "loss_change_attention": attention_change_loss,
        "loss_pixel_exist": exist_loss,
        "loss_pixel_severity": severity_loss,
    }


__all__: Iterable[str] = (
    "BoundedSpatialReliabilityGate",
    "ChangeRelationPyramid",
    "ConservativeStage3Rectifier",
    "LocalStructuralRelationDifference",
    "PixelDamageHierarchyHead",
    "ResidualSARAdapter",
    "build_hierarchical_targets",
    "reliable_auxiliary_losses",
)
