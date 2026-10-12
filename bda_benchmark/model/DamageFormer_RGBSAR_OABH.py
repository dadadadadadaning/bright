# -*- coding: utf-8 -*-
"""OA-HSDNet: optical-anchored hierarchical severity network.

This is a native RGB/SAR predictor, not a P3 logit adapter. Geometry is read
from RGB only; Stage2/3 evidence predicts Affected|Building; Stage3/4 evidence
predicts Destroyed|Affected. Controlled variants add shared attention,
task-specific attention, occurrence-conditioned severity, then a soft damage
evidence field. There is no router, FCA, flow, Conv3D or logit residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H4 import hierarchical_probabilities
from .DamageFormer_RGBSAR_H6_SDCR import P3RGBSpatialAnchor
from .DamageFormer_RGBSAR_Heterogeneous import (
    ConvGNAct, NativeSARPyramid, STAGE_CHANNELS, SwinRGBPyramid,
)


VALID_OABH_VARIANTS = ("n0", "n1_shared", "n1_stage", "n1_cond", "n2")
VALID_SAR_STYLE_MODES = ("none", "mixstyle", "adapter")


@dataclass(frozen=True)
class OABHVariant:
    name: str
    attention: str
    occurrence_conditioned_severity: bool
    soft_damage_evidence: bool


OABH_VARIANTS: Mapping[str, OABHVariant] = {
    "n0": OABHVariant("N0-Asymmetric-Scale-Baseline", "none", False, False),
    "n1_shared": OABHVariant("N1-Shared-CrossAttention", "shared", False, False),
    "n1_stage": OABHVariant("N1-Task-Specific-CrossAttention", "stage", False, False),
    "n1_cond": OABHVariant("N1-Occurrence-Conditioned-Severity", "stage", True, False),
    "n2": OABHVariant("N2-Soft-Damage-Extent", "stage", True, True),
}


def _cross_event_permutation(
    event_ids: Sequence[str], device: torch.device
) -> torch.Tensor | None:
    """Pair samples with another event without consulting labels or metrics."""

    events = tuple(str(event) for event in event_ids)
    count = len(events)
    if count < 2 or len(set(events)) < 2:
        return None
    groups: Dict[str, list[int]] = {}
    for index, event in enumerate(events):
        groups.setdefault(event, []).append(index)
    maximum = max(len(indices) for indices in groups.values())
    if maximum * 2 > count:
        return None
    # Placing each event contiguously and rotating by the largest group size
    # constructs a strict cross-event pairing whenever one is possible.
    ordered = [index for event in sorted(groups) for index in groups[event]]
    paired = ordered[maximum:] + ordered[:maximum]
    permutation = [0] * count
    for source, target in zip(ordered, paired):
        permutation[source] = target
    if any(events[index] == events[target] for index, target in enumerate(permutation)):
        return None
    return torch.tensor(permutation, dtype=torch.long, device=device)


class CrossEventSARStyleMix(nn.Module):
    """Train-only MixStyle restricted to SAR and different disaster events."""

    def __init__(self, probability: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("MixStyle probability must be within [0, 1]")
        if alpha <= 0.0:
            raise ValueError("MixStyle alpha must be positive")
        self.probability = float(probability)
        self.alpha = float(alpha)

    def forward(
        self, feature: torch.Tensor, event_ids: Sequence[str] | None = None
    ) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return feature
        if event_ids is None or len(event_ids) != feature.shape[0]:
            return feature
        if torch.rand((), device=feature.device) >= self.probability:
            return feature
        permutation = _cross_event_permutation(event_ids, feature.device)
        if permutation is None:
            return feature
        value = feature.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        variance = value.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = (variance + 1e-6).sqrt()
        normalized = (value - mean) / std
        concentration = torch.tensor(
            self.alpha, device=feature.device, dtype=torch.float32
        )
        mixing = torch.distributions.Beta(concentration, concentration).sample(
            (feature.shape[0], 1, 1, 1)
        )
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_std = mixing * std + (1.0 - mixing) * std[permutation]
        return (normalized * mixed_std + mixed_mean).to(feature.dtype)


class SAREventStyleSeparation(nn.Module):
    """Zero-gated SAR-internal content/style adapter with a live first gradient."""

    def __init__(self, channels: int, hidden_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(16, channels // hidden_ratio)
        self.stat_gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        groups = max(1, min(16, channels // 4))
        self.candidate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.transition = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        value = feature.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = (value.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
        content = (value - mean) / std
        gate = self.stat_gate(torch.cat([mean, std], dim=1))
        style = gate * (value - content)
        candidate = self.candidate(torch.cat([content, style], dim=1))
        output = value + torch.tanh(self.transition) * candidate
        return output.to(feature.dtype)


class EventRobustNativeSARPyramid(NativeSARPyramid):
    """Transform Stage1/2 before deeper SAR stages consume their features."""

    def __init__(
        self,
        style_mode: str = "none",
        mix_probability: float = 0.5,
        mix_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        style_mode = str(style_mode).lower()
        if style_mode not in VALID_SAR_STYLE_MODES:
            raise ValueError(f"SAR style mode must be one of {VALID_SAR_STYLE_MODES}")
        self.style_mode = style_mode
        if style_mode == "mixstyle":
            self.style_blocks = nn.ModuleList(
                CrossEventSARStyleMix(mix_probability, mix_alpha) for _ in range(2)
            )
        elif style_mode == "adapter":
            self.style_blocks = nn.ModuleList(
                SAREventStyleSeparation(STAGE_CHANNELS[index]) for index in range(2)
            )
        else:
            self.style_blocks = nn.ModuleList((nn.Identity(), nn.Identity()))
        self._batch_event_ids: Tuple[str, ...] | None = None

    def set_batch_events(self, event_ids: Sequence[str] | None) -> None:
        self._batch_event_ids = (
            None if event_ids is None else tuple(str(event) for event in event_ids)
        )

    def forward(self, sar: torch.Tensor):
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError(f"native SAR encoder expects [B,1,H,W], got {tuple(sar.shape)}")
        x = self.stem(sar)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < 2:
                block = self.style_blocks[index]
                x = (
                    block(x, self._batch_event_ids)
                    if isinstance(block, CrossEventSARStyleMix)
                    else block(x)
                )
            outputs.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return outputs


class WindowCrossAttention2d(nn.Module):
    """Use local RGB queries to read native-SAR keys/values."""

    def __init__(self, channels: int, heads: int = 4, window_size: int = 5) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.query_norm = nn.LayerNorm(channels)
        self.sar_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def _windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        batch, channels, height, width = x.shape
        size = self.window_size
        pad_h, pad_w = (-height) % size, (-width) % size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[-2:]
        x = x.reshape(batch, channels, padded_h // size, size, padded_w // size, size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(-1, size * size, channels), (
            batch, height, width, padded_h, padded_w
        )

    def _restore(self, x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        batch, height, width, padded_h, padded_w = shape
        size = self.window_size
        x = x.reshape(batch, padded_h // size, padded_w // size, size, size, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(batch, self.channels, padded_h, padded_w)[..., :height, :width]

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if rgb.shape != sar.shape:
            raise ValueError("cross-attention requires equal projected shapes")
        q, shape = self._windows(rgb)
        kv, sar_shape = self._windows(sar)
        if shape != sar_shape:
            raise RuntimeError("RGB/SAR window partitions diverged")
        readout, _ = self.attention(
            self.query_norm(q), self.sar_norm(kv), self.sar_norm(kv),
            need_weights=False,
        )
        return self._restore(self.output(readout), shape)


class ModalitySpecificScalePath(nn.Module):
    """Aggregate selected scales with independent RGB and SAR parameters."""

    def __init__(self, stage_indices: Sequence[int], channels: int = 128) -> None:
        super().__init__()
        self.stage_indices = tuple(int(index) for index in stage_indices)
        self.rgb_projections = nn.ModuleList(
            ConvGNAct(STAGE_CHANNELS[index], channels, 1) for index in self.stage_indices
        )
        self.sar_projections = nn.ModuleList(
            ConvGNAct(STAGE_CHANNELS[index], channels, 1) for index in self.stage_indices
        )
        merged = channels * len(self.stage_indices)
        self.rgb_merge = ConvGNAct(merged, channels, 3)
        self.sar_merge = ConvGNAct(merged, channels, 3)

    def _project(self, stages, modules: nn.ModuleList) -> torch.Tensor:
        values = [module(stages[index]) for index, module in zip(self.stage_indices, modules)]
        target = values[0].shape[-2:]
        values = [
            value if value.shape[-2:] == target else F.interpolate(
                value, target, mode="bilinear", align_corners=False
            ) for value in values
        ]
        return torch.cat(values, dim=1)

    def forward(self, rgb_stages, sar_stages) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = self.rgb_merge(self._project(rgb_stages, self.rgb_projections))
        sar = self.sar_merge(self._project(sar_stages, self.sar_projections))
        return rgb, sar


class ExplicitHeterogeneousRelation(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            ConvGNAct(4 * channels, 2 * channels, 1),
            ConvGNAct(2 * channels, channels, 3),
            ConvGNAct(channels, channels, 3),
        )

    def forward(self, rgb, sar, attended_sar=None) -> torch.Tensor:
        sar_readout = sar if attended_sar is None else sar + attended_sar
        return self.fusion(torch.cat([
            rgb, sar_readout, (rgb - sar_readout).abs(), rgb * sar_readout
        ], dim=1))


class OccurrenceConditionedSeverity(nn.Module):
    """Make destruction representation explicitly consume occurrence evidence."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.occurrence_projection = ConvGNAct(channels, channels, 1)
        self.condition = nn.Sequential(
            ConvGNAct(2 * channels, channels, 3), ConvGNAct(channels, channels, 3)
        )

    def forward(self, destruction, occurrence) -> torch.Tensor:
        occurrence = self.occurrence_projection(occurrence)
        occurrence = F.interpolate(
            occurrence, destruction.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.condition(torch.cat([destruction, occurrence], dim=1))


class SoftDamageEvidenceConditioner(nn.Module):
    """Dense evidence and normalized soft coverage, without hard regions."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.evidence_head = nn.Sequential(ConvGNAct(channels, 64, 3), nn.Conv2d(64, 1, 1))
        self.condition = nn.Sequential(
            ConvGNAct(2 * channels + 4, channels, 3), ConvGNAct(channels, channels, 3)
        )

    @staticmethod
    def _coverage(evidence, support, kernel: int) -> torch.Tensor:
        numerator = F.avg_pool2d(evidence * support, kernel, 1, kernel // 2)
        denominator = F.avg_pool2d(support, kernel, 1, kernel // 2).clamp_min(1e-4)
        return numerator / denominator

    def forward(self, destruction, occurrence, building_probability):
        support = F.interpolate(
            building_probability.detach(), occurrence.shape[-2:],
            mode="bilinear", align_corners=False,
        ).clamp(0.0, 1.0)
        evidence_logits = self.evidence_head(occurrence)
        evidence = evidence_logits.sigmoid() * support
        coverages = torch.cat([
            self._coverage(evidence, support, kernel) for kernel in (3, 7, 15)
        ], dim=1)
        occurrence = F.interpolate(
            occurrence, destruction.shape[-2:], mode="bilinear", align_corners=False
        )
        extent = F.interpolate(
            torch.cat([evidence, coverages], dim=1), destruction.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        return self.condition(torch.cat([destruction, occurrence, extent], dim=1)), evidence_logits, coverages


class OpticalAnchoredBuildingHierarchy(nn.Module):
    OCCURRENCE_STAGES = (1, 2)  # Stage2 + Stage3
    DESTRUCTION_STAGES = (2, 3)  # Stage3 + Stage4

    def __init__(
        self,
        variant="n0",
        pretrained_rgb=True,
        latent_channels=128,
        sar_style_mode="none",
        sar_mix_probability=0.5,
        sar_mix_alpha=0.1,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in VALID_OABH_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_OABH_VARIANTS}")
        self.variant = variant
        self.experiment_name = "OA_HSDNET"
        self.resume_protocol = "OA_HSDNET_resume_v2"
        self.best_weights_protocol = "OA_HSDNET_best_weights_v2"
        self.variant_config = OABH_VARIANTS[variant]
        self.rgb_encoder = SwinRGBPyramid(pretrained=pretrained_rgb)
        self.sar_style_mode = str(sar_style_mode).lower()
        self.sar_encoder = EventRobustNativeSARPyramid(
            self.sar_style_mode, sar_mix_probability, sar_mix_alpha
        )
        self.localization_anchor = P3RGBSpatialAnchor()

        self.occurrence_path = ModalitySpecificScalePath(self.OCCURRENCE_STAGES, latent_channels)
        self.destruction_path = ModalitySpecificScalePath(self.DESTRUCTION_STAGES, latent_channels)
        self.occurrence_relation = ExplicitHeterogeneousRelation(latent_channels)
        self.destruction_relation = ExplicitHeterogeneousRelation(latent_channels)
        attention = self.variant_config.attention
        self.shared_attention = WindowCrossAttention2d(latent_channels) if attention == "shared" else None
        self.occurrence_attention = WindowCrossAttention2d(latent_channels) if attention == "stage" else None
        self.destruction_attention = WindowCrossAttention2d(latent_channels) if attention == "stage" else None
        self.severity_conditioner = (
            OccurrenceConditionedSeverity(latent_channels)
            if self.variant_config.occurrence_conditioned_severity else None
        )
        self.evidence_conditioner = (
            SoftDamageEvidenceConditioner(latent_channels)
            if self.variant_config.soft_damage_evidence else None
        )
        self.occurrence_head = nn.Sequential(ConvGNAct(latent_channels, 96, 3), nn.Conv2d(96, 2, 1))
        self.severity_head = nn.Sequential(ConvGNAct(latent_channels, 96, 3), nn.Conv2d(96, 2, 1))
        self.training_phase = "joint"

    def set_batch_events(self, event_ids: Sequence[str] | None) -> None:
        self.sar_encoder.set_batch_events(event_ids)

    def checkpoint_contract(self) -> Dict[str, object]:
        first = self.sar_encoder.style_blocks[0]
        return {
            "sar_style_mode": self.sar_style_mode,
            "sar_mix_probability": getattr(first, "probability", None),
            "sar_mix_alpha": getattr(first, "alpha", None),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.rgb_encoder.eval()
        self.localization_anchor.eval()
        return self

    def set_training_phase(self, phase: str) -> None:
        phase = phase.lower()
        if phase not in ("heads", "deep", "joint"):
            raise ValueError("phase must be heads, deep, or joint")
        self.training_phase = phase
        for parameter in self.parameters():
            parameter.requires_grad = True
        for module in (self.rgb_encoder, self.localization_anchor):
            for parameter in module.parameters():
                parameter.requires_grad = False
        if phase == "joint":
            for parameter in self.sar_encoder.stem.parameters():
                parameter.requires_grad = False
            for parameter in self.sar_encoder.stages[0].parameters():
                parameter.requires_grad = False
            return
        for parameter in self.sar_encoder.parameters():
            parameter.requires_grad = False
        if phase == "deep":
            for stage in self.sar_encoder.stages[2:]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
            for parameter in self.sar_encoder.downsamples[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.sar_encoder.style_blocks.parameters():
                parameter.requires_grad = True

    def _read(self, branch, rgb, sar):
        if self.shared_attention is not None:
            return self.shared_attention(rgb, sar)
        module = self.occurrence_attention if branch == "occurrence" else self.destruction_attention
        return None if module is None else module(rgb, sar)

    def forward(self, pre_rgb, post_sar, return_dict=False):
        if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
            raise ValueError("OA-HSDNet expects one native SAR channel plus audit view")
        post_sar = post_sar[:, :1]
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)
        _, building_low = self.localization_anchor(rgb_stages)
        building_probability = building_low.softmax(1)[:, 1:2]

        r_occ, s_occ = self.occurrence_path(rgb_stages, sar_stages)
        occurrence_feature = self.occurrence_relation(
            r_occ, s_occ, self._read("occurrence", r_occ, s_occ)
        )
        r_des, s_des = self.destruction_path(rgb_stages, sar_stages)
        destruction_feature = self.destruction_relation(
            r_des, s_des, self._read("destruction", r_des, s_des)
        )
        if self.severity_conditioner is not None:
            destruction_feature = self.severity_conditioner(destruction_feature, occurrence_feature)
        evidence_outputs: Dict[str, torch.Tensor] = {}
        if self.evidence_conditioner is not None:
            destruction_feature, evidence_low, coverage_low = self.evidence_conditioner(
                destruction_feature, occurrence_feature, building_probability
            )
            evidence_outputs = {
                "damage_evidence_logits": F.interpolate(
                    evidence_low, input_size, mode="bilinear", align_corners=False
                ),
                "damage_evidence_coverage": coverage_low,
            }

        hierarchy_size = building_low.shape[-2:]
        occurrence_low = F.interpolate(
            self.occurrence_head(occurrence_feature), hierarchy_size,
            mode="bilinear", align_corners=False,
        )
        severity_low = F.interpolate(
            self.severity_head(destruction_feature), hierarchy_size,
            mode="bilinear", align_corners=False,
        )
        probabilities_low = hierarchical_probabilities(building_low, occurrence_low, severity_low)
        output_loc = F.interpolate(building_low, input_size, mode="bilinear", align_corners=False)
        output_dam = F.interpolate(
            probabilities_low.clamp_min(1e-7).log(), input_size,
            mode="bilinear", align_corners=False,
        )
        outputs = {
            "output_loc": output_loc,
            "output_dam": output_dam,
            "hierarchy_building_logits": output_loc,
            "hierarchy_affected_logits": F.interpolate(
                occurrence_low, input_size, mode="bilinear", align_corners=False
            ),
            "hierarchy_destroyed_logits": F.interpolate(
                severity_low, input_size, mode="bilinear", align_corners=False
            ),
            "occurrence_feature": occurrence_feature,
            "destruction_feature": destruction_feature,
            **evidence_outputs,
        }
        return outputs if return_dict else (output_loc, output_dam)


def _checkpoint_state(path: str | Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("initialisation checkpoint does not contain a state mapping")
    return state


def load_h6_encoder_initialisation(model, path: str | Path) -> Dict[str, int]:
    state = _checkpoint_state(path)
    report: Dict[str, int] = {}
    for module_name, source_name in (
        ("rgb_encoder", "rgb_encoder"),
        ("sar_encoder", "sar_encoder"),
        ("localization_anchor", "p3_spatial_anchor"),
    ):
        module = getattr(model, module_name)
        prefix = source_name + "."
        source = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        target = module.state_dict()
        compatible = {
            key: value for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        missing, unexpected = module.load_state_dict(compatible, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected {module_name} tensors: {unexpected}")
        report[f"{module_name}_loaded"] = len(compatible)
        report[f"{module_name}_missing"] = len(missing)
        if not compatible and module_name in ("rgb_encoder", "sar_encoder"):
            raise RuntimeError(f"H6 checkpoint supplied no compatible {module_name}")
    return report


def load_full_oabh_initialisation(model, path: str | Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_variant = str(checkpoint.get("variant", "")).lower()
    if checkpoint_variant and checkpoint_variant != model.variant:
        raise RuntimeError(f"OA-HSDNet initialisation {checkpoint_variant} != {model.variant}")
    state = checkpoint["model"]
    if getattr(model, "sar_style_mode", "none") == "adapter":
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = "sar_encoder.style_blocks."
        illegal_missing = [key for key in missing if not key.startswith(allowed)]
        if illegal_missing or unexpected:
            raise RuntimeError(
                "unsafe OA-HSDNet adapter transfer: "
                f"missing={illegal_missing}, unexpected={unexpected}"
            )
    else:
        model.load_state_dict(state, strict=True)


def build_oabh_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
    sar_style_mode: str = "none",
    sar_mix_probability: float = 0.5,
    sar_mix_alpha: float = 0.1,
):
    return OpticalAnchoredBuildingHierarchy(
        variant,
        pretrained_rgb,
        sar_style_mode=sar_style_mode,
        sar_mix_probability=sar_mix_probability,
        sar_mix_alpha=sar_mix_alpha,
    )


__all__ = [
    "VALID_OABH_VARIANTS", "VALID_SAR_STYLE_MODES", "OABH_VARIANTS",
    "CrossEventSARStyleMix", "SAREventStyleSeparation",
    "EventRobustNativeSARPyramid", "WindowCrossAttention2d",
    "ModalitySpecificScalePath", "ExplicitHeterogeneousRelation",
    "OccurrenceConditionedSeverity", "SoftDamageEvidenceConditioner",
    "OpticalAnchoredBuildingHierarchy", "load_h6_encoder_initialisation",
    "load_full_oabh_initialisation", "build_oabh_damageformer",
]
