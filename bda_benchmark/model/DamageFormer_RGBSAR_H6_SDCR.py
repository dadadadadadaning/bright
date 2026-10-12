# -*- coding: utf-8 -*-
"""H6-SDCR: sensor/damage disentanglement for heterogeneous RGB--SAR.

This branch is intentionally independent from the H5 instance experiments.
It starts from the validated H4-A4 pixel model and tests three architectural
hypotheses without adding a new supervised loss:

* A0: exact H4-A4 checkpoint control;
* A1: the validated P3 RGB encoder/localisation anchor is transplanted as
  weights (there is no teacher and no distillation);
* A2: image-conditioned sensor-gap prototypes are estimated from high-
  confidence unchanged pixels and removed before damage evidence is injected;
* A3: A2 plus low-frequency nuisance suppression, restricted to deep stages.

All new damage adapters use zero-initialised transitions.  A2/A3 therefore
start with exactly the A1 prediction after checkpoint migration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H4 import (
    DamageFormerRGBSARH4,
    ReliabilityGatedCommonUniqueStage,
    hierarchical_probabilities,
)
from .DamageFormer_RGBSAR_Heterogeneous import ConvGNAct
from .DamageFormer_S3_BGMP_K3 import ResBlock
from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


VALID_H6_VARIANTS = (
    "a0", "a1", "a2", "a3", "b0", "b1", "b2", "b1r",
)


@dataclass(frozen=True)
class H6Variant:
    name: str
    p3_rgb_anchor: bool
    sensor_gap_prototype: bool
    frequency_calibration: bool
    sparse_fusion: bool = False
    sensor_stages: Tuple[str, ...] = ()
    safe_p3_residual: bool = False
    class_sparse_router: bool = False


H6_VARIANTS: Mapping[str, H6Variant] = {
    "a0": H6Variant("H6-A0-H4A4-Control", False, False, False),
    "a1": H6Variant("H6-A1-P3-RGB-Spatial-Anchor", True, False, False),
    "a2": H6Variant(
        "H6-A2-Event-Conditioned-Sensor-Gap",
        True, True, False, sensor_stages=("stage3", "stage4"),
    ),
    "a3": H6Variant(
        "H6-A3-Frequency-Selective-SDCR",
        True, True, True, sensor_stages=("stage3", "stage4"),
    ),
    "b0": H6Variant(
        "H6-B0-Exact-P3-Safety-Control",
        True, False, False, safe_p3_residual=True,
    ),
    "b1": H6Variant(
        "H6-B1-P3-Safe-Stage4-NativeSAR-Residual",
        True, True, False, sensor_stages=("stage4",), safe_p3_residual=True,
    ),
    "b2": H6Variant(
        "H6-B2-P3-Safe-Stage34-NativeSAR-Residual",
        True, True, False,
        sensor_stages=("stage3", "stage4"), safe_p3_residual=True,
    ),
    "b1r": H6Variant(
        "H6-B1R-Class-Conditional-Sparse-Stage4-Residual",
        True,
        True,
        False,
        sensor_stages=("stage4",),
        safe_p3_residual=True,
        class_sparse_router=True,
    ),
}


class SafeClassResidualRouter(nn.Module):
    """Inject native-SAR evidence without changing the P3 initial output.

    The spatial proposal is deliberately non-zero, while the three
    building-class transitions start at exactly zero.  This gives the
    transitions a first-step gradient and guarantees exact P3 logits at
    initialisation.  Background has no transition and localization is never
    routed through this module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_predictor = nn.Sequential(
            ConvGNAct(8, 16, 3),
            nn.Conv2d(16, 1, 1, bias=True),
        )
        nn.init.kaiming_normal_(
            self.gate_predictor[-1].weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.gate_predictor[-1].bias)
        self.class_transition = nn.Parameter(torch.zeros(3))

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
        native_probability = native_logits.float().softmax(1).to(native_logits.dtype)
        confidence = torch.sigmoid(
            self.gate_predictor(
                torch.cat([p3_probability, native_probability], dim=1)
            )
        )
        building_transition = torch.tanh(self.class_transition)
        transition = torch.cat(
            [building_transition.new_zeros(1), building_transition], dim=0
        ).reshape(1, 4, 1, 1)
        delta = native_logits - p3_logits
        routed_delta = confidence * transition * delta
        return p3_logits + routed_delta, {
            "safe_residual_sar_gate": confidence,
            "safe_class_transition": transition,
            "safe_residual_delta_abs": routed_delta.abs().mean(1, keepdim=True),
            "native_p3_delta_abs": delta.abs().mean(1, keepdim=True),
        }


class ClassConditionalSparseResidualRouter(nn.Module):
    """Conservatively route native-SAR evidence into three building classes.

    Unlike B1's shared spatial gate, B1R predicts an independent gate for
    intact, damaged, and destroyed.  The background route is identically zero.
    Gate logits start at -4 (about 1.8% open probability) and the signed class
    transition is bounded to 10%, so the native expert cannot overwrite the
    frozen P3 decision surface.  Zero transitions preserve exact P3 logits at
    initialisation while retaining a non-zero first-step gradient.
    """

    def __init__(
        self,
        gate_bias: float = -4.0,
        maximum_transition: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < maximum_transition <= 1.0:
            raise ValueError("maximum_transition must be in (0, 1]")
        self.gate_predictor = nn.Sequential(
            ConvGNAct(8, 16, 3),
            nn.Conv2d(16, 3, 1, bias=True),
        )
        nn.init.zeros_(self.gate_predictor[-1].weight)
        nn.init.constant_(self.gate_predictor[-1].bias, float(gate_bias))
        self.class_transition = nn.Parameter(torch.zeros(3))
        self.maximum_transition = float(maximum_transition)

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
        native_probability = native_logits.float().softmax(1).to(native_logits.dtype)
        building_support = (1.0 - p3_probability[:, :1]).detach()
        building_gates = torch.sigmoid(
            self.gate_predictor(
                torch.cat([p3_probability, native_probability], dim=1)
            )
        ) * building_support
        gates = torch.cat([torch.zeros_like(building_support), building_gates], dim=1)

        building_transition = self.maximum_transition * torch.tanh(
            self.class_transition
        )
        transition = torch.cat(
            [building_transition.new_zeros(1), building_transition], dim=0
        ).reshape(1, 4, 1, 1)
        delta = native_logits - p3_logits
        routed_delta = gates * transition * delta
        return p3_logits + routed_delta, {
            "safe_residual_sar_gate": gates,
            "safe_class_transition": transition,
            "safe_residual_delta_abs": routed_delta.abs().mean(1, keepdim=True),
            "native_p3_delta_abs": delta.abs().mean(1, keepdim=True),
        }


class P3RGBSpatialAnchor(nn.Module):
    """The exact P3 RGB multi-scale fusion and localisation head.

    The Swin encoder is the model's existing ``rgb_encoder``.  Its weights are
    overwritten by P3 during migration; this module owns only P3's
    ``fusion_layer_1`` and ``clf_1`` equivalents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fusion = ResBlock(
            96 + 192 + 384 + 768,
            256,
            downsample=nn.Conv2d(96 + 192 + 384 + 768, 256, 1),
        )
        self.head = nn.Conv2d(256, 2, 1)

    @staticmethod
    def multiscale(stages: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(stages) != 4:
            raise ValueError("P3 RGB anchor requires four Swin stages")
        target = stages[0].shape[-2:]
        deep_to_shallow = [
            F.interpolate(value, target, mode="bilinear", align_corners=False)
            for value in reversed(stages[1:])
        ]
        return torch.cat([*deep_to_shallow, stages[0]], dim=1)

    def forward(
        self, stages: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.fusion(self.multiscale(stages))
        return feature, self.head(feature)


class SensorGapPrototypeAdapter(nn.Module):
    """Remove image-conditioned sensor nuisance before residual injection.

    For every image and channel the adapter estimates

        p_sensor = weighted_mean(f_sar - f_rgb | likely unchanged)

    and represents damage evidence as ``(f_sar - f_rgb) - p_sensor``.  The
    unchanged weight combines RGB building localisation with the SAR auxiliary
    intact probability.  It is detached, so the adapter cannot improve its
    objective by corrupting either confidence source.
    """

    def __init__(
        self,
        latent_channels: int,
        frequency_calibration: bool = False,
        lowpass_kernel: int = 5,
    ) -> None:
        super().__init__()
        if lowpass_kernel < 3 or lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be an odd integer >= 3")
        self.frequency_calibration = bool(frequency_calibration)
        self.lowpass_kernel = int(lowpass_kernel)
        self.evidence = nn.Sequential(
            ConvGNAct(4 * latent_channels, latent_channels, 3),
            nn.Conv2d(latent_channels, latent_channels, 1, bias=True),
        )
        # Keep the candidate non-zero and initialise only its scalar transition
        # at zero.  Zeroing both would make every first-step gradient zero.
        nn.init.kaiming_normal_(
            self.evidence[-1].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.evidence[-1].bias)
        self.causal_transition = nn.Parameter(torch.tensor(0.0))
        if self.frequency_calibration:
            self.frequency_gate = nn.Sequential(
                ConvGNAct(2, 8, 3),
                nn.Conv2d(8, 1, 1, bias=True),
            )
            nn.init.zeros_(self.frequency_gate[-1].weight)
            nn.init.zeros_(self.frequency_gate[-1].bias)
            self.frequency_transition = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _resize(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        joint: torch.Tensor,
        rgb_common: torch.Tensor,
        sar_common: torch.Tensor,
        rgb_unique: torch.Tensor,
        sar_unique: torch.Tensor,
        building_probability: torch.Tensor,
        sar_intact_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        building = self._resize(building_probability, rgb_common)
        intact = self._resize(sar_intact_probability, rgb_common)
        unchanged = (
            (1.0 - building) + building * intact
        ).detach().float().clamp(0.0, 1.0)

        gap = sar_common - rgb_common
        weights = unchanged.to(gap.dtype)
        mass = weights.sum((2, 3), keepdim=True).clamp_min(1.0)
        prototype = (gap * weights).sum((2, 3), keepdim=True) / mass
        corrected = gap - prototype

        low_frequency = F.avg_pool2d(
            corrected,
            kernel_size=self.lowpass_kernel,
            stride=1,
            padding=self.lowpass_kernel // 2,
        )
        frequency_gate = torch.zeros_like(unchanged)
        frequency_transition = corrected.new_zeros(())
        calibrated = corrected
        if self.frequency_calibration:
            frequency_gate = torch.sigmoid(
                self.frequency_gate(
                    torch.cat(
                        [low_frequency.abs().mean(1, keepdim=True), unchanged],
                        dim=1,
                    )
                )
            )
            frequency_transition = torch.tanh(self.frequency_transition)
            calibrated = corrected - (
                frequency_transition * frequency_gate * low_frequency
            )

        residual = self.evidence(
            torch.cat(
                [calibrated, calibrated.abs(), rgb_unique, sar_unique],
                dim=1,
            )
        )
        causal_transition = torch.tanh(self.causal_transition)
        output = joint + causal_transition * residual
        diagnostics = {
            "sensor_gap_prototype": prototype,
            "sensor_gap_raw_abs": gap.abs().mean(1, keepdim=True),
            "sensor_gap_corrected_abs": corrected.abs().mean(1, keepdim=True),
            "unchanged_confidence": unchanged.to(gap.dtype),
            "unchanged_mass": mass,
            "causal_transition": causal_transition.reshape(1, 1, 1, 1),
            "low_frequency_abs": low_frequency.abs().mean(1, keepdim=True),
            "frequency_gate": frequency_gate.to(gap.dtype),
            "frequency_transition": frequency_transition.reshape(1, 1, 1, 1),
        }
        return output, diagnostics


class DamageFormerRGBSARH6SDCR(DamageFormerRGBSARH4):
    """H6 model retaining the H4-A4 state dictionary as its safe control."""

    def __init__(
        self,
        variant: str = "a1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
    ) -> None:
        variant = variant.lower()
        if variant not in VALID_H6_VARIANTS:
            raise ValueError(f"unknown H6 variant {variant!r}")
        super().__init__(
            variant="a4",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h6_config = H6_VARIANTS[variant]
        if self.h6_config.p3_rgb_anchor:
            self.p3_spatial_anchor = P3RGBSpatialAnchor()
        if self.h6_config.safe_p3_residual:
            self.p3_full_anchor = DamageFormer_S8_MICB_DSMIL()
            self.safe_residual_router = (
                ClassConditionalSparseResidualRouter()
                if self.h6_config.class_sparse_router
                else SafeClassResidualRouter()
            )
            # HeterogeneousRGBSARDataset supplies a clipped one-channel view
            # normalised by these frozen train-split statistics.  Inverting
            # that affine transform and applying ImageNet channel statistics
            # reconstructs the exact pseudo-RGB view used by P3.
            self.register_buffer(
                "native_sar_mean",
                torch.tensor(56.64234075291341, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "native_sar_std",
                torch.tensor(45.34399136641941, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "imagenet_mean",
                torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
                .reshape(1, 3, 1, 1),
                persistent=True,
            )
            self.register_buffer(
                "imagenet_std",
                torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
                .reshape(1, 3, 1, 1),
                persistent=True,
            )
        if self.h6_config.sensor_gap_prototype:
            self.sensor_damage_adapters = nn.ModuleDict(
                {
                    stage_name: SensorGapPrototypeAdapter(
                        latent_channels[int(stage_name[-1]) - 1],
                        frequency_calibration=self.h6_config.frequency_calibration,
                    )
                    for stage_name in self.h6_config.sensor_stages
                }
            )

    def _p3_pseudo_rgb_sar(self, post_sar: torch.Tensor) -> torch.Tensor:
        if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
            raise ValueError("H6-B requires SAR shaped [B,1,H,W] or [B,2,H,W]")
        # The optional second channel is the un-clipped display-value audit view.
        # Falling back to channel 0 keeps old checkpoints readable, but the
        # strict B0 guard requires the two-channel H4 dataset contract.
        if post_sar.shape[1] == 2:
            display = post_sar[:, 1:2].float()
        else:
            display = post_sar.float() * self.native_sar_std + self.native_sar_mean
        repeated = display.repeat(1, 3, 1, 1)
        pseudo_rgb = (repeated - self.imagenet_mean) / self.imagenet_std
        return pseudo_rgb.to(post_sar.dtype)

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        if self.variant == "a0":
            return super().forward(pre_rgb, post_sar, return_dict=return_dict)

        native_sar = post_sar[:, :1]

        p3_localization_logits = None
        p3_damage_logits = None
        if self.h6_config.safe_p3_residual:
            # This is a frozen inference anchor, not a teacher and not a
            # distillation target.  It remains part of the final forward path.
            with torch.no_grad():
                p3_localization_logits, p3_damage_logits = self.p3_full_anchor(
                    pre_rgb,
                    self._p3_pseudo_rgb_sar(post_sar),
                )
            if self.variant == "b0":
                outputs = {
                    "output_loc": p3_localization_logits,
                    "output_dam": p3_damage_logits,
                    "safe_residual_sar_gate": p3_damage_logits.new_zeros(
                        (p3_damage_logits.shape[0], 1, *p3_damage_logits.shape[-2:])
                    ),
                    "safe_class_transition": p3_damage_logits.new_zeros(
                        (1, 4, 1, 1)
                    ),
                }
                if return_dict:
                    return outputs
                return outputs["output_loc"], outputs["output_dam"]

        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(native_sar)

        _, localization_logits_low = self.p3_spatial_anchor(rgb_stages)
        localization_probability = localization_logits_low.softmax(1)[:, 1:2]
        sar_aux_low = self.sar_aux_head(sar_stages[-1])
        sar_probabilities = sar_aux_low.float().softmax(1)
        sar_confidence = sar_probabilities.amax(1, keepdim=True)
        sar_intact_probability = sar_probabilities[:, 1:2].to(sar_aux_low.dtype)

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

            key = f"stage{index}"
            if (
                self.h6_config.sensor_gap_prototype
                and key in self.sensor_damage_adapters
            ):
                joint, sdcr_diagnostics = self.sensor_damage_adapters[key](
                    joint,
                    stage_diagnostics["rgb_common"],
                    stage_diagnostics["sar_common"],
                    stage_diagnostics["rgb_unique"],
                    stage_diagnostics["sar_unique"],
                    localization_probability,
                    sar_intact_probability,
                )
                stage_diagnostics.update(sdcr_diagnostics)
            joint_stages.append(joint)
            for name, value in stage_diagnostics.items():
                diagnostics[f"stage{index}_{name}"] = value

        damage_feature = self.damage_decoder(joint_stages)
        damage_feature = damage_feature * (
            0.5 + 0.5 * localization_probability.detach()
        )
        base_damage_logits_low = self.damage_head(damage_feature)
        building_low = self.hierarchy_building_head(damage_feature)
        affected_low = self.hierarchy_affected_head(damage_feature)
        destroyed_low = self.hierarchy_destroyed_head(damage_feature)
        hierarchy_probability = hierarchical_probabilities(
            building_low, affected_low, destroyed_low
        )
        hierarchy_logits_low = hierarchy_probability.clamp_min(1e-7).log()
        hierarchy_transition = torch.tanh(self.hierarchy_transition)
        damage_logits_low = base_damage_logits_low + hierarchy_transition * (
            hierarchy_logits_low - base_damage_logits_low
        )

        native_localization_logits = F.interpolate(
                localization_logits_low,
                input_size,
                mode="bilinear",
                align_corners=False,
            )
        native_damage_logits = F.interpolate(
                damage_logits_low,
                input_size,
                mode="bilinear",
                align_corners=False,
            )
        output_localization = native_localization_logits
        output_damage = native_damage_logits
        if self.h6_config.safe_p3_residual:
            if p3_localization_logits is None or p3_damage_logits is None:
                raise RuntimeError("H6-B P3 safety anchor was not evaluated")
            output_localization = p3_localization_logits
            # Optional context interface for descendant *selectors*.  Existing
            # H6 routers do not implement it, so their forward path is exactly
            # unchanged.  H7-V2 uses the frozen RGB localization evidence as a
            # safety condition for building-to-background corrections.
            set_context = getattr(self.safe_residual_router, "set_context", None)
            if callable(set_context):
                set_context(
                    native_sar=native_sar,
                    localization_logits=p3_localization_logits,
                )
            output_damage, safe_diagnostics = self.safe_residual_router(
                p3_damage_logits,
                native_damage_logits,
            )
            diagnostics.update(safe_diagnostics)

        outputs: Dict[str, torch.Tensor] = {
            "output_loc": output_localization,
            "output_dam": output_damage,
            "native_output_loc": native_localization_logits,
            "native_output_dam": native_damage_logits,
            "p3_output_dam": (
                p3_damage_logits
                if p3_damage_logits is not None
                else native_damage_logits
            ),
            "base_damage_logits": F.interpolate(
                base_damage_logits_low,
                input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "sar_aux_logits": F.interpolate(
                sar_aux_low,
                input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "localization_probability_low": localization_probability,
            "hierarchy_building_logits": F.interpolate(
                building_low, input_size, mode="bilinear", align_corners=False
            ),
            "hierarchy_affected_logits": F.interpolate(
                affected_low, input_size, mode="bilinear", align_corners=False
            ),
            "hierarchy_destroyed_logits": F.interpolate(
                destroyed_low, input_size, mode="bilinear", align_corners=False
            ),
            "hierarchy_transition": hierarchy_transition.reshape(1, 1, 1, 1),
            **diagnostics,
        }
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_h6_sdcr_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH6SDCR:
    return DamageFormerRGBSARH6SDCR(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H6_VARIANTS",
    "H6_VARIANTS",
    "P3RGBSpatialAnchor",
    "SensorGapPrototypeAdapter",
    "SafeClassResidualRouter",
    "DamageFormerRGBSARH6SDCR",
    "build_h6_sdcr_damageformer",
]
