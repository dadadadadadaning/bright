# -*- coding: utf-8 -*-
"""H12-DGB: Damage-Gate Bypass for weak RGB--SAR damage evidence.

H11-T1E showed that frozen deep SARMAE blocks participate in the prediction
but do not recover Damaged recall.  The remaining error is the first
``Intact vs Affected`` decision, which needs shallow/mid-resolution evidence
rather than Stage-4 scene semantics.

H12 therefore keeps the locked H6-A2 localisation, native SAR encoder, fusion
stack and severity boundary unchanged.  It adds one exactly-zero residual to
the occurrence logit from Stage2 (D1) or Stage2+Stage3 (D2/D3).  The residual
is confined to an eroded RGB building ROI.  There is no Router, FCA, flow,
prototype, SARMAE injection, or 3-D convolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H10_H3D import (
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR


VALID_H12_VARIANTS = ("d0", "d1", "d2", "d3")
STAGE_INDEX = {"stage1": 0, "stage2": 1, "stage3": 2, "stage4": 3}
# Soft bound on the Stage2/3 candidate.  The candidate is additionally scaled by
# a small positive transition, so this is a last-resort numeric bound rather
# than the expected operating range.
OCCURRENCE_CANDIDATE_BOUND = 8.0
# The bypass is allowed to make only a small correction to the locked H6-A2
# occurrence logit.  A sigmoid parameterisation keeps the scale finite and
# positive; the zero-initialised output layer still makes the complete model
# exactly H6-A2 at initialisation.
OCCURRENCE_TRANSITION_MAX = 0.05
LOCAL_MOMENT_EPS = 1.0e-6


@dataclass(frozen=True)
class H12Variant:
    name: str
    bypass_stages: Tuple[str, ...]
    fn_aware_occurrence: bool


H12_VARIANTS: Mapping[str, H12Variant] = {
    "d0": H12Variant("H12-D0-Locked-H6A2", (), False),
    "d1": H12Variant("H12-D1-Stage2-Damage-Gate-Bypass", ("stage2",), False),
    "d2": H12Variant(
        "H12-D2-Stage2-Stage3-Damage-Gate-Bypass", ("stage2", "stage3"), False
    ),
    "d3": H12Variant(
        "H12-D3-Best-Bypass-FN-Aware-Occurrence", ("stage2", "stage3"), True
    ),
}


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        group_count = min(8, out_channels)
        while out_channels % group_count:
            group_count -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
        )


def bound_occurrence_candidate(
    candidate: torch.Tensor,
    limit: float = OCCURRENCE_CANDIDATE_BOUND,
) -> torch.Tensor:
    """Map an unbounded candidate into (-limit, limit) without flattening γ=0.

    ``limit * tanh(x / limit)`` is identity-to-first-order at 0, so the first
    ``dL/dγ`` remains non-zero.  The bound is applied in fp32 because AMP bf16
    saturates near 6.5e4 and turns ``tanh(γ) * candidate`` into Inf.
    """

    scale = float(limit)
    if scale <= 0.0:
        raise ValueError("occurrence candidate bound must be positive")
    candidate = candidate.float()
    if not torch.isfinite(candidate).all():
        raise FloatingPointError("non-finite H12 occurrence candidate")
    return scale * torch.tanh(candidate / scale)


def bounded_occurrence_transition(
    raw_transition: torch.Tensor,
    maximum: float = OCCURRENCE_TRANSITION_MAX,
) -> torch.Tensor:
    """Return a finite positive residual scale in ``(0, maximum)``."""

    maximum = float(maximum)
    if maximum <= 0.0:
        raise ValueError("occurrence transition maximum must be positive")
    raw_transition = raw_transition.float()
    if not torch.isfinite(raw_transition).all():
        raise FloatingPointError("non-finite H12 occurrence transition")
    return maximum * torch.sigmoid(raw_transition)


def erode_building_support(
    building_probability: torch.Tensor,
    radius: int = 1,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Detach a compact RGB building ROI; background cannot receive the residual."""

    if radius < 0:
        raise ValueError("erosion radius must be non-negative")
    binary = (building_probability.detach() >= threshold).to(
        dtype=building_probability.dtype
    )
    if radius == 0:
        return binary
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - binary, kernel_size=kernel, stride=1, padding=radius)


class WeakDamageBypassStage(nn.Module):
    """Parameter-light Stage2/3 RGB--SAR difference head.

    Optical and SAR projections are not shared.  The mixer sees the projected
    pair, their absolute difference, a local product, and 3x3 local moments of
    the difference.  The last convolution is zero-initialised: the complete
    model is exactly H6-A2 initially, while this output layer receives a
    first-step gradient through the small positive bounded transition.
    """

    def __init__(self, source_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.rgb = ConvGNAct(source_channels, hidden_channels, 1)
        self.sar = ConvGNAct(source_channels, hidden_channels, 1)
        self.mixer = nn.Sequential(
            ConvGNAct(6 * hidden_channels, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels),
        )
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _local_moments(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep the moment path in fp32 and add epsilon before sqrt.  Without
        # epsilon, flat building/SAR regions have variance=0 and sqrt's
        # derivative becomes infinite even though the forward value is finite.
        value = value.float()
        mean = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        second = F.avg_pool2d(value.square(), kernel_size=3, stride=1, padding=1)
        variance = (second - mean.square()).clamp_min(0.0)
        std = torch.sqrt(variance + LOCAL_MOMENT_EPS)
        return mean, std

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        building_support: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb = self.rgb(rgb)
        sar = self.sar(sar)
        difference = (rgb - sar).abs()
        product = rgb * sar
        local_mean, local_std = self._local_moments(difference)
        feature = self.mixer(
            torch.cat(
                [rgb, sar, difference, product, local_mean, local_std],
                dim=1,
            )
        )
        residual = self.output(feature)
        support = F.interpolate(
            building_support,
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = residual * support
        return residual, {
            "h12_bypass_feature": feature,
            "h12_building_support": support,
            "h12_occurrence_candidate_abs": residual.abs().mean(1, keepdim=True),
        }


class DamageFormerRGBSARH12DGB(DamageFormerRGBSARH6SDCR):
    """Safe H6-A2 descendant that only edits the Intact/Affected gate."""

    def __init__(
        self,
        variant: str = "d1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
        hidden_channels: int = 32,
        erosion_radius: int = 1,
        bypass_stages: Sequence[str] | None = None,
    ) -> None:
        variant = str(variant).lower()
        if variant not in VALID_H12_VARIANTS:
            raise ValueError(f"unknown H12 variant {variant!r}")
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        config = H12_VARIANTS[variant]
        stages = tuple(bypass_stages) if bypass_stages is not None else config.bypass_stages
        unknown = [name for name in stages if name not in STAGE_INDEX]
        if unknown:
            raise ValueError(f"unknown H12 bypass stages: {unknown}")
        self.h12_config = H12Variant(
            name=config.name,
            bypass_stages=stages,
            fn_aware_occurrence=config.fn_aware_occurrence,
        )
        self.erosion_radius = int(erosion_radius)
        self.bypass_heads = nn.ModuleDict()
        if stages:
            for stage_name in stages:
                channels = int(latent_channels[STAGE_INDEX[stage_name]])
                self.bypass_heads[stage_name] = WeakDamageBypassStage(
                    channels, hidden_channels
                )
            self.affected_transition = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _full_resolution(
        value: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
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
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        # Locked H6-A2 is an inference expert.  D1/D2 must not build a Stage2
        # backward graph through the frozen encoder (memory) or apply an
        # unbounded AMP residual to Damaged/Destroyed logits (NaN at step 2).
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
        if not self.h12_config.bypass_stages:
            if return_dict:
                return outputs
            return outputs["output_loc"], outputs["output_dam"]

        device_type = "cuda" if pre_rgb.is_cuda else pre_rgb.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            building_probability = outputs["output_loc"].float().softmax(1)[:, 1:2]
            building_support = erode_building_support(
                building_probability, radius=self.erosion_radius
            )
            candidate = None
            for stage_name in self.h12_config.bypass_stages:
                residual_low, diagnostics = self.bypass_heads[stage_name](
                    outputs[f"{stage_name}_rgb_common"].float(),
                    outputs[f"{stage_name}_sar_common"].float(),
                    building_support,
                )
                residual = self._full_resolution(residual_low, base_logits)
                candidate = residual if candidate is None else candidate + residual
                outputs.update(
                    {
                        f"h12_{stage_name}_{key}": value
                        for key, value in diagnostics.items()
                    }
                )

            if candidate is None:
                raise RuntimeError("H12 bypass produced no occurrence candidate")
            bound = float(OCCURRENCE_CANDIDATE_BOUND)
            bounded_candidate = bound_occurrence_candidate(candidate, limit=bound)
            occurrence_transition = bounded_occurrence_transition(
                self.affected_transition
            )
            occurrence_residual = (occurrence_transition * bounded_candidate).clamp(
                -bound, bound
            )
            if not torch.isfinite(occurrence_residual).all():
                raise FloatingPointError("non-finite H12 occurrence residual")
            updated_logits = apply_hierarchical_residuals(
                base_logits.float(), occurrence_residual=occurrence_residual
            )
            if not torch.isfinite(updated_logits).all():
                raise FloatingPointError("non-finite H12 updated damage logits")
            affected_logits, severity_logits = factorize_building_logits(
                updated_logits
            )
        outputs.update(
            {
                "h12_occurrence_transition": occurrence_transition.reshape(1, 1, 1, 1),
                "h12_occurrence_candidate_raw_abs": candidate.detach().abs(),
                "h12_occurrence_residual_abs": occurrence_residual.abs(),
                "h12_building_support": building_support,
                "h10_affected_logits": affected_logits,
                "h10_severity_logits": severity_logits,
                "output_dam": updated_logits,
            }
        )
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_h12_dgb_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
    bypass_stages: Sequence[str] | None = None,
) -> DamageFormerRGBSARH12DGB:
    return DamageFormerRGBSARH12DGB(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
        bypass_stages=bypass_stages,
    )


__all__ = [
    "VALID_H12_VARIANTS",
    "H12_VARIANTS",
    "STAGE_INDEX",
    "OCCURRENCE_CANDIDATE_BOUND",
    "OCCURRENCE_TRANSITION_MAX",
    "LOCAL_MOMENT_EPS",
    "bound_occurrence_candidate",
    "bounded_occurrence_transition",
    "erode_building_support",
    "WeakDamageBypassStage",
    "DamageFormerRGBSARH12DGB",
    "build_h12_dgb_damageformer",
]
