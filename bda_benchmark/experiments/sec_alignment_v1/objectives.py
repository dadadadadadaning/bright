# -*- coding: utf-8 -*-
"""Pure objectives for the isolated SEC experiment family."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SECVariant:
    name: str
    offset_delta: bool
    prediction_consistency: bool
    alignment_only: bool = False
    support_mode: str = "gt"
    coarse_to_fine: bool = False


SEC_VARIANTS: Mapping[str, SECVariant] = {
    "s1": SECVariant("S1-Shift-Augmentation-Control", False, False),
    "s2": SECVariant("S2-Relative-Offset-Equivariance", True, False),
    "s3": SECVariant("S3-Offset-and-Prediction-Consistency", True, True),
    # S2R is the safe repair of S2: learn correspondence only and leave the
    # already-validated D1 representation/classifier untouched.
    "s2r": SECVariant(
        "S2R-Safe-Query-Key-Alignment", True, False,
        alignment_only=True, support_mode="gt"
    ),
    "s2r0": SECVariant(
        "S2R0-Full-Image-Safe-Alignment", True, False,
        alignment_only=True, support_mode="all"
    ),
    "s2rp": SECVariant(
        "S2RP-Predicted-Building-Safe-Alignment", True, False,
        alignment_only=True, support_mode="predicted"
    ),
    "s2rr": SECVariant(
        "S2RR-Predicted-Building-Ring-Safe-Alignment", True, False,
        alignment_only=True, support_mode="predicted_ring"
    ),
    "s2rcf": SECVariant(
        "S2RCF-Building-Guided-Coarse-to-Fine-Alignment", True, False,
        alignment_only=True, support_mode="predicted_ring",
        coarse_to_fine=True,
    ),
}

STAGE_STRIDES = {2: 8, 3: 16}


def shift_without_wrap(
    value: torch.Tensor, dx: int = 0, dy: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Translate a tensor and return a one-channel validity mask.

    Positive ``dx``/``dy`` moves image content right/down.  Vacated pixels are
    zero; content never wraps around to the opposite side.
    """
    if value.ndim != 4:
        raise ValueError("value must be [B,C,H,W]")
    height, width = value.shape[-2:]
    if abs(int(dx)) >= width or abs(int(dy)) >= height:
        raise ValueError("shift must be smaller than the image dimensions")
    result = torch.zeros_like(value)
    validity = torch.zeros(
        (value.shape[0], 1, height, width),
        device=value.device,
        dtype=torch.float32,
    )
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    result[..., target_y, target_x] = value[..., source_y, source_x]
    validity[..., target_y, target_x] = 1.0
    return result, validity


def reference_validity_for_shift(
    value: torch.Tensor, dx: int = 0, dy: int = 0
) -> torch.Tensor:
    """Return reference pixels recoverable after applying a known shift."""
    ones = torch.ones(
        (value.shape[0], 1, value.shape[-2], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    reference_mask, _ = shift_without_wrap(ones, dx=-int(dx), dy=-int(dy))
    return reference_mask


def sample_axis_shift(
    amounts: Sequence[int], rng: random.Random
) -> Tuple[int, int]:
    """Sample one signed horizontal or vertical translation."""
    candidates = sorted({abs(int(value)) for value in amounts if int(value) != 0})
    if not candidates:
        raise ValueError("at least one non-zero shift amount is required")
    amount = rng.choice(candidates) * rng.choice((-1, 1))
    return (amount, 0) if rng.choice(("x", "y")) == "x" else (0, amount)


def _support_at(
    building_mask: torch.Tensor,
    validity: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    support = building_mask.float() * validity.float()
    return F.interpolate(support, size=size, mode="area").clamp(0.0, 1.0)


def select_alignment_support(
    clean: Mapping[str, torch.Tensor],
    location: torch.Tensor,
    mode: str,
    *,
    ring_radius_pixels: int = 16,
) -> torch.Tensor:
    """Select the image region trusted by relative-offset supervision.

    GT support is used only by the training loss. Predicted modes detach the
    RGB localization probability so the alignment objective cannot rewrite
    the frozen building localizer. The dilated mode keeps a search margin for
    SAR evidence displaced just outside the optical building footprint.
    """
    mode = str(mode).lower()
    height, width = location.shape[-2:]
    if mode == "all":
        return torch.ones(
            (location.shape[0], 1, height, width),
            device=location.device,
            dtype=torch.float32,
        )
    if mode == "gt":
        return (location == 1).float().unsqueeze(1)
    if mode not in ("predicted", "predicted_ring"):
        raise ValueError(f"unknown alignment support mode: {mode}")
    if "localization_probability_low" not in clean:
        raise KeyError("clean output has no localization_probability_low")
    predicted = F.interpolate(
        clean["localization_probability_low"].detach().float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    if mode == "predicted_ring":
        radius = max(0, int(ring_radius_pixels))
        if radius:
            predicted = F.max_pool2d(
                predicted,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )
    return predicted


def relative_offset_equivariance_loss(
    clean: Mapping[str, torch.Tensor],
    shifted: Mapping[str, torch.Tensor],
    building_mask: torch.Tensor,
    validity: torch.Tensor,
    *,
    dx: int,
    dy: int,
    beta: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Supervise the *change* in offsets caused by a known SAR translation.

    The clean offset is detached.  Consequently this objective never assumes
    the original RGB--SAR pair has zero displacement; it only asks the shifted
    branch to explain the synthetic displacement added by this experiment.
    """
    losses = []
    stage_values: Dict[str, torch.Tensor] = {}
    for stage, stride in STAGE_STRIDES.items():
        x_key, y_key = f"stage{stage}_offset_x", f"stage{stage}_offset_y"
        if x_key not in clean or x_key not in shifted:
            continue
        clean_offset = torch.cat((clean[x_key], clean[y_key]), dim=1).detach().float()
        shifted_offset = torch.cat((shifted[x_key], shifted[y_key]), dim=1).float()
        delta = shifted_offset - clean_offset
        target = delta.new_tensor((dx / float(stride), dy / float(stride))).view(1, 2, 1, 1)
        support = _support_at(building_mask, validity, delta.shape[-2:])
        error = F.smooth_l1_loss(
            delta, target.expand_as(delta), beta=float(beta), reduction="none"
        ).mean(1, keepdim=True)
        loss = (error * support).sum() / support.sum().clamp_min(1.0)
        losses.append(loss)
        stage_values[f"stage{stage}_offset_delta_loss"] = loss.detach()
    if losses:
        total = torch.stack(losses).mean()
    else:
        total = shifted["output_dam"].float().sum() * 0.0
    return {"offset_delta_loss": total, **stage_values}


def building_prediction_consistency_loss(
    clean_logits: torch.Tensor,
    shifted_logits: torch.Tensor,
    building_mask: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Make shifted-SAR damage probabilities agree with the clean prediction."""
    teacher = F.softmax(clean_logits.detach().float(), dim=1)
    student = F.log_softmax(shifted_logits.float(), dim=1)
    divergence = F.kl_div(student, teacher, reduction="none").sum(1, keepdim=True)
    support = _support_at(building_mask, validity, divergence.shape[-2:])
    return (divergence * support).sum() / support.sum().clamp_min(1.0)


def linear_warmup(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(warmup_steps)))


__all__ = [
    "SEC_VARIANTS",
    "STAGE_STRIDES",
    "SECVariant",
    "building_prediction_consistency_loss",
    "linear_warmup",
    "relative_offset_equivariance_loss",
    "reference_validity_for_shift",
    "sample_axis_shift",
    "select_alignment_support",
    "shift_without_wrap",
]
