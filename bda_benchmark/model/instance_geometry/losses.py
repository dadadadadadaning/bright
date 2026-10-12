"""CenterNet focal, boundary, and instance-balanced offset losses."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def centernet_focal_loss(
    logits: Tensor,
    target: Tensor,
    valid: Optional[Tensor] = None,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> Tensor:
    """CenterNet modified focal loss for Gaussian heatmaps.

    Loss arithmetic is intentionally FP32. In bf16, ``1 - 1e-6`` rounds to
    one, so a sigmoid/clamp/log implementation can still evaluate ``log(0)``
    and produce ``0 * inf == NaN``.
    """

    if logits.shape != target.shape:
        raise ValueError(
            f"center logits/target mismatch: {logits.shape} vs {target.shape}"
        )
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0,0.5)")
    logits_fp32 = logits.float()
    probability = torch.sigmoid(logits_fp32)
    target = target.float()
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = valid.float()

    positive = (target >= 1.0 - 1e-6).to(probability.dtype) * valid
    negative = (target < 1.0 - 1e-6).to(probability.dtype) * valid
    negative_weight = (1.0 - target).pow(beta)
    positive_loss = (
        -F.logsigmoid(logits_fp32)
        * (1.0 - probability).pow(alpha)
    )
    negative_loss = (
        -F.logsigmoid(-logits_fp32)
        * probability.pow(alpha)
        * negative_weight
    )
    positive_count = positive.sum()
    if float(positive_count.detach().item()) > 0:
        return (
            (positive_loss * positive).sum()
            + (negative_loss * negative).sum()
        ) / positive_count.clamp_min(1.0)
    return (negative_loss * negative).sum() / negative.sum().clamp_min(1.0)


def boundary_bce_dice_loss(
    logits: Tensor,
    target: Tensor,
    valid: Optional[Tensor] = None,
    *,
    eps: float = 1e-6,
) -> Tensor:
    if logits.shape != target.shape:
        raise ValueError(
            f"boundary logits/target mismatch: {logits.shape} vs {target.shape}"
        )
    logits_fp32 = logits.float()
    target = target.float()
    valid = torch.ones_like(target) if valid is None else valid.float()
    bce = F.binary_cross_entropy_with_logits(
        logits_fp32,
        target,
        reduction="none",
    )
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

    probability = torch.sigmoid(logits_fp32) * valid
    target_valid = target * valid
    intersection = (probability * target_valid).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (
        probability.sum() + target_valid.sum() + eps
    )
    return bce + dice


def instance_balanced_offset_loss(
    offset_logits: Tensor,
    offset_target: Tensor,
    offset_valid: Tensor,
    offset_weight: Tensor,
    *,
    beta: float = 1.0,
) -> Tensor:
    """Smooth-L1 offset loss with equal total weight per building instance."""

    if offset_logits.shape != offset_target.shape:
        raise ValueError(
            f"offset logits/target mismatch: {offset_logits.shape} "
            f"vs {offset_target.shape}"
        )
    if offset_valid.shape != offset_logits[:, :1].shape:
        raise ValueError("offset_valid must have shape [B,1,H,W]")
    if offset_weight.shape != offset_valid.shape:
        raise ValueError("offset_weight must match offset_valid")
    per_channel = F.smooth_l1_loss(
        offset_logits.float(),
        offset_target.float(),
        reduction="none",
        beta=beta,
    )
    weights = offset_valid.float() * offset_weight.float()
    return (per_channel * weights).sum() / (
        2.0 * weights.sum().clamp_min(1.0)
    )


def reliable_geometry_loss(
    center_logits: Tensor,
    boundary_logits: Tensor,
    targets: Mapping[str, Any],
    offset_logits: Optional[Tensor] = None,
    *,
    center_weight: float = 1.0,
    boundary_weight: float = 1.0,
    offset_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Combined G1 or S9-Id geometry loss."""

    loss_center = centernet_focal_loss(
        center_logits,
        targets["center"],
        targets.get("center_valid"),
    )
    loss_boundary = boundary_bce_dice_loss(
        boundary_logits,
        targets["boundary"],
        targets.get("center_valid"),
    )
    total = center_weight * loss_center + boundary_weight * loss_boundary
    result = {
        "loss_center": loss_center,
        "loss_boundary": loss_boundary,
    }
    if offset_logits is not None:
        required = ("offset", "offset_valid", "offset_weight")
        missing = [key for key in required if key not in targets]
        if missing:
            raise KeyError(f"offset targets are missing: {missing}")
        loss_offset = instance_balanced_offset_loss(
            offset_logits,
            targets["offset"],
            targets["offset_valid"],
            targets["offset_weight"],
        )
        total = total + offset_weight * loss_offset
        result["loss_offset"] = loss_offset
    result["loss"] = total
    return result
