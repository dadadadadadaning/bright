"""One additive output-consistency objective; no model or base-loss duplication.

Train with ``student_logits`` whenever available.  This computes
KL(teacher.detach() || softmax(student_logits)) with log_softmax, including
underflowed student probabilities.  The probability-only fallback floors and
renormalizes the student distribution; its floor is reported explicitly.

The objective averages valid pixels within EACH image, then averages images.
An all-ignore image contributes a connected zero and remains in that image
denominator. Background pixels participate normally. Ground-truth class and
teacher-correctness statistics are diagnostics, never masks on the objective.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _batch_probability(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 4 or min(value.shape) == 0:
        raise ValueError(f"{name} must have shape [N,4,H,W] or [4,H,W]")
    return value


def _validate_probability(value: torch.Tensor, name: str) -> None:
    # Validation is not a training signal. Do not silently fix invalid inputs.
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool((detached < 0).any()) or bool((detached > 1).any()):
        raise ValueError(f"{name} values must be in [0,1]")
    total = detached.to(torch.float64).sum(dim=1)
    # Half-precision softmax can round by more than 1e-4; allow dtype rounding.
    tolerance = max(1e-5, 4.0 * torch.finfo(value.dtype).eps)
    if not bool(torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=0)):
        raise ValueError(f"{name} is not normalized over its four classes")


def _weighted_logratio(probability: torch.Tensor, log_other: torch.Tensor) -> torch.Tensor:
    # log(1) for a teacher zero avoids 0*log(0), without smoothing that teacher.
    safe_probability = torch.where(probability > 0, probability, torch.ones_like(probability))
    return probability * (safe_probability.log() - log_other)


def anchor_terms(
    student_probability: torch.Tensor,
    teacher_probability: torch.Tensor,
    target: torch.Tensor,
    weight: float = 0.1,
    *,
    student_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return additive KL and diagnostics; caller supplies the ORIGINAL loss.

    ``kl`` uses all non-255 pixels, including background and teacher mistakes.
    ``per_class_contribution`` partitions that same reduction by GT class and
    sums to ``kl``. ``per_class_conditional_kl`` instead averages a class's
    within-image means over images containing that class (zero if absent).
    ``valid_counts`` counts actual GT pixels, and ``class_present_image_counts``
    gives the conditional diagnostic's image denominator. No class balancing
    or confidence threshold is introduced into training.

    ``localization_kl + severity_kl_weighted == kl`` up to floating-point
    roundoff: foreground conditional KL is weighted by teacher foreground
    probability. This is the chain-rule decomposition of the same four-class
    objective, not two additional losses.
    """
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite non-negative real number")
    student = _batch_probability(student_probability, "student_probability")
    teacher_input = _batch_probability(teacher_probability, "teacher_probability")
    if student.shape != teacher_input.shape or student.device != teacher_input.device:
        raise ValueError("student and teacher must have identical shapes and devices")
    _validate_probability(student, "student_probability")
    _validate_probability(teacher_input, "teacher_probability")
    if not isinstance(target, torch.Tensor) or target.dtype == torch.bool or target.is_floating_point() or target.is_complex():
        raise TypeError("target must be an integer tensor with labels 0,1,2,3,255")
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.shape != student.shape[:1] + student.shape[2:] or target.device != student.device:
        raise ValueError("target shape/device must match the probability image grid")
    if bool(((target != 255) & ((target < 0) | (target > 3))).any()):
        raise ValueError("target may contain only 0,1,2,3 and ignore label 255")

    logits = None
    if student_logits is not None:
        logits = _batch_probability(student_logits, "student_logits")
        if logits.shape != student.shape or logits.device != student.device:
            raise ValueError("student_logits shape/device must match student_probability")
        if not bool(torch.isfinite(logits.detach()).all()):
            raise ValueError("student_logits contains non-finite values")
    dtype = torch.float64 if (student.dtype == torch.float64 or
                             teacher_input.dtype == torch.float64 or
                             (logits is not None and logits.dtype == torch.float64)) else torch.float32
    teacher = teacher_input.detach().to(dtype)
    teacher = teacher / teacher.sum(dim=1, keepdim=True)
    student = student.to(dtype)
    student_normalized = student / student.sum(dim=1, keepdim=True)
    if logits is not None:
        log_student = F.log_softmax(logits.to(dtype), dim=1)
        if not bool(torch.isfinite(log_student.detach()).all()):
            raise ValueError("student_logits range overflowed log_softmax; inspect upstream values")
        # Avoid accidentally anchoring a different output, crop or temperature.
        tolerance = max(1e-5, 4.0 * torch.finfo(student_probability.dtype).eps)
        if not bool(torch.allclose(student_normalized.detach(), log_student.detach().exp(),
                                   atol=tolerance, rtol=tolerance)):
            raise ValueError("student_probability does not match softmax(student_logits)")
        floor = 0.0
        clamped_count = torch.zeros((), dtype=torch.long, device=student.device)
    else:
        floor = 1e-12 if dtype == torch.float64 else 1e-8
        clamped_count = (student_normalized.detach() < floor).sum()
        stabilized = student_normalized.clamp_min(floor)
        stabilized = stabilized / stabilized.sum(dim=1, keepdim=True)
        log_student = stabilized.log()

    valid = target != 255
    per_image_count = valid.flatten(1).sum(dim=1)
    denominator = per_image_count.clamp_min(1).to(dtype)

    def reduce_image_mean(values: torch.Tensor, mask: torch.Tensor = valid) -> torch.Tensor:
        return ((values * mask.to(dtype)).flatten(1).sum(dim=1) / denominator).mean()

    pixel_kl = _weighted_logratio(teacher, log_student).sum(dim=1)
    kl = reduce_image_mean(pixel_kl)
    building_teacher = teacher[:, 1:].sum(dim=1)
    log_building_student = torch.logsumexp(log_student[:, 1:], dim=1)
    binary_teacher = torch.stack((teacher[:, 0], building_teacher), dim=1)
    binary_log_student = torch.stack((log_student[:, 0], log_building_student), dim=1)
    localization_pixel_kl = _weighted_logratio(binary_teacher, binary_log_student).sum(dim=1)
    teacher_conditional = teacher[:, 1:] / torch.where(
        building_teacher > 0, building_teacher, torch.ones_like(building_teacher)
    ).unsqueeze(1)
    log_student_conditional = log_student[:, 1:] - log_building_student.unsqueeze(1)
    severity_pixel_kl = building_teacher * _weighted_logratio(
        teacher_conditional, log_student_conditional
    ).sum(dim=1)

    class_masks = torch.stack([target == class_id for class_id in range(4)], dim=1)
    class_counts_per_image = class_masks.flatten(2).sum(dim=2)
    class_kl_sums_per_image = (pixel_kl.unsqueeze(1) * class_masks.to(dtype)).flatten(2).sum(dim=2)
    present_image_counts = (class_counts_per_image > 0).sum(dim=0)
    class_conditional = (
        class_kl_sums_per_image / class_counts_per_image.clamp_min(1).to(dtype)
    ).sum(dim=0) / present_image_counts.clamp_min(1).to(dtype)
    teacher_correct = valid & (teacher.argmax(dim=1) == target)

    return {
        "kl": kl,
        "weighted_kl": float(weight) * kl,
        "per_class_contribution": (class_kl_sums_per_image / denominator.unsqueeze(1)).mean(dim=0),
        "per_class_conditional_kl": class_conditional,
        "per_class_present": present_image_counts > 0,
        "valid_counts": class_counts_per_image.sum(dim=0),
        "class_present_image_counts": present_image_counts,
        "valid_pixels_per_image": per_image_count,
        "valid_pixel_count": per_image_count.sum(),
        "nonempty_image_count": (per_image_count > 0).sum(),
        "image_count": torch.as_tensor(student.shape[0], device=student.device),
        "localization_kl": reduce_image_mean(localization_pixel_kl),
        "severity_kl_weighted": reduce_image_mean(severity_pixel_kl),
        "teacher_building_mass": reduce_image_mean(building_teacher),
        "teacher_correct_kl_contribution": reduce_image_mean(pixel_kl, teacher_correct),
        "teacher_wrong_kl_contribution": reduce_image_mean(pixel_kl, valid & ~teacher_correct),
        "student_clamped_count": clamped_count,
        "probability_floor": torch.as_tensor(floor, dtype=dtype, device=student.device),
        "used_student_logits": torch.as_tensor(logits is not None, device=student.device),
    }


def serialized_anchor_terms(terms: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Detach diagnostic tensors for JSON; absent-class conditional means=null."""
    result = {key: value.detach().cpu().tolist() for key, value in terms.items()}
    if "per_class_conditional_kl" in result and "per_class_present" in result:
        result["per_class_conditional_kl"] = [
            value if present else None for value, present in
            zip(result["per_class_conditional_kl"], result["per_class_present"])
        ]
    return result


def gradient_alignment(
    original_loss: torch.Tensor,
    anchor_loss: torch.Tensor,
    reference_tensor: torch.Tensor,
    *,
    space: str = "logits",
) -> Dict[str, Any]:
    """Compare output gradients without accumulating or changing parameter grads.

    Pass the model's logits as ``reference_tensor`` and **weighted_kl** as
    ``anchor_loss`` to report the added objective's actual gradient strength.
    This reports output-space alignment, not full-parameter gradient alignment.
    Calling frequency is controlled by the runner; graphs are retained for the
    ordinary training backward pass. ``cosine=None`` when either norm is zero.
    """
    if not isinstance(space, str) or not space:
        raise ValueError("space must name the reference tensor's gradient space")
    if not isinstance(reference_tensor, torch.Tensor) or not reference_tensor.requires_grad:
        raise ValueError("reference_tensor must require gradients")
    gradients = []
    for name, loss in (("original_loss", original_loss), ("anchor_loss", anchor_loss)):
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not loss.requires_grad:
            raise ValueError(f"{name} must be a differentiable scalar tensor")
        gradient, = torch.autograd.grad(loss, reference_tensor, retain_graph=True,
                                        create_graph=False, allow_unused=True)
        gradients.append(torch.zeros_like(reference_tensor) if gradient is None else gradient)
    left, right = (gradient.detach().to(torch.float64).flatten() for gradient in gradients)
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("non-finite output gradients in consistency diagnostic")
    left_norm, right_norm = torch.linalg.vector_norm(left), torch.linalg.vector_norm(right)
    denominator = left_norm * right_norm
    cosine = None if denominator.item() == 0 else float((torch.dot(left, right) / denominator).clamp(-1, 1))
    return {
        "gradient_space": space,
        "original_grad_norm": float(left_norm),
        "anchor_grad_norm": float(right_norm),
        "anchor_to_original_grad_norm": None if left_norm.item() == 0 else float(right_norm / left_norm),
        "cosine": cosine,
        "anchor_gradient_uses_supplied_loss_weight": True,
    }
