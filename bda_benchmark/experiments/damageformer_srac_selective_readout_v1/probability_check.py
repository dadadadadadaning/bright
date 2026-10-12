"""Strict probability checks and failure-only, observation-only projection traces.

No helper here repairs a prediction. Replays are diagnostic evidence only; the
parent projector and the existing 2e-5 absolute/relative gate stay unchanged.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


PROBABILITY_TOLERANCE = 2e-5


class ProbabilityContractError(ValueError):
    """A failed original probability contract with strict-JSON-safe evidence."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _number(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


def _statistics(value):
    """Summarize a tensor without putting non-JSON floating values in a report."""
    detached = value.detach()
    cpu = detached.to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(cpu)
    finite_values = cpu[finite]
    return dict(
        shape=list(detached.shape), dtype=str(detached.dtype), device=str(detached.device),
        elements=detached.numel(), finite_count=int(finite.sum()),
        nonfinite_count=int((~finite).sum()), nan_count=int(torch.isnan(cpu).sum()),
        positive_infinity_count=int(torch.isposinf(cpu).sum()),
        negative_infinity_count=int(torch.isneginf(cpu).sum()),
        negative_count=int((cpu < 0).sum()), zero_count=int((cpu == 0).sum()),
        minimum=_number(cpu.min()) if cpu.numel() else None,
        maximum=_number(cpu.max()) if cpu.numel() else None,
        finite_minimum=_number(finite_values.min()) if finite_values.numel() else None,
        finite_maximum=_number(finite_values.max()) if finite_values.numel() else None,
    )


def _sum_statistics(sums):
    result = _statistics(sums)
    error = (sums.detach().to(device="cpu", dtype=torch.float64) - 1).abs()
    finite = torch.isfinite(error)
    result.update(
        max_abs_error_from_one=_number(error.max()) if error.numel() else None,
        mean_finite_abs_error_from_one=_number(error[finite].mean()) if bool(finite.any()) else None,
        outside_tolerance_count=int((~finite | (error > 2 * PROBABILITY_TOLERANCE)).sum()),
    )
    return result


def _probability_diagnostics(probability, name, reasons, checked_sum=None):
    result = dict(
        name=str(name), passed=False, reasons=list(reasons),
        atol=PROBABILITY_TOLERANCE, rtol=PROBABILITY_TOLERANCE,
        effective_abs_tolerance_against_one=2 * PROBABILITY_TOLERANCE,
    )
    if not isinstance(probability, torch.Tensor):
        result.update(type=type(probability).__name__, shape=None)
        return result
    result.update(_statistics(probability))
    if probability.ndim != 4 or probability.shape[1] != 4 or not probability.numel():
        return result
    # Float32/64 sums are separately recomputed on CPU, so they can distinguish
    # stored values from the original device/dtype reduction used by the gate.
    cpu64 = probability.detach().to(device="cpu", dtype=torch.float64)
    native_sum = (probability.detach().sum(1) if checked_sum is None else checked_sum).cpu()
    sum32 = cpu64.float().sum(1)
    sum64 = cpu64.sum(1)
    result["class_sums"] = dict(
        native=_sum_statistics(native_sum),
        cpu_float32=_sum_statistics(sum32), cpu_float64=_sum_statistics(sum64),
    )
    result["class_sums"]["native"]["reduction_device"] = str(probability.device)
    score = (sum64 - 1).abs()
    invalid = ~torch.isfinite(cpu64).all(1)
    negative_magnitude = (-cpu64.amin(1)).clamp_min(0)
    score = torch.maximum(score, negative_magnitude)
    score[invalid] = float("inf")
    worst = torch.topk(score.reshape(-1), min(4, score.numel())).indices.tolist()
    height, width = probability.shape[-2:]
    result["worst_pixels"] = []
    for flat in worst:
        batch, remainder = divmod(flat, height * width)
        row, column = divmod(remainder, width)
        coordinate = (batch, row, column)
        result["worst_pixels"].append(dict(
            coordinate_bhw=list(coordinate),
            class_probabilities=[_number(item) for item in cpu64[batch, :, row, column]],
            sum_native=_number(native_sum[coordinate]),
            sum_cpu_float32=_number(sum32[coordinate]),
            sum_cpu_float64=_number(sum64[coordinate]),
            abs_sum_error_cpu_float64=_number((sum64[coordinate] - 1).abs()),
            nonfinite_class_count=int((~torch.isfinite(cpu64[batch, :, row, column])).sum()),
            negative_class_count=int((cpu64[batch, :, row, column] < 0).sum()),
        ))
    return result


@torch.no_grad()
def validate_probability(probability, name):
    """Validate BCHW, four-class output using the original native-dtype gate.

    Successful checks return only a small summary. Rich CPU diagnostics are
    collected only on failure; tensors are never modified, clamped or replaced.
    """
    reasons, sums = [], None
    if not isinstance(probability, torch.Tensor):
        reasons.append("not_a_tensor")
    elif probability.ndim != 4 or probability.shape[1] != 4 or not probability.numel():
        reasons.append("expected_nonempty_B4HW")
    elif not probability.is_floating_point():
        reasons.append("expected_floating_dtype")
    else:
        if not bool(torch.isfinite(probability).all()):
            reasons.append("nonfinite_values")
        if bool((probability < 0).any()):
            reasons.append("negative_values")
        sums = probability.sum(1)
        if not torch.allclose(
            sums, torch.ones_like(probability[:, 0]),
            atol=PROBABILITY_TOLERANCE, rtol=PROBABILITY_TOLERANCE,
        ):
            reasons.append("class_sum_not_one")
    if reasons:
        normalized_only = reasons == ["class_sum_not_one"]
        message = ("Boundary intervention is not normalized: " if normalized_only
                   else "Invalid probability from boundary intervention: ") + str(name)
        raise ProbabilityContractError(message, _probability_diagnostics(probability, name, reasons, sums))
    return dict(
        name=str(name), passed=True, shape=list(probability.shape),
        dtype=str(probability.dtype), device=str(probability.device),
        minimum=float(probability.min()), maximum=float(probability.max()),
        max_abs_class_sum_error=float((sums - 1).abs().max()),
        atol=PROBABILITY_TOLERANCE, rtol=PROBABILITY_TOLERANCE,
    )


def _stage(value, class_dimension=None):
    result = _statistics(value)
    if class_dimension is not None:
        result["class_sums"] = _sum_statistics(value.sum(class_dimension))
    return result


def _comparison(replayed, observed):
    result = dict(replayed_shape=list(replayed.shape), observed_shape=list(observed.shape))
    if replayed.shape != observed.shape:
        result.update(comparable=False, reason="shape_mismatch")
        return result
    replay_cpu = replayed.detach().to(device="cpu", dtype=torch.float64)
    observed_cpu = observed.detach().to(device="cpu", dtype=torch.float64)
    difference = (replay_cpu - observed_cpu).abs()
    finite_pairs = torch.isfinite(replay_cpu) & torch.isfinite(observed_cpu)
    result.update(
        comparable=True, exactly_equal=bool(torch.equal(replay_cpu, observed_cpu)),
        allclose_at_original_tolerance=bool(torch.allclose(
            replay_cpu, observed_cpu, atol=PROBABILITY_TOLERANCE, rtol=PROBABILITY_TOLERANCE,
        )),
        nonfinite_pair_count=int((~finite_pairs).sum()),
        max_abs_difference=_number(difference.max()) if difference.numel() else None,
        max_finite_abs_difference=_number(difference[finite_pairs].max()) if bool(finite_pairs.any()) else None,
    )
    return result


def _replay(logits, q, ids, shape, output_size, observed, cpu_float32):
    result = dict(status="started", stages={})
    stages = result["stages"]
    try:
        if cpu_float32:
            logits = logits.detach().to(device="cpu", dtype=torch.float32)
            q = q.detach().to(device="cpu", dtype=torch.float32)
            ids = ids.detach().to(device="cpu")
        else:
            logits, q, ids = logits.detach(), q.detach(), ids.detach()
        result["device"] = str(logits.device)
        result["q_dtype"] = str(q.dtype)
        # Keep this arithmetic identical to project_region_probabilities in
        # damageformer_srac_transformer_v1/model.py. Only observations are added.
        batch, _, classes = logits.shape
        probability = logits.float().softmax(dim=-1)
        stages["softmax"] = _stage(probability, -1)
        pixels = q.new_zeros(batch, q.shape[-1], classes)
        if ids.shape[0] != q.shape[1]:
            raise ValueError("Q and candidate IDs disagree")
        for candidate in range(q.shape[1]):
            gathered = probability.gather(1, ids[candidate][None, :, None].expand(batch, -1, classes))
            pixels = pixels + gathered * q[:, candidate, :, None]
        stages["sparse_q_mixture"] = _stage(pixels, -1)
        pixels = pixels.transpose(1, 2).reshape(batch, classes, *shape)
        pixels = F.interpolate(pixels, size=output_size, mode="bilinear", align_corners=False)
        stages["bilinear"] = _stage(pixels, 1)
        raw_denominator = pixels.sum(1, keepdim=True)
        denominator = raw_denominator.clamp_min(1e-8)
        stages["class_denominator"] = dict(
            raw=_statistics(raw_denominator), clamped=_statistics(denominator),
            floor=1e-8, below_floor_count=int((raw_denominator < 1e-8).sum()),
        )
        normalized = pixels / denominator
        stages["normalized_result"] = _stage(normalized, 1)
        try:
            result["probability_contract"] = validate_probability(normalized, "diagnostic_replay")
        except ProbabilityContractError as error:
            result["probability_contract"] = error.diagnostics
        result["comparison_to_observed"] = _comparison(normalized, observed)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
    return result


@torch.no_grad()
def projection_diagnostics(logits, q, ids, shape, output_size, observed):
    """Trace CPU-float32 and native-device replays after a projection failure.

    Replays never become predictions. Each backend captures its own exception so
    a failed replay cannot discard the other backend's evidence or the failure.
    """
    report = dict(
        diagnostic_only=True, predictions_replaced=False,
        shape=[int(item) for item in shape], output_size=[int(item) for item in output_size],
        formula="float32 softmax -> sparse Q mixture -> bilinear align_corners=False -> divide by class sum clamped at 1e-8",
        sources={}, replays={},
    )
    for name, value in (("logits", logits), ("q", q), ("ids", ids), ("observed", observed)):
        try:
            report["sources"][name] = _statistics(value)
        except Exception as error:
            report["sources"][name] = dict(error_type=type(error).__name__, error=str(error))
    try:
        report["sources"]["q"]["candidate_mass"] = _sum_statistics(q.detach().sum(1))
    except Exception as error:
        report["sources"]["q"]["candidate_mass_error"] = dict(error_type=type(error).__name__, error=str(error))
    report["replays"]["cpu_float32"] = _replay(logits, q, ids, shape, output_size, observed, True)
    report["replays"]["native"] = _replay(logits, q, ids, shape, output_size, observed, False)
    return report
