"""CPU-only evidence for current FP32 tokens and stored historical FP16 tokens.

Both comparisons keep the existing elementwise atol=rtol=2e-5. Quantization
cells describe where the *current* FP32 value lies; the historical FP32 value
before storage is unavailable and cannot be reconstructed by this diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch

ATOL = RTOL = 2e-5
MAXIMUM_OUTLIERS = 20


def _cpu_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(name + " must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(name + " must already be on CPU; transfer the observed snapshot outside this helper")
    if value.layout != torch.strided:
        raise ValueError(name + " must use strided tensor storage")
    return value.detach()


def tensor_digest(value):
    value = _cpu_tensor(value, "checksum value")
    h = hashlib.sha256()
    h.update(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
    h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _coordinate(flat, shape):
    values = []
    for length in reversed(shape):
        values.append(flat % length)
        flat //= length
    return list(reversed(values))


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _comparison_tensors(actual, reference):
    a = _cpu_tensor(actual, "actual")
    b = _cpu_tensor(reference, "reference")
    if a.shape != b.shape:
        raise ValueError("Comparison shape mismatch")
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    finite = torch.isfinite(a64) & torch.isfinite(b64)
    difference = (a64 - b64).abs()
    outside = ~finite | (difference > ATOL + RTOL * b64.abs())
    return a64, b64, finite, difference, outside


def close_report(actual, reference):
    """The fixed existing threshold, evaluated in CPU double precision."""
    a, b, finite, difference, outside = _comparison_tensors(actual, reference)
    finite_difference = difference[finite]
    return dict(passed=not bool(outside.any()), atol=ATOL, rtol=RTOL,
                compared_values=a.numel(), exact_different_values=int((a != b).sum()),
                nonfinite_values=int((~finite).sum()), outside_tolerance=int(outside.sum()),
                max_abs=float(finite_difference.max()) if finite_difference.numel() else None,
                mean_abs=float(finite_difference.mean()) if finite_difference.numel() else None)


def cpu_cast(current_raw, stored):
    """Observe the current FP32-to-stored-dtype cast; never recreate old FP32."""
    current_raw = _cpu_tensor(current_raw, "current raw token")
    stored = _cpu_tensor(stored, "stored token")
    if current_raw.dtype != torch.float32 or stored.dtype != torch.float16:
        raise ValueError("Require current raw FP32 and historical stored FP16 tokens")
    if current_raw.shape != stored.shape:
        raise ValueError("Current/stored token shape mismatch")
    if not current_raw.numel():
        raise ValueError("Cannot diagnose an empty token tensor")
    return current_raw.float().cpu().to(stored.dtype)


def tensor_summary(value):
    value = _cpu_tensor(value, "summary value")
    finite = torch.isfinite(value)
    values = value[finite].to(torch.float64)
    return dict(dtype=str(value.dtype), shape=list(value.shape), device="cpu",
                compared_values=value.numel(), finite=bool(finite.all()),
                finite_values=int(finite.sum()), nonfinite_values=int((~finite).sum()),
                minimum=float(values.min()) if values.numel() else None,
                maximum=float(values.max()) if values.numel() else None,
                sha256=tensor_digest(value))


def rounding_cell(stored_scalar, current_raw_scalar, current_cast_scalar):
    """FP16 neighbors and midpoint interval around one *stored* finite value.

    Normalized distances use the local neighboring step in the direction of
    the difference. They are not a count of all representable intervening
    numbers across exponent boundaries. At max finite FP16, infinite neighbors
    make the finite midpoint interval unavailable, rather than proving a match.
    """
    old, raw, cast = map(float, (stored_scalar, current_raw_scalar, current_cast_scalar))
    if not math.isfinite(old):
        return dict(available=False, reason="stored value is nonfinite",
                    current_raw=_finite_number(raw), current_cast=_finite_number(cast))
    half = torch.tensor(old, dtype=torch.float16)
    lower = float(torch.nextafter(half, torch.full_like(half, -float("inf"))))
    upper = float(torch.nextafter(half, torch.full_like(half, float("inf"))))
    lower_midpoint, upper_midpoint = (lower + old) / 2.0, (old + upper) / 2.0
    finite_cell = math.isfinite(lower_midpoint) and math.isfinite(upper_midpoint)

    def distance(value):
        if not math.isfinite(value):
            return None
        step = (upper - old) if value >= old else (old - lower)
        return abs(value - old) / step if math.isfinite(step) and step > 0 else None

    if not math.isfinite(raw) or not finite_cell:
        position = "unavailable_nonfinite_raw_or_boundary"
        contains = None
    elif raw < lower_midpoint:
        position, contains = "below_lower_midpoint", False
    elif raw == lower_midpoint:
        position, contains = "at_lower_midpoint", True
    elif raw < upper_midpoint:
        position, contains = "inside_midpoint_interval", True
    elif raw == upper_midpoint:
        position, contains = "at_upper_midpoint", True
    else:
        position, contains = "above_upper_midpoint", False
    if not math.isfinite(cast):
        relation = "current_cast_nonfinite"
    elif cast == old:
        relation = "stored_value"
    elif cast == lower:
        relation = "lower_neighbor"
    elif cast == upper:
        relation = "upper_neighbor"
    else:
        relation = "beyond_lower_neighbor" if cast < old else "beyond_upper_neighbor"
    return dict(available=True, stored=old, lower_neighbor=_finite_number(lower),
                upper_neighbor=_finite_number(upper),
                lower_step=_finite_number(old - lower), upper_step=_finite_number(upper - old),
                lower_midpoint=_finite_number(lower_midpoint), upper_midpoint=_finite_number(upper_midpoint),
                finite_midpoint_interval=finite_cell, interval_contains_current_raw=contains,
                current_raw_position=position, current_nearest_fp16=_finite_number(cast),
                current_rounding_relation=relation,
                raw_distance_in_directional_local_steps=distance(raw),
                cast_distance_in_directional_local_steps=distance(cast),
                midpoint_endpoints_may_require_ties_to_even=True,
                historical_raw_fp32_reconstructed=False)


def compare_tokens(current_raw, stored, *, maximum_outliers=MAXIMUM_OUTLIERS):
    """Read-only report; no quantization rule is allowed to override either gate."""
    if type(maximum_outliers) is not int or not 1 <= maximum_outliers <= MAXIMUM_OUTLIERS:
        raise ValueError("maximum_outliers must be an integer from 1 to 20")
    current_raw, stored = (_cpu_tensor(value, name) for value, name in
                           ((current_raw, "current raw token"), (stored, "stored token")))
    cast = cpu_cast(current_raw, stored)
    raw_report, cast_report = close_report(current_raw, stored), close_report(cast, stored)
    raw64, old64, raw_finite, raw_difference, raw_outside = _comparison_tensors(current_raw, stored)
    cast64, _, cast_finite, cast_difference, cast_outside = _comparison_tensors(cast, stored)
    union = raw_outside | cast_outside
    # Preserve the fixed gates while making the diagnostically important cast
    # failures visible even when ordinary raw-to-FP16 rounding differs first.
    cast_coordinates = torch.nonzero(cast_outside.reshape(-1), as_tuple=False).flatten()
    raw_only_coordinates = torch.nonzero((raw_outside & ~cast_outside).reshape(-1), as_tuple=False).flatten()
    coordinates = torch.cat((cast_coordinates, raw_only_coordinates))[:maximum_outliers].tolist()
    flat_raw, flat_old, flat_cast = raw64.reshape(-1), old64.reshape(-1), cast64.reshape(-1)
    outliers = []
    for flat in coordinates:
        raw, old, rounded = (float(value[flat]) for value in (flat_raw, flat_old, flat_cast))
        outliers.append(dict(coordinate=_coordinate(flat, current_raw.shape), flat_index=flat,
                             old=_finite_number(old), current_raw=_finite_number(raw),
                             current_cast=_finite_number(rounded),
                             raw_absolute_difference=_finite_number(abs(raw - old)),
                             cast_absolute_difference=_finite_number(abs(rounded - old)),
                             outside_raw_tolerance=bool(raw_outside.reshape(-1)[flat]),
                             outside_cast_tolerance=bool(cast_outside.reshape(-1)[flat]),
                             fp16=rounding_cell(old, raw, rounded)))
    return dict(format="MEMORY_TOKEN_NUMERIC_DIAGNOSTIC_V1", diagnostic_only=True,
                full_validation_evidence=False, original_gate_unchanged=True,
                historical_raw_fp32_available=False,
                interpretation="The midpoint cell describes current FP32 rounding; it cannot reconstruct or prove agreement with unavailable historical FP32 tokens",
                shape=list(current_raw.shape), compared_values=current_raw.numel(),
                dtype_guard_passed=True, shape_guard_passed=True,
                current_raw=tensor_summary(current_raw), current_cast=tensor_summary(cast),
                stored=tensor_summary(stored), raw_to_stored=raw_report,
                cast_to_stored=cast_report, outliers=outliers,
                outlier_rule="Cast-outside coordinates first in row-major order, then raw-only-outside coordinates in row-major order; fixed tolerance unchanged",
                outlier_limit=maximum_outliers, outliers_available=int(union.sum()),
                outliers_saved=len(outliers))


numeric_report = compare_tokens
