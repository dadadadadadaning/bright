"""Check frozen token replay against a quantized, unchanged archive.

An archived FP16 token does not retain its original FP32 value. Its rounding
cell bounds the values that could have produced it under round-to-nearest.
Comparing an independently rounded replay directly to that archive can turn
a tiny numerical difference across a cell boundary into a whole FP16 ULP.

This helper tests compatibility with the archived cell using the existing
2e-5 absolute and relative numerical tolerance. It does not establish bitwise
replay or recover the original FP32 value. No archive/key is replaced, no
label check changes, and FP32 archives retain the ordinary direct check.
"""

from __future__ import annotations

import torch
from torch import Tensor


ATOL = 2e-5
RTOL = 2e-5


def _direct_report(actual: Tensor, expected: Tensor) -> dict:
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return dict(passed=False, finite=False, max_abs=None,
                    outside_tolerance=None, atol=ATOL, rtol=RTOL)
    difference = (actual - expected).abs()
    outside = difference > ATOL + RTOL * expected.abs()
    return dict(passed=not bool(outside.any()), finite=True,
                max_abs=float(difference.max()),
                outside_tolerance=int(outside.sum()), atol=ATOL, rtol=RTOL)


def _half_rounding_cell(stored: Tensor) -> tuple[Tensor, Tensor]:
    """Finite FP64 bounds of the closed hull of an FP16 rounding cell.

    Nearest-even assigns midpoint ties according to the significand parity.
    The closed hull has the same distance as the half-open cell, so it is
    sufficient for compatibility with a positive numerical tolerance. The
    maximum finite half has a virtual outer neighbor 65536; its overflow
    midpoint is 65520. Using the actual nextafter infinity as a bound would
    incorrectly accept arbitrary large finite replays.
    """
    # All finite-half arithmetic for midpoint construction is promoted before
    # addition. Adding 65504 and 65472 in half precision would overflow.
    value = stored.detach().cpu()
    previous = torch.nextafter(value, torch.full_like(value, -torch.inf)).double()
    following = torch.nextafter(value, torch.full_like(value, torch.inf)).double()
    previous = torch.where(torch.isneginf(previous),
                           torch.full_like(previous, -65536.0), previous)
    following = torch.where(torch.isposinf(following),
                            torch.full_like(following, 65536.0), following)
    archived = value.double()
    lower, upper = (previous + archived) / 2, (archived + following) / 2
    # IEEE conversion preserves the sign of a value rounded down to zero.
    # Thus +0 and -0 use their own half-cells. Exact signed zeros remain
    # numerically equal in the distance calculation and legacy comparisons.
    zero, negative = value == 0, torch.signbit(value)
    lower = torch.where(zero & ~negative, torch.zeros_like(lower), lower)
    upper = torch.where(zero & negative, torch.zeros_like(upper), upper)
    return lower, upper


@torch.no_grad()
def stored_token_replay(replayed_float32: Tensor, stored: Tensor) -> dict:
    """Report raw, rounded and storage-aware compatibility without mutation.

    Only raw FP32 replays and FP16/FP32 archives are accepted. A shape
    mismatch or nonfinite value returns a failed report; an unsupported dtype
    is a caller/configuration error. The decision uses raw FP32 values, never
    the replay after casting it to half. `max_abs_stored_precision` and the
    direct `legacy_report` retain the former rounded check for review, while
    `raw_report` records the strict raw-to-archive comparison separately.
    """
    if not isinstance(replayed_float32, Tensor) or not isinstance(stored, Tensor):
        raise TypeError("Replay and archive must be torch tensors")
    if replayed_float32.dtype != torch.float32:
        raise ValueError("Replay must retain the raw FP32 token values")
    if stored.dtype not in (torch.float16, torch.float32):
        raise ValueError("Stored tokens must have FP16 or FP32 archive precision")
    aware = stored.dtype == torch.float16
    base = dict(passed=False, finite=False, atol=ATOL, rtol=RTOL,
                storage_dtype=str(stored.dtype), quantization_aware=aware,
                max_abs_raw=None, max_abs_stored_precision=None,
                max_quantization_interval_distance=None,
                outside_tolerance=None, rounded_equal_fraction=None,
                d_min=None, d_max=None,
                definite_within_tol_count=None, storage_uncertain_count=None,
                definite_drift_count=None,
                legacy_report=None, raw_report=None,
                signed_zero_rule="sign-preserving cells; exact signed zeros numerically equal",
                check="FP16 rounding-cell compatibility" if aware else "FP32 direct replay")
    if replayed_float32.shape != stored.shape:
        return dict(base, shape=list(replayed_float32.shape),
                    expected_shape=list(stored.shape), error="shape mismatch")
    if not stored.numel():
        raise ValueError("Empty token replay cannot establish archive compatibility")
    # CPU FP64 computes distances to exact half midpoints independently of
    # model device, CUDA scatter/determinism settings and autocast.
    actual = replayed_float32.detach().double().cpu()
    archived = stored.detach().double().cpu()
    raw_report = _direct_report(actual, archived)
    if not raw_report["finite"]:
        return dict(base, legacy_report=raw_report, raw_report=raw_report,
                    error="nonfinite replay or archived token")
    rounded = replayed_float32.detach().cpu().to(stored.dtype).double()
    rounded_report = _direct_report(rounded, archived)
    if aware:
        lower, upper = _half_rounding_cell(stored)
        # Zero inside the storage uncertainty interval; exact positive
        # distance outside it. No key clipping or normalization occurs.
        distance = torch.maximum(lower - actual, actual - upper).clamp_min(0)
        maximum_distance = torch.maximum((actual - lower).abs(), (actual - upper).abs())
    else:
        distance = (actual - archived).abs()
        maximum_distance = distance
    tolerance = ATOL + RTOL * archived.abs()
    outside = distance > tolerance
    certain = maximum_distance <= tolerance
    uncertain = ~outside & ~certain
    return dict(base,
                passed=not bool(outside.any()), finite=True,
                max_abs_raw=raw_report["max_abs"],
                max_abs_stored_precision=rounded_report["max_abs"],
                max_quantization_interval_distance=float(distance.max()),
                d_min=float(distance.max()), d_max=float(maximum_distance.max()),
                outside_tolerance=int(outside.sum()),
                definite_within_tol_count=int(certain.sum()),
                storage_uncertain_count=int(uncertain.sum()),
                definite_drift_count=int(outside.sum()),
                rounded_equal_fraction=float((rounded == archived).double().mean()),
                rounded_finite=rounded_report["finite"],
                legacy_report=rounded_report, raw_report=raw_report)
