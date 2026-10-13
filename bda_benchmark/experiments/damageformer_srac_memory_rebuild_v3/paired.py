"""Current-forward FP32 paired keys and independently checked current labels.

No archived token, distribution or region geometry is read here. The caller owns
the frozen front forward and supplies its output together with the F256 feature
captured from that same forward. Ground truth is optional and never enters key
extraction or the active-region mask. At validation, retrieve and predict before
passing validation ground truth to this observation-only label function.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
    mean_pool, validate_routes,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import (
    GroupedRouting, POOL_IMPLEMENTATION,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report


ATOL = 2e-5
RTOL = 2e-5
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("front result must be a mapping")
    required = ("tokens", "q", "ids", "count", "shape", "valid_regions", "mass")
    if any(key not in result for key in required):
        raise ValueError("front result lacks required current-forward fields")
    q, ids = result["q"], result["ids"]
    if not isinstance(q, Tensor) or not isinstance(ids, Tensor):
        raise ValueError("Q and IDs must be tensors")
    shape = result["shape"]
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)):
        raise ValueError("feature grid shape must contain two positive integers")
    q_check = validate_routes(q, ids, result["count"], tuple(shape))
    if q.shape[0] != 1:
        raise ValueError("paired extraction requires one image per front forward")
    count = result["count"]
    tokens, valid, mass = (result[key] for key in ("tokens", "valid_regions", "mass"))
    if not isinstance(tokens, Tensor) or tokens.shape != (1, count, 128):
        raise ValueError("current front tokens must have shape [1,K,128]")
    if tokens.dtype != torch.float32:
        raise ValueError("actual FP32 front tokens required; refusing to upcast " + str(tokens.dtype))
    if (not isinstance(valid, Tensor) or valid.shape != (1, count)
            or valid.dtype != torch.bool):
        raise ValueError("front valid_regions must be boolean [1,K]")
    if (not isinstance(mass, Tensor) or mass.shape != (1, count)
            or not mass.is_floating_point()):
        raise ValueError("front mass must be floating [1,K]")
    if any(value.device != q.device for value in (tokens, valid, mass)):
        raise ValueError("current front fields must share the Q device")
    if (not bool(torch.isfinite(tokens).all()) or not bool(torch.isfinite(mass).all())
            or bool((mass < 0).any())):
        raise ValueError("current tokens/mass must be finite and mass nonnegative")
    return q_check


def _target_fractions(target: Tensor, result: Mapping[str, Any], *, reference=False) -> Tensor:
    """Area fractions of current labels; invalid labels contribute zero mass."""
    if (not isinstance(target, Tensor) or target.ndim != 3 or target.shape[0] != 1
            or min(target.shape[-2:]) < 1 or target.dtype not in _INTEGER_DTYPES):
        raise ValueError("current target must be integer [1,H,W]")
    if not reference and target.device != result["q"].device:
        raise ValueError("current target and Q must share the caller device")
    labels = target.detach().to(device="cpu") if reference else target.detach()
    valid = (labels >= 0) & (labels < 4)
    dtype = torch.float64 if reference else torch.float32
    one_hot = F.one_hot(labels.to(torch.long).clamp(0, 3), num_classes=4)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype) * valid[:, None]
    return F.interpolate(one_hot, size=tuple(result["shape"]), mode="area")


def _normalize_counts(counts: Tensor) -> tuple[Tensor, Tensor]:
    label_mass = counts.sum(-1)
    denominator = torch.where(label_mass > 0, label_mass, torch.ones_like(label_mass))
    return counts / denominator[..., None], label_mass


@torch.no_grad()
def validate_label_reference(target: Tensor, result: Mapping[str, Any],
                             distribution: Tensor, label_mass: Tensor) -> dict[str, Any]:
    """All-route CPU FP64 scatter reference independent of grouped reductions.

    The reference rebuilds area fractions directly from current ground truth on
    CPU in FP64, then accumulates every candidate occurrence with scatter_add_.
    Duplicate IDs remain additive. Comparisons cover weighted counts, valid-label
    mass and normalized distributions, including tiny positive and empty rows.
    A failed reference is a hard error in ``extract_pair``; this standalone
    function returns the report so callers can also audit saved label values.
    """
    _validate_result(result)
    count = result["count"]
    if (not isinstance(distribution, Tensor) or distribution.shape != (count, 4)
            or not distribution.is_floating_point()
            or not isinstance(label_mass, Tensor) or label_mass.shape != (count,)
            or not label_mass.is_floating_point()):
        raise ValueError("labels must have distribution[K,4] and label_mass[K]")
    if (not bool(torch.isfinite(distribution).all())
            or not bool(torch.isfinite(label_mass).all())
            or bool((distribution < 0).any()) or bool((label_mass < 0).any())):
        raise ValueError("label evidence must be finite and nonnegative")
    fractions = _target_fractions(target, result, reference=True)[0].flatten(1).T.contiguous()
    q = result["q"].detach().cpu().double()[0]
    ids = result["ids"].detach().cpu()
    counts = torch.zeros(count, 4, dtype=torch.float64, device="cpu")
    # This reference deliberately does not use GroupedRouting, mean_pool or
    # native soft_pool_unpool, and does not combine duplicate occurrences first.
    for slot in range(ids.shape[0]):
        counts.scatter_add_(0, ids[slot, :, None].expand(-1, 4),
                            fractions * q[slot, :, None])
    reference_distribution, reference_mass = _normalize_counts(counts)
    actual_distribution = distribution.detach().cpu().double()
    actual_mass = label_mass.detach().cpu().double()
    reports = {
        "weighted_counts": close_report(actual_distribution * actual_mass[:, None], counts,
                                         atol=ATOL, rtol=RTOL),
        "label_mass": close_report(actual_mass, reference_mass, atol=ATOL, rtol=RTOL),
        "distribution": close_report(actual_distribution, reference_distribution,
                                      atol=ATOL, rtol=RTOL),
    }
    return {
        "passed": all(report["passed"] for report in reports.values()),
        "reference_dtype": str(torch.float64), "reference_device": "cpu",
        "reference_implementation": "cpu_fp64_area_then_scatter_add_v1",
        "routes_checked": ids.numel(), "regions_checked": count,
        "invalid_target_pixels": int(((target < 0) | (target >= 4)).sum()),
        "zero_label_regions": int((reference_mass == 0).sum()),
        "max_abs": max(report["max_abs"] for report in reports.values()),
        "atol": ATOL, "rtol": RTOL, **reports,
    }


def _extract_labels(target: Tensor, result: Mapping[str, Any],
                    routing: GroupedRouting | None = None) -> dict[str, Any]:
    fractions = _target_fractions(target, result)
    label_mean, route_mass = mean_pool(fractions, result["q"], result["ids"],
                                      result["count"], routing=routing)
    counts = label_mean[0] * route_mass[0, :, None]
    distribution, label_mass = _normalize_counts(counts)
    label_check = validate_label_reference(target, result, distribution, label_mass)
    if not label_check["passed"]:
        raise ValueError("current-label FP64 reference failed: " + str(label_check))
    return dict(distribution=distribution, label_mass=label_mass, label_check=label_check)


@torch.no_grad()
def extract_labels(target: Tensor, result: Mapping[str, Any]) -> dict[str, Any]:
    """Post-hoc current labels after neighbors/predictions have been fixed.

    Does not run the front, read any historical bank or compute either key.
    Returns distribution[K,4], label_mass[K] and the independent label_check.
    """
    _validate_result(result)
    return _extract_labels(target, result)


@torch.no_grad()
def extract_pair(result: Mapping[str, Any], feature: Tensor,
                 target: Tensor | None = None) -> dict[str, Any]:
    """Return row-aligned actual FP32 T128 and exact-positive-mass Q-mean F256.

    ``t128``/``f256`` have [K,128]/[K,256], ``mass``/``active`` have [K].
    ``active`` uses only front validity, Q mass and finite nonzero keys. Labels do
    not alter it. With target, add ``distribution[K,4]``, ``label_mass[K]`` and
    the mandatory independent FP64 label-reference report. Empty label rows are
    exactly zero; callers exclude those rows from a labelled support bank.
    """
    q_check = _validate_result(result)
    shape = tuple(result["shape"])
    if not isinstance(feature, Tensor) or feature.shape != (1, 256, *shape):
        raise ValueError("captured current F256 must have shape [1,256,h,w]")
    if feature.dtype != torch.float32:
        raise ValueError("actual FP32 captured F256 required; refusing to upcast " + str(feature.dtype))
    if feature.device != result["q"].device or not bool(torch.isfinite(feature).all()):
        raise ValueError("captured F256 must be finite and share the front device")
    routing = GroupedRouting(result["ids"], result["count"])
    means, mass = mean_pool(feature, result["q"], result["ids"], result["count"], routing=routing)
    front_mass_check = close_report(mass, result["mass"], atol=ATOL, rtol=RTOL)
    if not front_mass_check["passed"]:
        raise ValueError("front Q mass disagrees with current grouped mass: " + str(front_mass_check))
    t128, f256, mass = result["tokens"][0].detach(), means[0].detach(), mass[0].detach()
    if not bool(torch.isfinite(f256).all()):
        raise ValueError("current pooled F256 is nonfinite")
    active = result["valid_regions"][0] & (mass > 0)
    active = active & torch.isfinite(t128).all(-1) & torch.isfinite(f256).all(-1)
    # Match the bank/retrieval eligibility convention exactly in both spaces;
    # tiny nonzero keys do not become a different support set for one key type.
    active = active & (t128.norm(dim=-1) > 1e-8) & (f256.norm(dim=-1) > 1e-8)
    output = {
        "t128": t128, "f256": f256, "mass": mass, "active": active,
        "q_check": {**q_check, "front_mass": front_mass_check,
                    "pool_implementation": POOL_IMPLEMENTATION,
                    "key_dtype": str(torch.float32),
                    "q_dtype": str(result["q"].dtype),
                    "mean_pool_denominator": "exact positive mass; one only for zero mass"},
    }
    if target is not None:
        output.update(_extract_labels(target, result, routing))
    return output
