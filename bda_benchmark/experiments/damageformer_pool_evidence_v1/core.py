"""Offline, GT-conditioned descriptions of the *actual* pooling supports.

No result in this module is a model input. The route, attention, and GT are
never modified. Diagnostics use float64 sums and protect only zero
denominators: a tiny positive slot must not be silently treated as empty.

Mass and effective sample size (ESS) are in native feature-position units,
not original-image pixel units. Full-resolution GT contributes its area
fractions to each native cell; nearest-neighbour GT downsampling is forbidden.
"""
from collections.abc import Mapping

import torch
from torch.nn import functional as F


BIN_DEFINITIONS = {
    "semantic_purity": "GT four-class max fraction; high >= .95, mid [.8,.95), low < .8; valid GT mass > 0",
    "binary_purity": "GT background/foreground max fraction; same edges; valid GT mass > 0",
    "foreground_purity": "GT Intact/Damaged/Destroyed max fraction CONDITIONAL on foreground mass; same edges; foreground mass > 0",
    "mass": "positive native pooling mass: tiny (0,1e-6], subpixel (1e-6,1], small (1,16), large [16,infinity)",
    "ess": "effective native positions: tiny (0,1], small (1,4], medium (4,16), large [16,infinity)",
}


def _checked_route(route):
    q, ids = route["q"], route["ids"]
    k, shape = int(route["k"]), tuple(route["shape"])
    if q.ndim != 3 or ids.shape != q.shape or ids.dtype != torch.long:
        raise ValueError("Expected q and long ids shaped [batch, supports, native_positions]")
    if not q.is_floating_point() or q.shape[1] == 0 or k <= 0:
        raise ValueError("q must be floating point, with nonempty supports and a positive slot count")
    if len(shape) != 2 or any(int(x) != x or x <= 0 for x in shape):
        raise ValueError("route shape must contain two positive integer dimensions")
    shape = tuple(int(x) for x in shape)
    if shape[0] * shape[1] != q.shape[-1]:
        raise ValueError("route shape does not match its number of native positions")
    if ids.device != q.device or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Route weights must be finite/nonnegative and IDs must share their device")
    if (ids < 0).any() or (ids >= k).any():
        raise ValueError("Route ID outside [0, k)")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=2e-5, rtol=0):
        raise ValueError("Route weights must sum to one across supports at every native position")
    return q.detach(), ids.detach(), k, shape


def _coalesce(q, ids):
    """Merge repeated slot IDs PER native position before squaring for ESS.

    Border candidates can share a slot ID. Squaring each candidate separately
    would overestimate the number of independent contributing positions.
    Padded supports after coalescing have zero weight.
    """
    sorted_ids, order = ids.sort(dim=1)
    sorted_q = q.gather(1, order)
    starts = torch.ones_like(sorted_ids, dtype=torch.bool)
    starts[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    destination = starts.long().cumsum(1) - 1
    merged_q = torch.zeros_like(q).scatter_add(1, destination, sorted_q)
    # Exactly one run-start contributes an ID. Integer addition avoids any
    # repeated-write scatter ambiguity, including on deterministic CUDA runs.
    merged_ids = torch.zeros_like(ids).scatter_add(1, destination, sorted_ids * starts.long())
    return merged_q, merged_ids


def _safe_ratio(numerator, denominator):
    """Unlike clamp_min(eps), preserve ratios even for arbitrarily tiny slots."""
    positive = denominator > 0
    safe = torch.where(positive, denominator, torch.ones_like(denominator))
    return torch.where(positive, numerator / safe, torch.zeros_like(numerator))


@torch.no_grad()
def label_fractions(gt, native_shape):
    """Return [B,N,4] GT class-area fractions; all non-0..3 IDs are ignored.

    Only aligned integer-factor mappings are supported. Fail explicitly on
    incompatible resolutions rather than removing small labelled objects by
    a silent nearest-neighbour resize.
    """
    if gt.ndim != 3 or gt.is_floating_point() or gt.dtype == torch.bool:
        raise ValueError("GT must contain integer class IDs in [batch, original_height, original_width]")
    h, w = (int(x) for x in native_shape)
    height, width = gt.shape[-2:]
    if h <= 0 or w <= 0 or height < h or width < w or height % h or width % w:
        raise ValueError(
            f"GT/native resolution must use aligned integer area factors, got {height}x{width} -> {h}x{w}; "
            "do not silently replace this mapping with nearest-neighbour resizing"
        )
    kernel = (height // h, width // w)
    fractions = []
    for label in range(4):
        mask = (gt == label).to(dtype=torch.float64).unsqueeze(1)
        fractions.append(F.avg_pool2d(mask, kernel_size=kernel, stride=kernel))
    return torch.cat(fractions, 1).flatten(2).transpose(1, 2)


def _scatter_mass(values, weights, ids, k):
    """Raw sums, without means, thresholds, or denominator clamping."""
    b, n, channels = values.shape
    result = values.new_zeros(b, k, channels)
    for support in range(weights.shape[1]):
        result.scatter_add_(
            1, ids[:, support, :, None].expand(b, n, channels),
            weights[:, support, :, None] * values,
        )
    return result


def _purity_bins(name, value, active):
    return {
        name + "__high": active & (value >= .95),
        name + "__mid": active & (value >= .8) & (value < .95),
        name + "__low": active & (value < .8),
    }


def _describe(weights, ids, k, fractions):
    ones = fractions.new_ones(fractions.shape[0], fractions.shape[1], 1)
    mass = _scatter_mass(ones, weights, ids, k)[..., 0]
    square_sum = _scatter_mass(ones, weights.square(), ids, k)[..., 0]
    class_mass = _scatter_mass(fractions, weights, ids, k)
    valid_mass = class_mass.sum(-1)
    foreground_mass = class_mass[..., 1:].sum(-1)
    occupied, gt_valid, foreground_valid = mass > 0, valid_mass > 0, foreground_mass > 0
    purity = _safe_ratio(class_mass.max(-1).values, valid_mass)
    binary_purity = _safe_ratio(torch.maximum(class_mass[..., 0], foreground_mass), valid_mass)
    foreground_purity = _safe_ratio(class_mass[..., 1:].max(-1).values, foreground_mass)
    foreground_fraction = _safe_ratio(foreground_mass, valid_mass)
    # q duplicates have been coalesced, so each square represents one native
    # position. ESS describes support concentration, not independent GT pixels.
    ess = _safe_ratio(mass.square(), square_sum)
    masks = {}
    masks.update(_purity_bins("semantic_purity", purity, gt_valid))
    masks.update(_purity_bins("binary_purity", binary_purity, gt_valid))
    masks.update(_purity_bins("foreground_purity", foreground_purity, foreground_valid))
    masks.update({
        "mass__tiny": occupied & (mass <= 1e-6),
        "mass__subpixel": (mass > 1e-6) & (mass <= 1),
        "mass__small": (mass > 1) & (mass < 16),
        "mass__large": mass >= 16,
        "ess__tiny": occupied & (ess <= 1),
        "ess__small": occupied & (ess > 1) & (ess <= 4),
        "ess__medium": occupied & (ess > 4) & (ess < 16),
        "ess__large": occupied & (ess >= 16),
    })
    return {
        "mass": mass,
        "valid_mass": valid_mass,
        "class_mass": class_mass,
        "foreground_mass": foreground_mass,
        "purity": purity,
        "binary_purity": binary_purity,
        "foreground_purity": foreground_purity,
        "foreground_fraction": foreground_fraction,
        "valid_fraction": _safe_ratio(valid_mass, mass),
        "ess": ess,
        "weight_square_sum": square_sum,
        "occupied": occupied,
        "gt_valid": gt_valid,
        "foreground_valid": foreground_valid,
        "slot_masks": masks,
    }


@torch.no_grad()
def region_evidence(route, attention, gt):
    """Describe plain-Q statistics and attention-weighted main-path supports.

    Args:
        route: q [B,S,N], ids [B,S,N], k, shape (h,w); q is normalized.
        attention: [B,1,h,w] weights in [0,1], normally sigmoid(score(F)).
        gt: [B,H,W] original-resolution integer labels, only 0..3 valid.

    Returns:
        stats/attention dictionaries of [B,K] slot statistics, [B,K,4]
        class masses and mutually-exclusive masks within each bin family.
        Families overlap each other and MUST NOT be added together.
    """
    q, ids, k, shape = _checked_route(route)
    expected = (q.shape[0], 1, *shape)
    if tuple(attention.shape) != expected or attention.device != q.device:
        raise ValueError(f"attention must have shape {expected} on the route device")
    if not torch.isfinite(attention).all() or (attention < 0).any() or (attention > 1).any():
        raise ValueError("attention must be finite and in [0,1]")
    if gt.device != q.device or gt.shape[0] != q.shape[0]:
        raise ValueError("GT batch/device must match route")
    q, ids = _coalesce(q.to(dtype=torch.float64), ids)
    fractions = label_fractions(gt, shape)
    score = attention.detach().to(dtype=torch.float64).flatten(2)
    return {
        "stats": _describe(q, ids, k, fractions),
        "attention": _describe(q * score, ids, k, fractions),
    }


@torch.no_grad()
def bin_exposure(route, slot_masks, output_size):
    """Linear Q backprojection of slot membership, followed by output resize.

    slot_masks is an ordered name -> boolean [B,K] mapping (or sequence).
    Returns [B,T,H,W] in FP32. No channel renormalization is performed:
    a foreground-only collection intentionally has exposure sum <= 1.
    These maps are diagnostic fractions, not independent pixel predictions
    and not a causal attribution of nonlinear regional classifier errors.
    """
    q, ids, k, shape = _checked_route(route)
    masks = list(slot_masks.values()) if isinstance(slot_masks, Mapping) else list(slot_masks)
    if not masks:
        raise ValueError("At least one slot mask is required")
    expected = (q.shape[0], k)
    for mask in masks:
        if tuple(mask.shape) != expected or mask.dtype != torch.bool or mask.device != q.device:
            raise ValueError(f"Every slot mask must be bool {expected} on the route device")
    if len(output_size) != 2 or any(int(x) != x or x <= 0 for x in output_size):
        raise ValueError("output_size must contain two positive integers")
    values = torch.stack(masks, -1).to(dtype=torch.float32)
    b, _, n = q.shape
    result = values.new_zeros(b, n, len(masks))
    for support in range(q.shape[1]):
        index = ids[:, support, :, None].expand(b, n, len(masks))
        result.add_(values.gather(1, index) * q[:, support, :, None].float())
    result = result.transpose(1, 2).reshape(b, len(masks), *shape)
    return F.interpolate(result, size=tuple(int(x) for x in output_size),
                         mode="bilinear", align_corners=False)
