"""Post-hoc dual-slot diagnostics; no labels enter model or retrieval inputs.

Role 0 is the background-weighted slot, role 1 the building-weighted slot.
Strata overlap. Their weights are ground-truth mass in feature-grid cell units,
not counts of regions or raw pixels. ``gt_building_boundary`` denotes a binary
building/background boundary, never a building-instance boundary. A foreground
cell is on this boundary when a background-containing cell occurs in its 3x3
feature-grid neighborhood. Ignore pixels and image padding are not background.

Feature cosine is undefined for a zero-norm slot. Label distances require valid
labels in both slots. Undefined measurements are NaN internally and omitted with
their own denominators by RegionAccumulator; missing values are never zeroes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _ids2(ids: Tensor, weights: Tensor, count: int) -> Tensor:
    if ids.ndim == 3 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.shape != weights.shape or ids.dtype != torch.long:
        raise ValueError("route IDs must be int64 [S,N] or [1,S,N]")
    if ids.device != weights.device or bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError("invalid route IDs or route device")
    return ids


def _pool(values: Tensor, weights: Tensor, ids: Tensor, count: int) -> Tensor:
    """[C,N] values, [S,N] routes -> unnormalized [R,C] mass."""
    result = values.new_zeros(count, values.shape[0])
    pixels = values.transpose(0, 1)
    for candidate in range(weights.shape[0]):
        result.scatter_add_(0, ids[candidate, :, None].expand_as(pixels),
                            pixels * weights[candidate, :, None])
    return result


def _mass(weights: Tensor, ids: Tensor, count: int) -> Tensor:
    result = weights.new_zeros(count)
    result.scatter_add_(0, ids.reshape(-1), weights.reshape(-1))
    return result


def _ess(weights: Tensor, ids: Tensor, count: int) -> Tensor:
    """ESS of actual region/pixel weights, merging repeated candidate IDs.

    If two candidates refer to the same region at a pixel, the pixel has one
    combined weight. Counting it twice would overstate the effective support.
    This sparse implementation never constructs a region-by-all-pixels matrix.
    """
    mass = _mass(weights, ids, count)
    square = weights.new_zeros(count)
    for candidate in range(weights.shape[0]):
        same_region = ids == ids[candidate: candidate + 1]
        combined = (weights * same_region).sum(0)
        square.scatter_add_(0, ids[candidate], weights[candidate] * combined)
    return torch.where(square > 0, mass.square() / square.clamp_min(torch.finfo(weights.dtype).tiny),
                       torch.full_like(mass, float("nan")))


def _masked(value: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid, value, torch.full_like(value, float("nan")))


@torch.no_grad()
def slot_diagnostics(feature: Tensor, q: Tensor, ids: Tensor, building: Tensor,
                     target: Tensor, slots: Mapping[str, Tensor | int]) -> dict:
    """Audit one image using the exact slots returned by original prepare_slots.

    Returns ``region_rows`` metric tensors [R], ``weights`` stratum masses [R],
    full-resolution Boolean ``pixel_masks`` [H,W], and post-hoc query labels
    [R,2,4], their validity [R,2], and valid-label mass [R,2]. B is bilinearly
    resized exactly as routing does; valid one-hot GT is area-resized without
    renormalizing partially ignored cells before pooling.
    """
    if feature.ndim != 4 or feature.shape[0] != 1 or min(feature.shape) < 1:
        raise ValueError("feature must be [1,C,h,w]")
    h, w = feature.shape[-2:]
    if q.ndim != 3 or q.shape != (1, 9, h * w):
        raise ValueError("Q must be [1,9,h*w]")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()) or not torch.allclose(
            q.sum(1), torch.ones_like(q[:, 0]), atol=2e-5, rtol=2e-5):
        raise ValueError("Q weights must be finite nonnegative and sum to one")
    if target.ndim != 3 or target.shape[0] != 1 or min(target.shape) < 1:
        raise ValueError("target must be [1,H,W]")
    if target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("target must contain integer labels")
    if building.ndim != 4 or building.shape[:2] != (1, 1) or min(building.shape) < 1:
        raise ValueError("building must be [1,1,H,W]")
    if not bool(torch.isfinite(building).all()) or bool(((building < 0) | (building > 1)).any()):
        raise ValueError("building probabilities must be finite in [0,1]")
    if any(t.device != feature.device for t in (q, ids, building, target)):
        raise ValueError("all inputs must be on the same device")

    slot_f = slots["slot_f"]
    slot_mass = slots["slot_mass"]
    slot_fraction = slots["slot_fraction"]
    slot_valid = slots["slot_valid"]
    if not isinstance(slot_f, Tensor) or slot_f.ndim != 4 or slot_f.shape[0] != 1 or slot_f.shape[2:] != (2, feature.shape[1]):
        raise ValueError("slot_f must be [1,R,2,C]")
    count = slot_f.shape[1]
    if slots["slot_count"] != 2 * count:
        raise ValueError("slot_count disagrees with slot_f")
    for name, value in (("slot_mass", slot_mass), ("slot_fraction", slot_fraction), ("slot_valid", slot_valid)):
        if not isinstance(value, Tensor) or value.shape != (1, count, 2) or value.device != feature.device:
            raise ValueError(name + " must be [1,R,2] on feature device")
    route_q = slots["route_q"]
    route_ids = slots["route_ids"]
    if not isinstance(route_q, Tensor) or route_q.shape != (1, 18, h * w):
        raise ValueError("slot routes must have shape [1,18,h*w]")
    if route_q.device != feature.device or not bool(torch.isfinite(route_q).all()) or bool((route_q < 0).any()):
        raise ValueError("invalid slot weights")
    if not torch.allclose(route_q.sum(1), q.sum(1), atol=2e-5, rtol=2e-5):
        raise ValueError("slot weights do not conserve Q mass")
    ids = _ids2(ids, q[0], count)
    route_ids = _ids2(route_ids, route_q[0], 2 * count)
    if not bool(torch.isfinite(slot_f).all()) or not bool(torch.isfinite(slot_mass).all()):
        raise ValueError("slot features/masses must be finite")
    computed_mass = _mass(route_q[0], route_ids, 2 * count).reshape(count, 2)
    if not torch.allclose(computed_mass, slot_mass[0], atol=2e-5, rtol=2e-5):
        raise ValueError("saved slot masses do not match supplied routes")

    valid = target != 255
    if bool((valid & ((target < 0) | (target > 3))).any()):
        raise ValueError("target must contain only 0..3 or ignore 255")
    safe = torch.where(valid, target, 0).long()
    onehot = F.one_hot(safe, 4).permute(0, 3, 1, 2).float() * valid[:, None]
    grid_gt = F.interpolate(onehot, size=(h, w), mode="area")[0].flatten(1)
    label_sum = _pool(grid_gt, route_q[0].float(), route_ids, 2 * count).reshape(count, 2, 4)
    label_mass = label_sum.sum(-1)
    query_valid = (label_mass > 0) & slot_valid[0].bool()
    query_labels = label_sum / torch.where(label_mass > 0, label_mass, torch.ones_like(label_mass))[..., None]
    parent_label_sum = _pool(grid_gt, q[0].float(), ids, count)
    parent_label_mass = parent_label_sum.sum(-1)

    bg_grid = grid_gt[0].reshape(1, 1, h, w) > 0
    fg_grid = grid_gt[1:].sum(0).reshape(1, 1, h, w) > 0
    boundary_grid = fg_grid & (F.max_pool2d(bg_grid.float(), 3, stride=1, padding=1) > 0)
    full_boundary = F.interpolate(boundary_grid.float(), size=target.shape[-2:], mode="nearest")[0, 0].bool()
    gt_building = valid[0] & (target[0] > 0)
    full_b = F.interpolate(building.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    masks = {
        "all_labeled": valid[0],
        "gt_building": gt_building,
        "gt_building_boundary": gt_building & full_boundary,
        "gt_building_interior": gt_building & ~full_boundary,
        "low_B_true_building": gt_building & (full_b < 0.5),
        "gt_intact": valid[0] & (target[0] == 1),
        "gt_damaged": valid[0] & (target[0] == 2),
        "gt_destroyed": valid[0] & (target[0] == 3),
    }
    mask_names = list(masks)
    mask_grid = F.interpolate(torch.stack([masks[k] for k in mask_names])[None].float(),
                              size=(h, w), mode="area")[0].flatten(1)
    pooled_masks = _pool(mask_grid, q[0].float(), ids, count)
    weights = {name: pooled_masks[:, i] for i, name in enumerate(mask_names)}

    parent_mass = slot_mass[0].float().sum(-1)
    parent_active = parent_mass > 0
    b_grid = F.interpolate(building.float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].flatten()
    b_stats = _pool(torch.stack([b_grid, b_grid.square()]), q[0].float(), ids, count)
    b_stats /= torch.where(parent_active, parent_mass, torch.ones_like(parent_mass))[:, None]
    b_mean = _masked(b_stats[:, 0], parent_active)
    b_var = _masked((b_stats[:, 1] - b_stats[:, 0].square()).clamp_min(0), parent_active)
    means = slot_f[0].float()
    norm = torch.linalg.vector_norm(means, dim=-1)
    both_slots = slot_valid[0].bool().all(-1)
    both_nonzero = both_slots & (norm > 0).all(-1)
    cosine = _masked((means[:, 0] * means[:, 1]).sum(-1) /
                     (norm[:, 0] * norm[:, 1]).clamp_min(torch.finfo(means.dtype).tiny), both_nonzero).clamp(-1, 1)
    rel_l2 = _masked(torch.linalg.vector_norm(means[:, 0] - means[:, 1], dim=-1) /
                     (0.5 * norm.sum(-1)).clamp_min(torch.finfo(means.dtype).tiny), both_slots & (norm.sum(-1) > 0))
    p, r = query_labels[:, 0], query_labels[:, 1]
    midpoint = 0.5 * (p + r)
    eps = torch.finfo(p.dtype).tiny
    js = 0.5 * ((p * (p.clamp_min(eps).log2() - midpoint.clamp_min(eps).log2())).sum(-1)
                + (r * (r.clamp_min(eps).log2() - midpoint.clamp_min(eps).log2())).sum(-1))
    both_labels = query_valid.all(-1)
    tv = _masked(0.5 * (p - r).abs().sum(-1), both_labels)
    js = _masked(js.clamp_min(0), both_labels)
    effective = _ess(route_q[0].float(), route_ids, 2 * count).reshape(count, 2)
    parent_ess = _ess(q[0].float(), ids, count)
    rows = {
        "both_slots_active": _masked(both_slots.float(), parent_active),
        "feature_cosine": cosine,
        "feature_relative_l2": rel_l2,
        "label_total_variation": tv,
        "label_js_bits": js,
        "feature_cosine_gt_0_99": _masked((cosine > 0.99).float(), torch.isfinite(cosine)),
        "label_tv_lt_0_05": _masked((tv < 0.05).float(), torch.isfinite(tv)),
        "parent_mass": parent_mass,
        "parent_ess": parent_ess,
        "parent_valid_label_mass": parent_label_mass,
        "parent_gt_foreground_fraction": _masked(parent_label_sum[:, 1:].sum(-1) /
                                                 parent_label_mass.clamp_min(eps), parent_label_mass > 0),
        "building_probability_mean": b_mean,
        "building_probability_variance": b_var,
    }
    for role in range(2):
        prefix = "slot" + str(role) + "_"
        rows[prefix + "mass"] = slot_mass[0, :, role].float()
        rows[prefix + "fraction"] = _masked(slot_fraction[0, :, role].float(), parent_active)
        rows[prefix + "ess"] = effective[:, role]
        rows[prefix + "active"] = _masked(slot_valid[0, :, role].float(), parent_active)
        rows[prefix + "ess_lt_2"] = _masked((effective[:, role] < 2).float(), torch.isfinite(effective[:, role]))
        rows[prefix + "positive_fraction_lt_0_01"] = _masked(
            ((slot_fraction[0, :, role] > 0) & (slot_fraction[0, :, role] < 0.01)).float(), parent_active)
        rows[prefix + "valid_label_mass"] = label_mass[:, role]
        rows[prefix + "gt_foreground_fraction"] = _masked(query_labels[:, role, 1:].sum(-1), query_valid[:, role])
        rows[prefix + "gt_damaged_fraction"] = _masked(query_labels[:, role, 2], query_valid[:, role])
    return {"region_rows": rows, "weights": weights, "pixel_masks": masks,
            "query_labels": query_labels, "query_valid": query_valid,
            "query_label_mass": label_mass}


class RegionAccumulator:
    """Mass-weighted aggregates, with an independent valid denominator per field."""

    def __init__(self) -> None:
        self.images = 0
        self.regions = 0
        self._fields: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)

    @torch.no_grad()
    def add(self, rows: Mapping[str, Tensor], weights: Mapping[str, Tensor]) -> None:
        if not rows or not weights:
            raise ValueError("rows and stratum weights must be nonempty")
        names, strata = list(rows), list(weights)
        metrics = torch.stack([rows[k].detach().double() for k in names])
        masses = torch.stack([weights[k].detach().double() for k in strata])
        if metrics.ndim != 2 or masses.ndim != 2 or metrics.shape[1] != masses.shape[1]:
            raise ValueError("each metric and weight must have shape [R]")
        if masses.device != metrics.device or not bool(torch.isfinite(masses).all()) or bool((masses < 0).any()):
            raise ValueError("weights must be finite nonnegative and share metric device")
        finite = torch.isfinite(metrics)
        safe = torch.where(finite, metrics, torch.zeros_like(metrics))
        weighted_sum = (safe @ masses.T).cpu().tolist()
        weight_sum = (finite.double() @ masses.T).cpu().tolist()
        counts = (finite.double() @ (masses > 0).double().T).cpu().tolist()
        self.images += 1
        self.regions += metrics.shape[1]
        for i, metric in enumerate(names):
            for j, stratum in enumerate(strata):
                row = self._fields[stratum].setdefault(metric, {"n": 0, "weight_sum": 0.0, "weighted_sum": 0.0})
                row["n"] += int(counts[i][j])
                row["weight_sum"] += weight_sum[i][j]
                row["weighted_sum"] += weighted_sum[i][j]

    def summary(self) -> dict:
        return {"images": self.images, "regions_seen": self.regions,
                "weight_units": "GT mass in feature-grid cells; nonexclusive strata",
                "strata": {stratum: {name: {**row, "mean": row["weighted_sum"] / row["weight_sum"]
                                           if row["weight_sum"] > 0 else None}
                                      for name, row in fields.items()}
                           for stratum, fields in self._fields.items()}}

    report = summary
