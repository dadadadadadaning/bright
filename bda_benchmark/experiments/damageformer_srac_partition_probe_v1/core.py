"""Isolated partition interventions using an existing front's exact weights.

No GT enters these functions.  Changing a setting here is a fixed-checkpoint
inference intervention, not a trained alternative model or causal ablation.
The legacy path preserves the parent's operation order.  Generic reductions
are necessary because the old center and pooling helpers hard-code 9 slots.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch.nn import functional as F


def _positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


@lru_cache(maxsize=8)
def _candidate_cpu(height: int, width: int, stride: int, radius: int):
    gh = (height + stride - 1) // stride
    gw = (width + stride - 1) // stride
    yy = torch.arange(height, dtype=torch.long)[:, None].expand(height, width)
    xx = torch.arange(width, dtype=torch.long)[None, :].expand(height, width)
    base_y, base_x = yy // stride, xx // stride
    ids, valid = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cy, cx = base_y + dy, base_x + dx
            ok = (cy >= 0) & (cy < gh) & (cx >= 0) & (cx < gw)
            ids.append((cy.clamp(0, gh - 1) * gw + cx.clamp(0, gw - 1)).reshape(-1))
            valid.append(ok.reshape(-1))
    return torch.stack(ids), torch.stack(valid), gh, gw


def candidate_map_radius(height, width, stride, radius, device):
    """Return ids[S,N], validity[S,N], grid_h, grid_w; r=1 is the parent map.

    Wider candidates do NOT create more seeds. Invalid candidates retain safe
    clamped IDs but must receive zero assignment via the validity mask.
    """
    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    stride = _positive_int(stride, "stride")
    if isinstance(radius, bool) or int(radius) != radius or int(radius) < 0:
        raise ValueError("radius must be a nonnegative integer")
    ids, valid, gh, gw = _candidate_cpu(height, width, stride, int(radius))
    return ids.to(device=device), valid.to(device=device), gh, gw


def building_log_prior(pixel_b, center_b, mode="legacy", strength=0.5, leakage=0.1):
    """Broadcast additive building prior without masking any valid candidate.

    ``legacy`` is precisely the parent's compatibility prior, including its
    center clamp. ``bounded`` has finite maximum opposition |lambda*log(eps)|,
    and relaxes both uncertain pixels and uncertain centers. This limits but
    cannot eliminate harm from high-confidence localization mistakes.
    """
    if not math.isfinite(float(strength)) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    center_b = center_b.clamp(1e-4, 1 - 1e-4)
    compatibility = pixel_b * center_b + (1 - pixel_b) * (1 - center_b)
    if mode == "legacy":
        return float(strength) * compatibility.clamp_min(1e-4).log()
    if mode != "bounded":
        raise ValueError("Unknown building prior: " + str(mode))
    if not math.isfinite(float(leakage)) or not 0 < leakage < 1:
        raise ValueError("bounded leakage must be between zero and one")
    confidence = (2 * pixel_b - 1).abs() * (2 * center_b - 1).abs()
    return float(strength) * confidence * (leakage + (1 - leakage) * compatibility).log()


def _centers(values, q, index, count):
    """Differentiable generic-S reduction; r=1 preserves parent arithmetic."""
    batch, _, channels = values.shape
    mass = values.new_zeros(batch, count)
    sums = values.new_zeros(batch, count, channels)
    for candidate in range(q.shape[1]):
        mass = mass.scatter_add(1, index[:, candidate], q[:, candidate])
        sums = sums.scatter_add(
            1, index[:, candidate, :, None].expand(-1, -1, channels),
            values * q[:, candidate, :, None],
        )
    return sums / mass.clamp_min(1e-6)[:, :, None], mass


def _gather(centers, index):
    return centers.gather(1, index[..., None].expand(-1, -1, centers.shape[-1]))


def run_partition(module, embedding, building, *, iterations=None, radius=1,
                  stride=None, prior_mode="legacy", prior_strength=None, leakage=0.1):
    """Replay an existing partition module, varying only specified settings.

    Uses the module's original norm/query/key/logit_scale and never changes its
    attributes, parameters, training mode or buffers. Building is detached as
    in the parent. Q remains differentiable to embedding and partition weights.

    Trace center shifts are in seed-spacing units. q_l1_previous is the mean
    per-pixel L1 difference over candidates (range 0..2). For an unoccupied
    center, a shift is not interpreted. ``entropy_valid_map`` normalizes H(Q)
    by the actually available candidate count. The original entropy_map is
    deliberately H/log(9), even for radius=2, to keep the trained adapter's
    input convention unchanged; it is then NOT bounded by one.
    """
    if embedding.ndim != 4 or building.ndim != 4:
        raise ValueError("Expected embedding[B,C,H,W] and building[B,1,Hb,Wb]")
    batch, _, height, width = embedding.shape
    if building.shape[:2] != (batch, 1) or building.device != embedding.device:
        raise ValueError("Building probability batch/channel/device mismatch")
    iterations = _positive_int(module.iterations if iterations is None else iterations, "iterations")
    stride = _positive_int(module.stride if stride is None else stride, "stride")
    strength = module.building_weight if prior_strength is None else float(prior_strength)
    ids, valid, gh, gw = candidate_map_radius(height, width, stride, radius, embedding.device)
    count, supports = gh * gw, ids.shape[0]
    trace = []
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        pixels = embedding.float().flatten(2).transpose(1, 2)
        pixels = module.norm(pixels)
        query = F.normalize(module.query(pixels), dim=-1)
        values = F.normalize(pixels, dim=-1)
        yy, xx = torch.meshgrid(torch.arange(height, device=embedding.device),
                                torch.arange(width, device=embedding.device), indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2).float()
        coordinates = coordinates.div(float(stride)).expand(batch, -1, -1)
        building_grid = F.interpolate(building.detach().float(), (height, width),
                                      mode="bilinear", align_corners=False)
        building_flat = building_grid.flatten(2).transpose(1, 2)
        index = ids[None].expand(batch, -1, -1)
        assignment = embedding.new_zeros(batch, supports, height * width, dtype=torch.float32)
        assignment[:, supports // 2] = 1.0
        scale = module.logit_scale.exp().clamp(1e-3, 100.0)
        for iteration in range(iterations):
            center_value, mass = _centers(values, assignment, index, count)
            center_xy, _ = _centers(coordinates, assignment, index, count)
            center_building, _ = _centers(building_flat, assignment, index, count)
            scores, allowed = [], []
            for candidate in range(supports):
                candidate_index = index[:, candidate]
                key = F.normalize(module.key(_gather(center_value, candidate_index)), dim=-1)
                similarity = (query * key).sum(-1) * scale
                distance = (coordinates - _gather(center_xy, candidate_index)).square().sum(-1)
                center_b = _gather(center_building, candidate_index).squeeze(-1)
                pixel_b = building_flat.squeeze(-1)
                score = similarity / module.temperature
                score = score - module.spatial_weight * distance
                score = score + building_log_prior(pixel_b, center_b, prior_mode, strength, leakage)
                occupied = mass.gather(1, candidate_index) > 1e-6
                allowed_here = valid[candidate][None] & occupied
                allowed.append(allowed_here)
                scores.append(score.masked_fill(~allowed_here, -torch.inf))
            previous = assignment
            assignment = torch.stack(scores, dim=1).softmax(dim=1)
            available_count = torch.stack(allowed, dim=1).sum(dim=1)
            # Detached observables never alter the computation used by Q.
            with torch.no_grad():
                next_xy, next_mass = _centers(coordinates.detach(), assignment.detach(), index, count)
                common = (mass > 1e-6) & (next_mass > 1e-6)
                shifts = (next_xy - center_xy.detach()).square().sum(-1).sqrt()
                center_shift = shifts[common].mean() if bool(common.any()) else shifts.new_zeros(())
                entropy = -(assignment.detach() * assignment.detach().clamp_min(1e-8).log()).sum(1)
                denominator = available_count.float().clamp_min(2).log()
                normalized_entropy = torch.where(available_count > 1, entropy / denominator, torch.zeros_like(entropy))
                hard_ids = index.gather(1, assignment.argmax(1, keepdim=True)).squeeze(1)
                hard_mass = torch.zeros_like(next_mass).scatter_add_(1, hard_ids, torch.ones_like(hard_ids, dtype=next_mass.dtype))
                entry = {
                    "q_l1_previous": (assignment.detach() - previous.detach()).abs().sum(1).mean(),
                    "center_shift": center_shift,
                    "soft_occupied": (next_mass > 1e-6).float().sum(1).mean(),
                    "hard_occupied": (hard_mass > 0).float().sum(1).mean(),
                    "mass_sum": next_mass.sum(1).mean(),
                    "mass_max": next_mass.amax(),
                    "row_sum_error": (assignment.detach().sum(1) - 1).abs().amax(),
                    "entropy": entropy.mean(),
                    "entropy_valid_normalized": normalized_entropy.mean(),
                    "available_candidates_mean": available_count.float().mean(),
                    "available_candidates_min": available_count.amin(),
                    "available_candidates_max": available_count.amax(),
                    "invalid_assignment_max": assignment.detach().masked_fill(valid[None], 0).amax(),
                }
                # One small CPU transfer per iteration, not one synchronization
                # per diagnostic. Plain numbers make the trace JSON serializable.
                keys = list(entry)
                numbers = torch.stack([entry[key].detach().float() for key in keys]).cpu().tolist()
                trace.append({"iteration": iteration + 1, **dict(zip(keys, numbers))})
        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(1)
        denominator = available_count.float().clamp_min(2).log()
        entropy_valid = torch.where(available_count > 1, entropy / denominator, torch.zeros_like(entropy))
    return {
        "q": assignment, "ids": ids, "shape": (height, width),
        "grid_shape": (gh, gw), "count": count,
        "entropy_map": (entropy / math.log(9.0)).reshape(batch, 1, height, width),
        "entropy_valid_map": entropy_valid.reshape(batch, 1, height, width),
        "candidate_valid": valid, "trace": trace,
    }


def sparse_pool(content, q, ids, count):
    """Generic-S Q-weighted means and SAME-Q reconstruction, no dense KxN."""
    if content.ndim != 4 or q.ndim != 3 or ids.ndim != 2:
        raise ValueError("Expected content[B,C,H,W], Q[B,S,N], ids[S,N]")
    batch, channels, height, width = content.shape
    pixels = height * width
    if q.shape[0] != batch or q.shape[2] != pixels or ids.shape != q.shape[1:]:
        raise ValueError("Association shape does not match content")
    if ids.dtype != torch.long or ids.device != content.device or q.device != content.device:
        raise ValueError("Association IDs must be long and tensors on the same device")
    count = _positive_int(count, "count")
    values = content.flatten(2).transpose(1, 2)
    numerator = content.new_zeros(batch, count, channels)
    mass = content.new_zeros(batch, count, 1)
    for candidate in range(q.shape[1]):
        idx = ids[candidate].view(1, pixels, 1)
        weight = q[:, candidate].unsqueeze(-1)
        numerator.scatter_add_(1, idx.expand(batch, pixels, channels), values * weight)
        mass.scatter_add_(1, idx.expand(batch, pixels, 1), weight)
    tokens = numerator / mass.clamp_min(1e-6)
    reconstruction = content.new_zeros(batch, pixels, channels)
    for candidate in range(q.shape[1]):
        idx = ids[candidate].view(1, pixels, 1).expand(batch, pixels, channels)
        reconstruction.add_(tokens.gather(1, idx) * q[:, candidate].unsqueeze(-1))
    return {"tokens": tokens, "mass": mass.squeeze(-1),
            "reconstruction": reconstruction.transpose(1, 2).reshape(batch, channels, height, width)}


def _geometry(q, ids, shape, count):
    height, width = shape
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=q.device),
                            torch.linspace(0, 1, width, device=q.device), indexing="ij")
    values = torch.stack((yy, xx, yy.square(), xx.square()))[None].expand(q.shape[0], -1, -1, -1)
    pooled = sparse_pool(values.float(), q, ids, count)
    mean = pooled["tokens"][..., :2]
    spread = (pooled["tokens"][..., 2:] - mean.square()).clamp_min(0).sqrt()
    return torch.cat((pooled["mass"][..., None] / float(height * width), mean, spread), dim=-1)


def embedding_with_partition(module, adapted, raw, partition):
    """Replay the existing SuperpixelEmbedding weights for any support count.

    Statistics, attention, geometry, optional context, epsilon and normalization
    all match the parent. Only its hard-coded-nine pooling is generalized.
    """
    q, ids = partition["q"], partition["ids"]
    shape, count = partition["shape"], partition["count"]
    if tuple(adapted.shape[-2:]) != tuple(shape) or tuple(raw.shape[-2:]) != tuple(shape):
        raise ValueError("Region content and partition resolutions must match")
    spatial_weight = module.score(adapted).float().sigmoid().flatten(2)
    main = sparse_pool(adapted.float(), q * spatial_weight, ids, count)
    stats = sparse_pool(torch.cat((raw.float(), raw.float().square()), dim=1), q, ids, count)
    mean, second = stats["tokens"].chunk(2, dim=-1)
    std = (second - mean.square()).clamp_min(0).add(1e-6).sqrt()
    geometry = _geometry(q, ids, shape, count)
    tokens = module.main(main["tokens"])
    tokens = tokens + module.skip_scale.tanh() * module.stats(torch.cat((mean, std), dim=-1))
    tokens = module.norm(tokens + module.geometry(geometry))
    valid = stats["mass"] > 1e-6
    if module.context is not None:
        tokens = module.context(tokens, valid)
    else:
        tokens = tokens.masked_fill(~valid[..., None], 0)
    return tokens, stats["mass"], geometry, valid
