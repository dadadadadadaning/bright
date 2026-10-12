"""Predicted-building instances and adaptive same-building region candidates.

No GT enters the forward partition path.  Predicted building instances are
constructed from the frozen A1 building probability and are used as a hard
constraint: a building pixel can only associate with region seeds owned by the
same predicted building.  Background is assigned to token 0 (NULL).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def predicted_building_instances(
    p_building: torch.Tensor,
    size: Tuple[int, int],
    threshold: float = 0.50,
    min_area: int = 8,
) -> torch.Tensor:
    """Return 4-connected predicted building IDs at ``size``.

    IDs are compact per sample and start at 1.  0 is background.  The operation
    is intentionally detached/non-differentiable; the frozen A1 localization
    head provides the building prior and GT is never consulted.
    """
    prob = F.interpolate(
        p_building.detach(), size=size, mode="bilinear", align_corners=False
    )[:, 0]
    outputs: List[torch.Tensor] = []
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    try:
        from scipy import ndimage  # type: ignore
    except Exception:
        ndimage = None

    for sample in prob.float().cpu().numpy():
        binary = sample >= float(threshold)
        if ndimage is not None:
            labels, n = ndimage.label(binary, structure=structure)
        else:
            labels, n = _label_4_connected(binary)
        compact = np.zeros_like(labels, dtype=np.int64)
        new_id = 1
        for old_id in range(1, int(n) + 1):
            mask = labels == old_id
            if int(mask.sum()) < int(min_area):
                continue
            compact[mask] = new_id
            new_id += 1
        outputs.append(torch.from_numpy(compact))
    return torch.stack(outputs).to(device=p_building.device)


def _label_4_connected(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """Tiny scipy-free 4-connected fallback for preflight portability."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int64)
    current = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


def _choose_grid(
    bbox_h: int,
    bbox_w: int,
    target_k: int,
    max_k: int,
) -> Tuple[int, int]:
    """Choose a compact grid with at most ``max_k`` cells and near ``target_k``."""
    limit = max(1, int(max_k))
    target_k = max(1, min(int(target_k), limit))
    aspect = max(float(bbox_h), 1.0) / max(float(bbox_w), 1.0)
    best = None
    for gh in range(1, limit + 1):
        for gw in range(1, limit + 1):
            k = gh * gw
            if k > limit:
                continue
            grid_aspect = gh / max(gw, 1)
            count_cost = abs(k - target_k)
            aspect_cost = abs(np.log((grid_aspect + 1e-6) / (aspect + 1e-6)))
            # Region count is primary; shape regularity breaks ties / near-ties.
            cost = count_cost + 0.20 * aspect_cost
            candidate = (cost, -k if k <= target_k else k, gh, gw)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return int(best[2]), int(best[3])


def _nearest_mask_pixel(coords: np.ndarray, y: float, x: float) -> Tuple[int, int]:
    dist = (coords[:, 0] - y) ** 2 + (coords[:, 1] - x) ** 2
    index = int(dist.argmin())
    return int(coords[index, 0]), int(coords[index, 1])


@torch.no_grad()
def adaptive_building_candidates(
    building_ids: torch.Tensor,
    *,
    target_region_area: int = 16,
    min_region_area: int = 4,
    max_regions_per_building: int = 16,
) -> Dict[str, torch.Tensor]:
    """Build per-building adaptive grid seeds and same-building 3x3 candidates.

    Returns padded token metadata.  Token 0 is NULL/background. Candidate IDs
    are Bx9xN because different samples have different region layouts.
    """
    if building_ids.ndim != 3:
        raise ValueError(f"building_ids must be BxHxW, got {tuple(building_ids.shape)}")
    bsz, h, w = building_ids.shape
    device = building_ids.device
    cpu = building_ids.detach().cpu().numpy().astype(np.int64)

    per_sample: List[Dict[str, object]] = []
    max_tokens = 1
    for sample in cpu:
        candidate_ids = np.zeros((9, h * w), dtype=np.int64)
        candidate_valid = np.zeros((9, h * w), dtype=np.bool_)
        candidate_distance2 = np.zeros((9, h * w), dtype=np.float32)
        # Background has one deterministic NULL candidate.
        bg_flat = np.flatnonzero(sample.reshape(-1) == 0)
        candidate_valid[0, bg_flat] = True

        local_xy = np.zeros((2, h, w), dtype=np.float32)
        token_building = [0]
        token_building_area = [0.0]
        token_target_mass = [0.0]
        token_seed_xy = [(0.0, 0.0)]
        token_cursor = 1

        building_values = [int(v) for v in np.unique(sample) if int(v) > 0]
        for building_id in building_values:
            coords = np.argwhere(sample == building_id)
            area = int(coords.shape[0])
            if area <= 0:
                continue
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            bh, bw = int(y1 - y0 + 1), int(x1 - x0 + 1)

            desired = max(1, int(ceil(area / max(1, int(target_region_area)))))
            by_min_area = max(1, area // max(1, int(min_region_area)))
            limit = max(1, min(int(max_regions_per_building), by_min_area))
            gh, gw = _choose_grid(bh, bw, desired, limit)
            k_b = gh * gw
            base = token_cursor
            token_cursor += k_b

            # Every grid cell owns one seed, snapped to the nearest building pixel.
            seeds: List[Tuple[int, int]] = []
            for gy in range(gh):
                for gx in range(gw):
                    cy = y0 + (gy + 0.5) * bh / gh
                    cx = x0 + (gx + 0.5) * bw / gw
                    in_cell = coords[
                        (coords[:, 0] >= y0 + gy * bh / gh)
                        & (coords[:, 0] < y0 + (gy + 1) * bh / gh + 1e-6)
                        & (coords[:, 1] >= x0 + gx * bw / gw)
                        & (coords[:, 1] < x0 + (gx + 1) * bw / gw + 1e-6)
                    ]
                    source = in_cell if len(in_cell) else coords
                    sy, sx = _nearest_mask_pixel(source, cy, cx)
                    seeds.append((sy, sx))
                    token_building.append(building_id)
                    token_building_area.append(float(area))
                    token_target_mass.append(float(area) / float(k_b))
                    token_seed_xy.append((float(sy), float(sx)))

            for y, x in coords.tolist():
                # Building-local position in [-1, 1].
                local_xy[0, y, x] = 2.0 * ((x - x0 + 0.5) / max(float(bw), 1.0)) - 1.0
                local_xy[1, y, x] = 2.0 * ((y - y0 + 0.5) / max(float(bh), 1.0)) - 1.0
                gy0 = min(gh - 1, max(0, int((y - y0) * gh / max(bh, 1))))
                gx0 = min(gw - 1, max(0, int((x - x0) * gw / max(bw, 1))))
                flat = y * w + x
                slot = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        gy = gy0 + dy
                        gx = gx0 + dx
                        if 0 <= gy < gh and 0 <= gx < gw:
                            local_seed = gy * gw + gx
                            token_id = base + local_seed
                            sy, sx = seeds[local_seed]
                            norm_y = (y - sy) / max(float(bh), 1.0)
                            norm_x = (x - sx) / max(float(bw), 1.0)
                            candidate_ids[slot, flat] = token_id
                            candidate_valid[slot, flat] = True
                            candidate_distance2[slot, flat] = norm_y * norm_y + norm_x * norm_x
                        slot += 1
                # Defensive fallback for very small/degenerate grids.
                if not candidate_valid[:, flat].any():
                    candidate_ids[0, flat] = base
                    candidate_valid[0, flat] = True

        max_tokens = max(max_tokens, token_cursor)
        per_sample.append(
            {
                "candidate_ids": candidate_ids,
                "candidate_valid": candidate_valid,
                "candidate_distance2": candidate_distance2,
                "local_xy": local_xy,
                "token_building": np.asarray(token_building, dtype=np.int64),
                "token_building_area": np.asarray(token_building_area, dtype=np.float32),
                "token_target_mass": np.asarray(token_target_mass, dtype=np.float32),
                "token_seed_xy": np.asarray(token_seed_xy, dtype=np.float32),
                "token_count": token_cursor,
            }
        )

    ids = torch.zeros((bsz, 9, h * w), dtype=torch.long, device=device)
    valid = torch.zeros((bsz, 9, h * w), dtype=torch.bool, device=device)
    distance2 = torch.zeros((bsz, 9, h * w), dtype=torch.float32, device=device)
    local_xy = torch.zeros((bsz, 2, h, w), dtype=torch.float32, device=device)
    token_building = torch.zeros((bsz, max_tokens), dtype=torch.long, device=device)
    token_building_area = torch.zeros((bsz, max_tokens), dtype=torch.float32, device=device)
    token_target_mass = torch.zeros((bsz, max_tokens), dtype=torch.float32, device=device)
    token_seed_xy = torch.zeros((bsz, max_tokens, 2), dtype=torch.float32, device=device)
    token_count_per_sample = torch.zeros((bsz,), dtype=torch.long, device=device)

    for bi, item in enumerate(per_sample):
        count = int(item["token_count"])
        ids[bi] = torch.from_numpy(item["candidate_ids"]).to(device)
        valid[bi] = torch.from_numpy(item["candidate_valid"]).to(device)
        distance2[bi] = torch.from_numpy(item["candidate_distance2"]).to(device)
        local_xy[bi] = torch.from_numpy(item["local_xy"]).to(device)
        token_building[bi, :count] = torch.from_numpy(item["token_building"]).to(device)
        token_building_area[bi, :count] = torch.from_numpy(item["token_building_area"]).to(device)
        token_target_mass[bi, :count] = torch.from_numpy(item["token_target_mass"]).to(device)
        token_seed_xy[bi, :count] = torch.from_numpy(item["token_seed_xy"]).to(device)
        token_count_per_sample[bi] = count

    return {
        "candidate_ids": ids,
        "candidate_valid": valid,
        "candidate_distance2": distance2,
        "building_local_xy": local_xy,
        "token_building_ids": token_building,
        "token_building_area": token_building_area,
        "token_target_mass": token_target_mass,
        "token_seed_xy": token_seed_xy,
        "token_count_per_sample": token_count_per_sample,
        "token_count": torch.tensor(max_tokens, dtype=torch.long, device=device),
    }


def assert_same_building_candidates(
    building_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    token_building_ids: torch.Tensor,
) -> None:
    """Raise if a valid non-background candidate belongs to another building."""
    bsz, h, w = building_ids.shape
    n = h * w
    flat_building = building_ids.reshape(bsz, n)
    for j in range(candidate_ids.shape[1]):
        ids = candidate_ids[:, j]
        owners = token_building_ids.gather(1, ids)
        valid = candidate_valid[:, j]
        mismatch = valid & (flat_building != owners)
        # background -> NULL is valid and equal (0 == 0)
        if bool(mismatch.any().item()):
            where = torch.nonzero(mismatch, as_tuple=False)[0].tolist()
            raise AssertionError(f"cross-building candidate detected at batch/pixel={where}")


__all__ = [
    "predicted_building_instances",
    "adaptive_building_candidates",
    "assert_same_building_candidates",
]
