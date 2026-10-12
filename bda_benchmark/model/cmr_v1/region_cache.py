# -*- coding: utf-8 -*-
"""Compact, identity-frozen predicted regions for isolated CMR-V1.

The cache stores the *frozen-P3 predicted component*, never a GT mask.  Masks
are cropped to their bounding box and bit-packed so a few thousand building
instances do not require dense 640x640 tensors.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch


REGION_CACHE_PROTOCOL = "CMR_V1_ISOLATED_frozen_predicted_region_cache_v1"


def pack_component(component: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int]]:
    mask = np.asarray(component, dtype=np.bool_)
    if mask.ndim != 2 or not bool(mask.any()):
        raise ValueError("component must be a non-empty 2-D mask")
    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    local = mask[y1:y2, x1:x2]
    packed = np.packbits(local.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy()), (x1, y1, x2, y2), tuple(map(int, mask.shape))


def unpack_component(
    packed: torch.Tensor | np.ndarray,
    bbox: Sequence[int],
    image_shape: Sequence[int],
) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    height, width = (int(value) for value in image_shape)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"invalid bbox {tuple(bbox)} for image shape {(height, width)}")
    count = (x2 - x1) * (y2 - y1)
    raw = packed.detach().cpu().numpy() if isinstance(packed, torch.Tensor) else np.asarray(packed)
    local = np.unpackbits(raw.astype(np.uint8, copy=False), bitorder="little", count=count)
    component = np.zeros((height, width), dtype=np.bool_)
    component[y1:y2, x1:x2] = local.reshape(y2 - y1, x2 - x1).astype(np.bool_)
    return component


def validate_region_cache(payload: Mapping[str, object], role: str | None = None) -> None:
    if payload.get("protocol") != REGION_CACHE_PROTOCOL:
        raise RuntimeError("unknown CMR-V1 region-cache protocol")
    if role is not None and payload.get("role") != role:
        raise RuntimeError(f"region-cache role mismatch: {payload.get('role')} != {role}")
    if bool(payload.get("test_accessed", True)):
        raise RuntimeError("CMR-V1 refuses a Test-derived region cache")
    rows = payload["source_region_indices"]
    bboxes = payload["bboxes"]
    shapes = payload["image_shapes"]
    masks = payload["packed_component_masks"]
    size = int(rows.numel())
    if bboxes.shape != (size, 4) or shapes.shape != (size, 2) or len(masks) != size:
        raise RuntimeError("region-cache arrays do not align")
    if int(torch.unique(rows.long()).numel()) != size:
        raise RuntimeError("region-cache source rows are not unique")


def source_row_lookup(payload: Mapping[str, object], source_size: int) -> torch.Tensor:
    validate_region_cache(payload)
    rows = payload["source_region_indices"].long()
    if bool((rows < 0).any()) or (rows.numel() and int(rows.max()) >= int(source_size)):
        raise RuntimeError("region cache contains an out-of-range P0 source row")
    lookup = torch.full((int(source_size),), -1, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long)
    return lookup


__all__ = [
    "REGION_CACHE_PROTOCOL",
    "pack_component",
    "source_row_lookup",
    "unpack_component",
    "validate_region_cache",
]
