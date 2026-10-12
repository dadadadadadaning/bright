# -*- coding: utf-8 -*-
"""Identity-safe replay of frozen-P3 connected building components for CMR-V1.

The historical P0 cache stores bbox/area/Raw-Mean key, but not the full binary
component mask.  An exact bbox+area replay is attempted first.  If CUDA/model
numerics move a boundary by a few pixels, a fallback is accepted ONLY when the
candidate is uniquely supported by both geometry and the original cached
Raw-Mean embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from model.DamageFormer_S8_MICB_DSMIL import connected_components_4
from script.standard_ML.p3_hard_occurrence_runtime import component_bbox


@dataclass(frozen=True)
class ReplayMatch:
    component: np.ndarray
    mode: str
    bbox_iou: float
    area_ratio: float
    key_cosine: float
    key_margin: float
    candidate_count: int


def build_component_index(
    components: Sequence[np.ndarray],
) -> dict[tuple[tuple[int, int, int, int], int], list[np.ndarray]]:
    """Index one image's components once for O(1) exact P0 identity lookup."""
    index: dict[tuple[tuple[int, int, int, int], int], list[np.ndarray]] = {}
    for component in components:
        key = (component_bbox(component), int(component.sum()))
        index.setdefault(key, []).append(component)
    return index


def bbox_iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = (int(x) for x in a)
    bx1, by1, bx2, by2 = (int(x) for x in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def symmetric_area_ratio(a: int, b: int) -> float:
    a, b = int(a), int(b)
    if a <= 0 or b <= 0:
        return 0.0
    return float(min(a, b)) / float(max(a, b))


def replay_cached_component(
    building_mask: torch.Tensor | np.ndarray,
    expected_bbox: Sequence[int],
    expected_area: int,
    expected_key: torch.Tensor,
    key_fn: Callable[[np.ndarray], torch.Tensor],
    *,
    precomputed_components: Sequence[np.ndarray] | None = None,
    exact_index: Mapping[
        tuple[tuple[int, int, int, int], int], Sequence[np.ndarray]
    ] | None = None,
    min_bbox_iou: float = 0.90,
    min_area_ratio: float = 0.90,
    min_key_cosine: float = 0.995,
    min_key_margin: float = 0.001,
) -> ReplayMatch | None:
    """Replay one cached P0 component without silently swapping instances.

    Exact bbox+area has priority.  The verified fallback is deliberately strict:
      * geometry must strongly overlap the historical bbox/area;
      * the recomputed Raw-Mean key must nearly equal the cached P0 key;
      * if multiple candidates survive, the best key match must be separated.
    """

    expected_bbox = tuple(int(x) for x in expected_bbox)
    expected_area = int(expected_area)
    components = (
        list(precomputed_components)
        if precomputed_components is not None
        else connected_components_4(building_mask)
    )

    exact_candidates = (
        list(exact_index.get((expected_bbox, expected_area), ()))
        if exact_index is not None else [
            component for component in components
            if int(component.sum()) == expected_area and component_bbox(component) == expected_bbox
        ]
    )
    if len(exact_candidates) == 1:
        component = exact_candidates[0]
        return ReplayMatch(
            component=component,
            mode="exact_bbox_area",
            bbox_iou=1.0,
            area_ratio=1.0,
            key_cosine=1.0,
            key_margin=1.0,
            candidate_count=1,
        )
    # More than one exact candidate should be impossible for disjoint connected
    # components.  If it ever occurs, deliberately fall through to the normal
    # key-aware ambiguity check below instead of choosing the first component.

    expected_key = F.normalize(expected_key.detach().float().flatten(), dim=0, eps=1e-8)
    scored: list[tuple[float, float, float, np.ndarray]] = []
    for component in components:
        area = int(component.sum())
        iou = bbox_iou_xyxy(component_bbox(component), expected_bbox)
        area_ratio = symmetric_area_ratio(area, expected_area)
        if iou < float(min_bbox_iou) or area_ratio < float(min_area_ratio):
            continue
        current_key = F.normalize(key_fn(component).detach().float().flatten(), dim=0, eps=1e-8)
        if current_key.numel() != expected_key.numel():
            raise RuntimeError(
                f"Raw-Mean key dimension mismatch: current={current_key.numel()} "
                f"cached={expected_key.numel()}"
            )
        cosine = float(torch.dot(current_key.cpu(), expected_key.cpu()).item())
        scored.append((cosine, iou, area_ratio, component))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_cos, best_iou, best_area_ratio, best_component = scored[0]
    second_cos = scored[1][0] if len(scored) > 1 else -1.0
    margin = float(best_cos - second_cos) if len(scored) > 1 else 2.0

    if best_cos < float(min_key_cosine):
        return None
    if len(scored) > 1 and margin < float(min_key_margin):
        return None

    return ReplayMatch(
        component=best_component,
        mode="verified_geometry_plus_rawmean",
        bbox_iou=float(best_iou),
        area_ratio=float(best_area_ratio),
        key_cosine=float(best_cos),
        key_margin=float(margin),
        candidate_count=len(scored),
    )


__all__ = [
    "ReplayMatch",
    "bbox_iou_xyxy",
    "build_component_index",
    "replay_cached_component",
    "symmetric_area_ratio",
]
