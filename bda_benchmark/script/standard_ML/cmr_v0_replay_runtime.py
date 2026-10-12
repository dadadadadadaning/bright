# -*- coding: utf-8 -*-
"""Pure geometry helpers and gates for the CMR-V0S0 replay audit."""
from __future__ import annotations

from typing import Mapping, Sequence


def bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = map(int, first)
    bx1, by1, bx2, by2 = map(int, second)
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1)


def replay_decision(
    *,
    exact_rate: float,
    guarded_identity_rate: float,
    minimum_exact_rate: float = 0.99,
    minimum_guarded_rate: float = 0.99,
) -> tuple[dict[str, bool], str]:
    checks = {
        "exact_component_rate_at_least_declared": exact_rate >= minimum_exact_rate,
        "guarded_identity_rate_at_least_declared": guarded_identity_rate >= minimum_guarded_rate,
    }
    if checks["exact_component_rate_at_least_declared"]:
        decision = "ADMIT_TO_V0_TOKEN_CACHE_EXACT_REPLAY"
    elif checks["guarded_identity_rate_at_least_declared"]:
        decision = "REBUILD_P0_M3_V0_IN_ONE_RUNTIME_DO_NOT_FUZZY_MATCH_OLD_CACHE"
    else:
        decision = "STOP_CACHE_RUNTIME_DIVERGENCE_REBUILD_AND_REAUDIT"
    return checks, decision


def guarded_identity(row: Mapping[str, float]) -> bool:
    return bool(
        float(row.get("bbox_iou", 0.0)) >= 0.99
        and float(row.get("relative_area_error", 1.0)) <= 0.01
        and float(row.get("mean_key_cosine", -1.0)) >= 0.999
        and float(row.get("probability_max_abs_delta", 1.0)) <= 1e-5
    )

