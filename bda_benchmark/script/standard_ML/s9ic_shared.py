"""Shared S9-Ic/S9-Id model construction and checkpoint metadata."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.DamageFormer_S9Ic_ReliableGeometry import (
    ReliableGeometryHead,
)
from bda_benchmark.model.DamageFormer_S9Id_CenterOffset import (
    CenterOffsetGeometryHead,
)
from bda_benchmark.model.instance_geometry.s8_backbone import FrozenS8Backbone


EXPECTED_FEATURE_KEY = "fused_roi"
EXPECTED_FEATURE_CHANNELS = 256
TARGET_DEFINITION = "source_scale_deterministic_edt_v2"
GAUSSIAN_SIGMA_RULE = "clip(sqrt(native_area/pi)/3,1,3)"


def build_geometry_head(
    variant: str,
    *,
    in_channels: int = EXPECTED_FEATURE_CHANNELS,
    hidden_channels: int = 64,
):
    key = variant.strip().lower().replace("_", "-")
    if key in {"g1", "g2", "g3", "s9-ic-g1", "s9-ic-g2", "s9-ic-g3"}:
        return ReliableGeometryHead(in_channels, hidden_channels)
    if key in {"id", "s9-id", "center-offset"}:
        return CenterOffsetGeometryHead(in_channels, hidden_channels)
    raise KeyError(f"Unknown geometry variant: {variant!r}")


def load_frozen_s8(
    checkpoint_path: str,
    *,
    allow_unsafe: bool = False,
) -> FrozenS8Backbone:
    return FrozenS8Backbone.from_checkpoint(
        checkpoint_path,
        feature_key=EXPECTED_FEATURE_KEY,
        expected_channels=EXPECTED_FEATURE_CHANNELS,
        allow_unsafe=allow_unsafe,
    )


def checkpoint_metadata(variant: str, step: int) -> Mapping[str, Any]:
    key = variant.strip().lower().replace("_", "-")
    is_g1 = key in {"g1", "s9-ic-g1"}
    return {
        "variant": variant,
        "step": int(step),
        "feature_key": EXPECTED_FEATURE_KEY,
        "target_definition": TARGET_DEFINITION,
        "gaussian_sigma_rule": GAUSSIAN_SIGMA_RULE,
        "s8_strict_loaded": True,
        "s8_frozen": True,
        "boundary_head_trained": True,
        "boundary_permission_enabled_in_g1": False if is_g1 else None,
        "rollback_enabled_in_g1": False if is_g1 else None,
        "final_selection_locked": False,
    }
