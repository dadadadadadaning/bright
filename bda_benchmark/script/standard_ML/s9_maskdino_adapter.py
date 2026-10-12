"""Compatibility entry for BRIGHT MaskDINO integration helpers."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.maskdino import (
    S8MultiScaleFeatureAdapter,
    compute_query_budget,
    instances_to_maskdino_target,
)

__all__ = [
    "S8MultiScaleFeatureAdapter",
    "compute_query_budget",
    "instances_to_maskdino_target",
]
