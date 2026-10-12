"""Compatibility entry for S9-Ic/S9-Id target generation and losses."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.losses import (
    boundary_bce_dice_loss,
    centernet_focal_loss,
    instance_balanced_offset_loss,
    reliable_geometry_loss,
)
from bda_benchmark.model.instance_geometry.targets import (
    build_center_offset_targets,
    build_unique_gaussian_targets,
    deterministic_edt_center,
    map_point_to_size,
)

__all__ = [
    "boundary_bce_dice_loss",
    "build_center_offset_targets",
    "build_unique_gaussian_targets",
    "centernet_focal_loss",
    "deterministic_edt_center",
    "instance_balanced_offset_loss",
    "map_point_to_size",
    "reliable_geometry_loss",
]
