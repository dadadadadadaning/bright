"""Compatibility entry for S9-Id Center+Offset targets, loss, and decoder."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.DamageFormer_S9Id_CenterOffset import (
    CenterOffsetGeometryHead,
)
from bda_benchmark.model.instance_geometry.config import GeometryDecoderConfig
from bda_benchmark.model.instance_geometry.decode import (
    decode_center_offset_instances,
)
from bda_benchmark.model.instance_geometry.losses import (
    instance_balanced_offset_loss,
    reliable_geometry_loss,
)
from bda_benchmark.model.instance_geometry.targets import (
    build_center_offset_targets,
)

__all__ = [
    "CenterOffsetGeometryHead",
    "GeometryDecoderConfig",
    "build_center_offset_targets",
    "decode_center_offset_instances",
    "instance_balanced_offset_loss",
    "reliable_geometry_loss",
]
