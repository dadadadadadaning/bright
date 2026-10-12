"""Instance-geometry components shared by S9-Ic, S9-Id, and MaskDINO adapters."""

from .config import GeometryDecoderConfig, geometry_preset
from .decode import (
    decode_center_offset_instances,
    decode_center_watershed_instances,
    extract_region_peaks,
    native_local_maxima,
)
from .losses import (
    centernet_focal_loss,
    instance_balanced_offset_loss,
    reliable_geometry_loss,
)
from .maskdino import (
    S8MultiScaleFeatureAdapter,
    compute_query_budget,
    instances_to_maskdino_target,
)
from .metrics import CenterMetricAccumulator, evaluate_center_mechanism
from .s8_backbone import FrozenS8Backbone, FrozenS8Output
from .targets import (
    build_center_offset_targets,
    build_unique_gaussian_targets,
    deterministic_edt_center,
    map_point_to_size,
)

__all__ = [
    "CenterMetricAccumulator",
    "FrozenS8Backbone",
    "FrozenS8Output",
    "GeometryDecoderConfig",
    "S8MultiScaleFeatureAdapter",
    "build_center_offset_targets",
    "build_unique_gaussian_targets",
    "centernet_focal_loss",
    "compute_query_budget",
    "decode_center_offset_instances",
    "decode_center_watershed_instances",
    "deterministic_edt_center",
    "evaluate_center_mechanism",
    "extract_region_peaks",
    "geometry_preset",
    "instance_balanced_offset_loss",
    "instances_to_maskdino_target",
    "native_local_maxima",
    "map_point_to_size",
    "reliable_geometry_loss",
]
