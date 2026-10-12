"""Isolated Stage-0 audit for semantic-to-pseudo-instance research."""

from .core import (
    STAGE0_VARIANTS,
    build_stage0_decision,
    building_histogram_from_map,
    building_metrics_from_histogram,
    damage_class_components,
    evaluate_geometry,
    pooled_instance_raster,
)

__all__ = [
    "STAGE0_VARIANTS",
    "build_stage0_decision",
    "building_histogram_from_map",
    "building_metrics_from_histogram",
    "damage_class_components",
    "evaluate_geometry",
    "pooled_instance_raster",
]

