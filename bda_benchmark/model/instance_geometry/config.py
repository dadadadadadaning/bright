"""Locked and ablation-friendly configuration for S9 instance geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class GeometryDecoderConfig:
    """Decoder settings with independent watershed/gate/rollback switches."""

    name: str = "S9-Ic-G1"
    building_threshold: float = 0.50
    peak_threshold: float = 0.35
    nms_kernel: int = 3
    min_peak_distance_native: int = 3
    max_markers_per_region: int = 4
    min_region_area_to_split: int = 256
    min_instance_area: int = 32
    watershed_enabled: bool = True
    boundary_gate_enabled: bool = False
    rollback_enabled: bool = False
    boundary_gate_threshold: float = 0.35
    rollback_min_child_ratio: float = 0.05
    fallback_to_region_max: bool = True
    watershed_boundary_weight: float = 0.50
    offset_assignment_radius: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.building_threshold <= 1.0:
            raise ValueError("building_threshold must be in [0,1]")
        if not 0.0 <= self.peak_threshold <= 1.0:
            raise ValueError("peak_threshold must be in [0,1]")
        if self.nms_kernel < 1 or self.nms_kernel % 2 == 0:
            raise ValueError("nms_kernel must be a positive odd integer")
        if self.min_peak_distance_native < 0:
            raise ValueError("min_peak_distance_native must be non-negative")
        if self.max_markers_per_region < 1:
            raise ValueError("max_markers_per_region must be positive")
        if self.min_region_area_to_split < 1 or self.min_instance_area < 1:
            raise ValueError("minimum areas must be positive")
        if not 0.0 <= self.boundary_gate_threshold <= 1.0:
            raise ValueError("boundary_gate_threshold must be in [0,1]")
        if not 0.0 <= self.rollback_min_child_ratio < 0.5:
            raise ValueError("rollback_min_child_ratio must be in [0,0.5)")
        if self.watershed_boundary_weight < 0:
            raise ValueError("watershed_boundary_weight must be non-negative")
        if self.offset_assignment_radius <= 0:
            raise ValueError("offset_assignment_radius must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs: Any) -> "GeometryDecoderConfig":
        return replace(self, **kwargs)


def geometry_preset(name: str) -> GeometryDecoderConfig:
    """Return the locked G1/G2/G3/S9-Id decoder preset."""

    key = name.strip().lower().replace("_", "-")
    if key in {"g1", "s9-ic-g1", "s9ic-g1"}:
        return GeometryDecoderConfig(
            name="S9-Ic-G1",
            watershed_enabled=True,
            boundary_gate_enabled=False,
            rollback_enabled=False,
        )
    if key in {"g2", "s9-ic-g2", "s9ic-g2"}:
        return GeometryDecoderConfig(
            name="S9-Ic-G2",
            watershed_enabled=True,
            boundary_gate_enabled=True,
            rollback_enabled=False,
        )
    if key in {"g3", "s9-ic-g3", "s9ic-g3"}:
        return GeometryDecoderConfig(
            name="S9-Ic-G3",
            watershed_enabled=True,
            boundary_gate_enabled=True,
            rollback_enabled=True,
        )
    if key in {"id", "s9-id", "center-offset"}:
        return GeometryDecoderConfig(
            name="S9-Id",
            watershed_enabled=False,
            boundary_gate_enabled=False,
            rollback_enabled=False,
        )
    raise KeyError(f"Unknown geometry preset: {name!r}")
