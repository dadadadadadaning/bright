from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass(frozen=True)
class RGIDConfig:
    in_channels: int = 256
    hidden_channels: int = 64
    num_shared_blocks: int = 2

    center_focal_alpha: float = 2.0
    center_focal_beta: float = 4.0
    gaussian_sigma_min: float = 1.0
    gaussian_sigma_max: float = 3.0
    gaussian_radius_divisor: float = 3.0
    boundary_width: int = 1
    boundary_pos_weight: float = 4.0
    center_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0

    peak_threshold: float = 0.35
    nms_kernel: int = 3
    min_peak_distance: float = 3.0
    max_markers_per_region: int = 4

    min_split_region_area: int = 256
    watershed_boundary_weight: float = 1.0
    watershed_distance_weight: float = 1.0

    boundary_gate_threshold: float = 0.5
    cut_band_radius: int = 1

    min_child_area: int = 32
    min_child_parent_ratio: float = 0.05
    area_tolerance_pixels: int = 0
    require_exactly_one_center_per_child: bool = True

    match_iou_threshold: float = 0.5
    merge_split_overlap_threshold: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RGIDConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})
