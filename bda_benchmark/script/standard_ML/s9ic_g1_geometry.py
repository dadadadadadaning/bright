"""S9-Ic G1/G2/G3 decoder with independent ablation switches."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import BuildingBagSpec
from bda_benchmark.model.instance_geometry.config import GeometryDecoderConfig
from bda_benchmark.model.instance_geometry.decode import (
    decode_center_watershed_instances,
    extract_region_peaks as _extract_region_peaks,
    native_local_maxima,
)
from bda_benchmark.script.standard_ML.s9_instance_common import (
    instances_from_bag_specs,
)


@dataclass(frozen=True)
class G1DecoderConfig:
    building_threshold: float = 0.50
    peak_threshold: float = 0.35
    nms_kernel: int = 3
    min_peak_distance_native: int = 3
    max_markers_per_region: int = 4
    min_region_area_to_split: int = 128
    min_instance_area: int = 32
    watershed_enabled: bool = True
    boundary_permission_enabled: bool = False
    rollback_enabled: bool = False
    boundary_gate_threshold: float = 0.35
    rollback_min_child_ratio: float = 0.08
    fallback_to_region_max: bool = True
    watershed_boundary_weight: float = 0.50

    def canonical(self) -> GeometryDecoderConfig:
        return GeometryDecoderConfig(
            name="S9-Ic",
            building_threshold=self.building_threshold,
            peak_threshold=self.peak_threshold,
            nms_kernel=self.nms_kernel,
            min_peak_distance_native=self.min_peak_distance_native,
            max_markers_per_region=self.max_markers_per_region,
            min_region_area_to_split=self.min_region_area_to_split,
            min_instance_area=self.min_instance_area,
            watershed_enabled=self.watershed_enabled,
            boundary_gate_enabled=self.boundary_permission_enabled,
            rollback_enabled=self.rollback_enabled,
            boundary_gate_threshold=self.boundary_gate_threshold,
            rollback_min_child_ratio=self.rollback_min_child_ratio,
            fallback_to_region_max=self.fallback_to_region_max,
            watershed_boundary_weight=self.watershed_boundary_weight,
        )


def _canonical(config: G1DecoderConfig | GeometryDecoderConfig) -> GeometryDecoderConfig:
    return config.canonical() if isinstance(config, G1DecoderConfig) else config


def extract_region_peaks(
    center_probability,
    region_mask,
    config: G1DecoderConfig | GeometryDecoderConfig,
):
    return _extract_region_peaks(
        center_probability,
        region_mask,
        _canonical(config),
    )


def g1_geometry_instances(
    building_probability,
    center_probability,
    boundary_probability,
    instance_builder: Callable[..., Sequence[Any]],
    config: G1DecoderConfig | GeometryDecoderConfig,
):
    """Decode instances and retain the legacy ``(instances,specs,diag)`` API."""

    canonical = _canonical(config)
    masks, diagnostics = decode_center_watershed_instances(
        building_probability,
        center_probability,
        boundary_probability,
        config=canonical,
    )
    specs = [
        BuildingBagSpec(
            batch_index=0,
            mask=torch.from_numpy(mask.astype(np.bool_, copy=False)),
            label=-1,
            area=int(mask.sum()),
        )
        for mask in masks
    ]
    instances = instances_from_bag_specs(specs)
    return instances, specs, diagnostics


__all__ = [
    "G1DecoderConfig",
    "extract_region_peaks",
    "g1_geometry_instances",
    "native_local_maxima",
]
