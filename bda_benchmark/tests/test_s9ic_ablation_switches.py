#!/usr/bin/env python3
"""Synthetic independence test for watershed, boundary gate, and rollback."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.config import GeometryDecoderConfig
from bda_benchmark.model.instance_geometry.decode import (
    decode_center_watershed_instances,
)


def _scene():
    yy, xx = np.mgrid[:64, :64]
    left = (yy - 32) ** 2 + (xx - 23) ** 2 <= 15**2
    right = (yy - 32) ** 2 + (xx - 41) ** 2 <= 15**2
    building = np.zeros((64, 64), dtype=np.float32)
    building[left | right] = 0.95
    center = np.zeros((16, 16), dtype=np.float32)
    center[8, 6] = 0.95
    center[8, 10] = 0.90
    return building, center


def main():
    building, center = _scene()
    boundary_low = np.zeros_like(center)
    boundary_high = np.zeros_like(center)
    boundary_high[:, 8] = 0.95
    base = GeometryDecoderConfig(
        min_instance_area=8,
        min_region_area_to_split=32,
        min_peak_distance_native=2,
    )

    instances, report = decode_center_watershed_instances(
        building,
        center,
        boundary_low,
        config=base.with_overrides(watershed_enabled=False),
    )
    assert len(instances) == 1
    assert report["watershed_disabled_regions"] == 1

    instances, report = decode_center_watershed_instances(
        building,
        center,
        boundary_low,
        config=base,
    )
    assert len(instances) == 2, report

    instances, report = decode_center_watershed_instances(
        building,
        center,
        boundary_low,
        config=base.with_overrides(boundary_gate_enabled=True),
    )
    assert len(instances) == 1
    assert report["boundary_gate_rejected_regions"] == 1

    instances, report = decode_center_watershed_instances(
        building,
        center,
        boundary_high,
        config=base.with_overrides(boundary_gate_enabled=True),
    )
    assert len(instances) == 2, report
    assert report["boundary_gate_accepted_regions"] == 1

    edge_center = center.copy()
    edge_center[8, 10] = 0.0
    edge_center[8, 14] = 0.90
    instances, report = decode_center_watershed_instances(
        building,
        edge_center,
        boundary_high,
        config=base.with_overrides(
            rollback_enabled=True,
            rollback_min_child_ratio=0.45,
        ),
    )
    assert len(instances) == 1, report
    assert report["rollback_regions"] == 1
    print("S9-Ic independent switches: PASS")


if __name__ == "__main__":
    main()
