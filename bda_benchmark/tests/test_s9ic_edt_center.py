#!/usr/bin/env python3
"""Regression test for the predeclared source-scale EDT center target."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.targets import (
    build_unique_gaussian_targets,
    deterministic_edt_center,
    map_point_to_size,
)


@dataclass
class Spec:
    batch_index: int
    mask: torch.Tensor
    label: int
    area: int


mask = np.zeros((40, 40), dtype=np.bool_)
mask[5:34, 5:12] = True
mask[27:34, 5:34] = True
center = deterministic_edt_center(mask)
distance = distance_transform_edt(mask)
assert np.isclose(distance[center], distance[mask].max())
assert center == deterministic_edt_center(mask)


def builder(loc, damage, min_area=1):
    return [Spec(0, torch.from_numpy(mask), 1, int(mask.sum()))]


labels = torch.zeros((1, 40, 40), dtype=torch.long)
targets = build_unique_gaussian_targets(
    labels,
    labels,
    builder,
    output_size=(20, 20),
    min_area=1,
)
mapped = map_point_to_size(center, mask.shape, (20, 20))
recorded = tuple(targets["instance_centers"][0][1:3])
assert recorded == mapped, (recorded, mapped)
assert targets["unique_center_pixels"] == 1
print("S9-Ic source-scale EDT center test: PASS")
