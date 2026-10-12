#!/usr/bin/env python3
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script/standard_ML"))

model_stub = types.ModuleType(
    "model.DamageFormer_S8_MICB_DSMIL"
)

@dataclass
class BuildingBagSpec:
    batch_index: int
    mask: torch.Tensor
    label: int = -1
    area: int = 0

model_stub.BuildingBagSpec = BuildingBagSpec
sys.modules[
    "model.DamageFormer_S8_MICB_DSMIL"
] = model_stub

common_stub = types.ModuleType(
    "script.standard_ML.s9_instance_common"
)

@dataclass(frozen=True)
class MaskInstance:
    instance_id: int
    batch_index: int
    mask: np.ndarray
    area: int
    bbox_yxyx: tuple
    label: int | None = None
    purity: float = 0.0

common_stub.MaskInstance = MaskInstance

def instances_from_specs(specs, keep_labels=False):
    result = []
    for index, spec in enumerate(specs):
        mask = spec.mask.numpy().astype(bool)
        ys, xs = np.nonzero(mask)
        result.append(
            MaskInstance(
                index,
                spec.batch_index,
                mask,
                int(mask.sum()),
                (
                    int(ys.min()),
                    int(xs.min()),
                    int(ys.max()) + 1,
                    int(xs.max()) + 1,
                ),
            )
        )
    return result

common_stub.instances_from_bag_specs = instances_from_specs
common_stub.make_predicted_specs = (
    lambda loc, builder, min_area=32: builder(
        loc,
        torch.where(loc == 1, 1, 0),
        min_area=min_area,
    )
)
sys.modules[
    "script.standard_ML.s9_instance_common"
] = common_stub

from s9ic_g1_geometry import (
    G1DecoderConfig,
    extract_region_peaks,
    g1_geometry_instances,
)


region = np.zeros((96, 96), dtype=bool)
region[15:82, 12:85] = True

yy, xx = np.mgrid[:24, :24]
wide = np.exp(
    -((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 3.0 ** 2)
).astype(np.float32)
peaks = extract_region_peaks(
    wide,
    region,
    G1DecoderConfig(),
)
assert len(peaks) == 1, peaks

two = np.zeros((24, 24), dtype=np.float32)
two[12, 8] = 0.95
two[12, 16] = 0.90
peaks = extract_region_peaks(
    two,
    region,
    G1DecoderConfig(),
)
assert len(peaks) == 2, peaks

close = np.zeros((24, 24), dtype=np.float32)
close[12, 10] = 0.95
close[12, 12] = 0.90
peaks = extract_region_peaks(
    close,
    region,
    G1DecoderConfig(min_peak_distance_native=3),
)
assert len(peaks) == 1, peaks


def fake_builder(loc, damage, min_area=32):
    mask = loc[0].numpy() == 1
    return [
        BuildingBagSpec(
            0,
            torch.from_numpy(mask),
            1,
            int(mask.sum()),
        )
    ]


yyf, xxf = np.mgrid[:96, :96]
left = (yyf - 48) ** 2 + (xxf - 34) ** 2 <= 18 ** 2
right = (yyf - 48) ** 2 + (xxf - 62) ** 2 <= 18 ** 2
building = np.zeros((96, 96), dtype=np.float32)
building[left | right] = 0.95
center = np.zeros((24, 24), dtype=np.float32)
center[12, 8] = 0.95
center[12, 15] = 0.95
boundary = np.zeros_like(center)
boundary[:, 12] = 1.0

instances, specs, diagnostics = g1_geometry_instances(
    building,
    center,
    boundary,
    fake_builder,
    G1DecoderConfig(min_region_area_to_split=128),
)
assert len(instances) == len(specs) == 2, diagnostics
assert diagnostics["boundary_gate_accepted_regions"] == 0
assert diagnostics["rollback_regions"] == 0

print("S9-Ic-G1 native NMS/decoder test: PASS")
