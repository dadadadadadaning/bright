#!/usr/bin/env python3
"""Synthetic Center+Offset target, loss, and pixel-assignment test."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.DamageFormer_S9Id_CenterOffset import (
    CenterOffsetGeometryHead,
)
from bda_benchmark.model.instance_geometry.config import geometry_preset
from bda_benchmark.model.instance_geometry.decode import (
    decode_center_offset_instances,
)
from bda_benchmark.model.instance_geometry.losses import reliable_geometry_loss
from bda_benchmark.model.instance_geometry.targets import (
    build_center_offset_targets,
)


@dataclass
class Spec:
    batch_index: int
    mask: torch.Tensor
    label: int
    area: int


def _builder(loc, damage, min_area=1):
    first = torch.zeros((32, 32), dtype=torch.bool)
    second = torch.zeros((32, 32), dtype=torch.bool)
    first[8:24, 3:15] = True
    second[5:27, 15:29] = True
    return [
        Spec(0, first, 1, int(first.sum())),
        Spec(0, second, 2, int(second.sum())),
    ]


def main():
    labels = torch.zeros((1, 32, 32), dtype=torch.long)
    targets = build_center_offset_targets(
        labels,
        labels,
        _builder,
        output_size=(32, 32),
        min_area=1,
    )
    assert int((targets["center"] == 1).sum()) == 2

    specs = _builder(labels, labels)
    totals = []
    for spec in specs:
        totals.append(float(targets["offset_weight"][0, 0][spec.mask].sum()))
    assert abs(totals[0] - totals[1]) < 1e-4, totals

    head = CenterOffsetGeometryHead(16, hidden_channels=16)
    output = head(torch.randn(1, 16, 32, 32))
    losses = reliable_geometry_loss(
        output["center_logits"],
        output["boundary_logits"],
        targets,
        output["offset_logits"],
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])

    building = np.zeros((32, 32), dtype=np.float32)
    building[5:27, 3:29] = 0.95
    center = np.zeros((32, 32), dtype=np.float32)
    center[16, 9] = 0.95
    center[16, 22] = 0.95
    yy, xx = np.mgrid[:32, :32]
    offset = np.zeros((2, 32, 32), dtype=np.float32)
    left = building > 0.5
    left[:, 16:] = False
    right = (building > 0.5) & ~left
    offset[0][left] = 9 - xx[left]
    offset[1][left] = 16 - yy[left]
    offset[0][right] = 22 - xx[right]
    offset[1][right] = 16 - yy[right]

    instances, report = decode_center_offset_instances(
        building,
        center,
        offset,
        config=geometry_preset("id").with_overrides(
            min_instance_area=8,
            offset_assignment_radius=2.0,
        ),
    )
    assert len(instances) == 2, report
    assert report["watershed_used"] is False
    assert report["assignment_rate"] > 0.99
    print("S9-Id Center+Offset: PASS")


if __name__ == "__main__":
    main()
