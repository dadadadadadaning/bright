#!/usr/bin/env python3
"""Synthetic G1 smoke test plus optional strict S8 checkpoint load."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.DamageFormer_S9Ic_ReliableGeometry import (
    ReliableGeometryHead,
)
from bda_benchmark.model.instance_geometry.losses import reliable_geometry_loss
from bda_benchmark.model.instance_geometry.s8_backbone import FrozenS8Backbone
from bda_benchmark.model.instance_geometry.targets import (
    build_unique_gaussian_targets,
)


@dataclass
class _Spec:
    batch_index: int
    mask: torch.Tensor
    label: int
    area: int


def _builder(loc, damage, min_area=32):
    mask = torch.zeros((64, 64), dtype=torch.bool)
    mask[12:50, 16:52] = True
    return [_Spec(0, mask, 1, int(mask.sum()))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--s9ib_json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", default="none")
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        backbone = FrozenS8Backbone.from_checkpoint(
            args.checkpoint,
            allow_unsafe=args.allow_unsafe_checkpoint,
        )
        backbone.assert_frozen()

    head = ReliableGeometryHead(256, 64)
    feature = torch.randn(1, 256, 16, 16)
    prediction = head(feature)
    labels = torch.zeros((1, 64, 64), dtype=torch.long)
    targets = build_unique_gaussian_targets(
        labels,
        labels,
        _builder,
        output_size=(16, 16),
    )
    losses = reliable_geometry_loss(
        prediction["center_logits"],
        prediction["boundary_logits"],
        targets,
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    print(
        "S9-Ic-G1 smoke: PASS",
        {
            key: float(value.detach().item())
            for key, value in losses.items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

