#!/usr/bin/env python3
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script/standard_ML"))

instance_common = types.ModuleType(
    "script.standard_ML.s9_instance_common"
)
instance_common.normalize_label_tensor = (
    lambda x: x[:, 0] if x.ndim == 4 else x
)
sys.modules[
    "script.standard_ML.s9_instance_common"
] = instance_common

from s9ic_targets_losses import (
    build_unique_gaussian_targets,
    reliable_geometry_loss,
)


@dataclass
class Spec:
    batch_index: int
    mask: torch.Tensor
    label: int
    area: int


def builder(loc, damage, min_area=32):
    h, w = loc.shape[-2:]
    full = np.zeros((h, w), dtype=bool)
    full[12:44, 15:50] = True
    border = np.zeros((h, w), dtype=bool)
    border[0:16, 55:76] = True
    return [
        Spec(0, torch.from_numpy(full), 1, int(full.sum())),
        Spec(0, torch.from_numpy(border), 1, int(border.sum())),
    ]


loc = torch.zeros((1, 80, 80), dtype=torch.long)
damage = torch.zeros_like(loc)
targets = build_unique_gaussian_targets(
    loc,
    damage,
    builder,
    output_size=(20, 20),
)

center = targets["center"][0, 0].numpy()
assert center.shape == (20, 20)
assert int(np.sum(center == 1.0)) == 1, np.argwhere(center == 1.0)
assert targets["complete_instance_count"] == 1
assert targets["border_touching"] == 1
assert targets["unique_center_pixels"] == 1
assert targets["mapped_center_collisions"] == 0
assert float(targets["center_valid"].min()) == 0.0
assert float(targets["boundary"].sum()) > 0.0

center_logits = torch.randn(
    1, 1, 20, 20, requires_grad=True
)
boundary_logits = torch.randn(
    1, 1, 20, 20, requires_grad=True
)
losses = reliable_geometry_loss(
    center_logits,
    boundary_logits,
    targets,
)
assert torch.isfinite(losses["loss"])
losses["loss"].backward()
assert torch.isfinite(center_logits.grad).all()
assert torch.isfinite(boundary_logits.grad).all()

# Regression: bf16 cannot represent 1 - 1e-6. Extreme logits must therefore
# remain finite without relying on a sigmoid clamp near one.
bf16_center = torch.tensor(
    [[[[1000.0, -1000.0], [-1000.0, 1000.0]]]],
    dtype=torch.bfloat16,
    requires_grad=True,
)
bf16_boundary = torch.tensor(
    [[[[-1000.0, 1000.0], [1000.0, -1000.0]]]],
    dtype=torch.bfloat16,
    requires_grad=True,
)
bf16_targets = {
    "center": torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    ),
    "center_valid": torch.ones((1, 1, 2, 2), dtype=torch.float32),
    "boundary": torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]]]],
        dtype=torch.float32,
    ),
}
bf16_losses = reliable_geometry_loss(
    bf16_center,
    bf16_boundary,
    bf16_targets,
)
assert bf16_losses["loss"].dtype == torch.float32
assert torch.isfinite(bf16_losses["loss"])
bf16_losses["loss"].backward()
assert torch.isfinite(bf16_center.grad).all()
assert torch.isfinite(bf16_boundary.grad).all()

print("S9-Ic unique target/loss test: PASS")
