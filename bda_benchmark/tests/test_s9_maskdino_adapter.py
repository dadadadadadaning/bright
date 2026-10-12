#!/usr/bin/env python3
"""Synthetic official-instance target and S8 pyramid adapter test."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.maskdino import (
    S8MultiScaleFeatureAdapter,
    compute_query_budget,
    instances_to_maskdino_target,
)


def main():
    instance_map = torch.zeros((32, 32), dtype=torch.long)
    instance_map[2:12, 3:15] = 7
    instance_map[15:29, 16:30] = 11
    damage = torch.zeros_like(instance_map)
    damage[instance_map == 7] = 1
    damage[instance_map == 11] = 3
    target = instances_to_maskdino_target(
        instance_map,
        damage,
        image_id=5,
    )
    assert target["labels"].tolist() == [0, 2]
    assert target["masks"].shape == (2, 32, 32)
    assert target["boxes"].shape == (2, 4)
    assert bool(((target["boxes"] >= 0) & (target["boxes"] <= 1)).all())

    budget = compute_query_budget(
        [12, 19, 22, 40, 80, 95],
        minimum=20,
        maximum=200,
        round_to=10,
    )
    assert budget["recommended_queries"] % 10 == 0
    assert budget["recommended_queries"] >= budget["percentile_instances"]

    adapter = S8MultiScaleFeatureAdapter(
        in_channels=32,
        out_channels=64,
    )
    features = adapter({"fused_roi": torch.randn(2, 32, 64, 64)})
    assert list(features) == ["res2", "res3", "res4", "res5"]
    assert features["res2"].shape == (2, 64, 64, 64)
    assert features["res5"].shape == (2, 64, 8, 8)
    assert adapter.output_shape()["res5"]["stride"] == 32
    print("S9 MaskDINO adapter: PASS")


if __name__ == "__main__":
    main()
