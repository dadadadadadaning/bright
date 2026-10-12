from dataclasses import dataclass

import numpy as np
import torch

from s9_instance_common import (
    instances_from_bag_specs,
    iou_matrix,
    match_instances,
)


@dataclass
class FakeBagSpec:
    batch_index: int
    mask: torch.Tensor
    label: int = -1
    area: int = 0


def main():
    mask1 = torch.zeros((12, 12), dtype=torch.bool)
    mask1[1:5, 1:5] = True
    mask2 = torch.zeros((12, 12), dtype=torch.bool)
    mask2[7:11, 7:11] = True

    specs = [
        FakeBagSpec(0, mask1, label=1, area=16),
        FakeBagSpec(0, mask2, label=3, area=16),
    ]
    instances = instances_from_bag_specs(
        specs,
        keep_labels=True,
    )
    assert len(instances) == 2
    assert instances[0].label == 1
    assert instances[1].label == 3
    assert instances[0].area == 16
    assert instances[1].bbox_yxyx == (7, 7, 11, 11)

    matrix = iou_matrix(instances, instances)
    matches = match_instances(
        matrix,
        threshold=0.5,
        method="greedy",
    )
    assert len(matches) == 2
    assert all(abs(item.iou - 1.0) < 1e-12 for item in matches)

    print("Synthetic S9-A v3 exact-BagSpec test: PASS")


if __name__ == "__main__":
    main()
