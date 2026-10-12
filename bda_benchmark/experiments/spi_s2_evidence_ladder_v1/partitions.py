from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.method import (
    SplitRiskEvidence,
    random_matched_partition,
    selective_partition,
)


def regions_to_label_map(regions: Sequence[torch.Tensor], shape: tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(regions, 1):
        value = np.asarray(torch.as_tensor(mask).detach().cpu(), dtype=bool)
        if value.shape != shape:
            raise ValueError("region shape mismatch")
        if np.any((labels > 0) & value):
            raise ValueError("LocCC regions overlap")
        labels[value] = index
    return labels


def label_map_to_regions(labels: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
    array = np.asarray(torch.as_tensor(labels).detach().cpu(), dtype=np.int32)
    return [torch.from_numpy(array == value) for value in sorted(np.unique(array)) if value > 0]


def selective_rgid_regions(
    parent_regions: Sequence[torch.Tensor],
    rgid_child_labels: np.ndarray | torch.Tensor,
    evidence_by_parent: Mapping[int, SplitRiskEvidence],
) -> tuple[list[torch.Tensor], dict[str, object]]:
    shape = tuple(torch.as_tensor(rgid_child_labels).shape)
    parent_labels = regions_to_label_map(parent_regions, shape)
    output = selective_partition(
        parent_labels=parent_labels,
        rgid_child_labels=rgid_child_labels,
        evidence_by_parent=evidence_by_parent,
    )
    conserved = bool(np.array_equal(output > 0, parent_labels > 0))
    if not conserved:
        raise RuntimeError("Selective RGID changed building support")
    return label_map_to_regions(output), {
        "support_conserved": conserved,
        "parent_regions": len(parent_regions),
        "output_regions": int(output.max()),
        "accepted_parents": sum(bool(value.accepted) for value in evidence_by_parent.values()),
    }


def matched_random_rgid_regions(
    parent_regions: Sequence[torch.Tensor],
    rgid_child_labels: np.ndarray | torch.Tensor,
    *,
    split_parent_count: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[int]]:
    shape = tuple(torch.as_tensor(rgid_child_labels).shape)
    parent_labels = regions_to_label_map(parent_regions, shape)
    output, selected = random_matched_partition(
        parent_labels=parent_labels,
        rgid_child_labels=rgid_child_labels,
        split_parent_count=split_parent_count,
        seed=seed,
    )
    if not np.array_equal(output > 0, parent_labels > 0):
        raise RuntimeError("random RGID control changed building support")
    return label_map_to_regions(output), selected

