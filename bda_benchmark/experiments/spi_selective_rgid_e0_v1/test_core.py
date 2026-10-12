from __future__ import annotations

import numpy as np
import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.method import SplitRiskEvidence
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    RegionLabelMap,
    fixed_s2_candidate_from_labels,
)

from .core import (
    partition_from_selected_parents,
    region_map,
    s2_prediction,
    split_oracle_prediction,
    threshold_evidence,
)


def _evidence(score: float, children: int = 2, peaks: int = 2):
    return SplitRiskEvidence(
        parent_id=1, area=100, area_score=.5, elongation_score=.1,
        holes_score=0, entropy_score=.2, peak_count=peaks, peak_score=.3,
        boundary_score=.4, risk_score=score, child_count=children,
        accepted=True, rejection_reason="accepted",
    )


def test_threshold_is_monotonic_and_keeps_structural_rejection():
    rows = {1: _evidence(.6), 2: _evidence(.9, children=1)}
    assert threshold_evidence(rows, .5)[1].accepted
    assert not threshold_evidence(rows, .7)[1].accepted
    assert not threshold_evidence(rows, .1)[2].accepted


def test_region_map_remaps_sparse_ids():
    labels = np.asarray([[0, 4], [9, 9]])
    result = region_map(labels)
    assert result.count == 2
    assert set(torch.unique(result.labels).tolist()) == {0, 1, 2}


def test_split_oracle_selects_only_better_parent():
    parents = torch.tensor([[1, 1, 2, 2]])
    target = torch.tensor([[2, 2, 1, 1]])
    loccc = torch.tensor([[1, 1, 1, 1]])
    split = torch.tensor([[2, 2, 2, 2]])
    output, counts = split_oracle_prediction(loccc, split, target, parents)
    assert torch.equal(output, torch.tensor([[2, 2, 1, 1]]))
    assert counts["rgid_selected"] == 1


def test_prediction_only_s2_matches_locked_full_candidate():
    generator = torch.Generator().manual_seed(7341)
    probabilities = torch.rand((4, 11, 13), generator=generator)
    probabilities /= probabilities.sum(dim=0, keepdim=True)
    labels = torch.zeros((11, 13), dtype=torch.long)
    labels[1:6, 1:5] = 1
    labels[2:9, 7:12] = 4  # sparse IDs exercise the fast bincount path
    remapped = region_map(labels)
    expected = fixed_s2_candidate_from_labels(
        probabilities, remapped
    ).probabilities.argmax(0)
    observed = s2_prediction(probabilities, labels)
    assert torch.equal(observed, expected)


def test_vectorized_selective_partition_preserves_membership():
    parents = np.asarray(
        [[0, 1, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2]], dtype=np.int32
    )
    children = np.asarray(
        [[0, 1, 1, 2, 3, 3], [0, 1, 2, 2, 3, 4]], dtype=np.int32
    )
    result = partition_from_selected_parents(parents, children, [1])
    assert np.array_equal(result > 0, parents > 0)
    assert result[0, 1] == result[1, 1]
    assert result[0, 1] != result[0, 3]
    assert result[0, 4] == result[1, 5]  # unselected parent 2 remains one region
