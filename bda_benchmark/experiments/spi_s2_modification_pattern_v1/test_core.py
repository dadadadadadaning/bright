import torch

from bda_benchmark.experiments.spi_s2_modification_pattern_v1.core import (
    DISTANCE_NAMES,
    FEATURE_NAMES,
    TOPOLOGY_NAMES,
    distance_features,
    feature_groups,
    topology_features,
    utility_counts,
)


def test_concentrated_change_has_larger_cluster_fraction():
    building = torch.ones(12, 12, dtype=torch.bool)
    concentrated = torch.zeros_like(building); concentrated[3:7, 3:7] = True
    fragmented = torch.zeros_like(building); fragmented[2, 2] = True; fragmented[2, 9] = True; fragmented[9, 2] = True; fragmented[9, 9] = True
    a = topology_features(concentrated, building)
    b = topology_features(fragmented, building)
    assert len(a) == len(TOPOLOGY_NAMES)
    assert a[1] > b[1]


def test_boundary_distance_distinguishes_edge_and_interior():
    building = torch.zeros(12, 12, dtype=torch.bool); building[1:11, 1:11] = True
    edge = torch.zeros_like(building); edge[1, 2:10] = True
    interior = torch.zeros_like(building); interior[5:7, 5:7] = True
    a = distance_features(edge, building); b = distance_features(interior, building)
    assert len(a) == len(DISTANCE_NAMES)
    assert b[0] > a[0]


def test_feature_groups_have_registered_dimensions():
    building = torch.zeros(8, 8, dtype=torch.bool); building[1:7, 1:7] = True
    base = torch.ones(8, 8, dtype=torch.long)
    candidate = base.clone(); candidate[2:4, 2:4] = 2
    probs = torch.full((4, 8, 8), 0.05); probs[1] = 0.85
    candidate_probs = probs.clone(); candidate_probs[1, 2:4, 2:4] = 0.35; candidate_probs[2, 2:4, 2:4] = 0.55
    groups = feature_groups(building=building, base_probabilities=probs,
                            candidate_probabilities=candidate_probs, base_prediction=base,
                            candidate_prediction=candidate, shuffle_seed=7)
    for name, values in groups.items():
        assert values.shape == (len(FEATURE_NAMES[name]),)
        assert torch.isfinite(values).all()


def test_utility_counts_corrected_and_harmed():
    base = torch.tensor([[1, 1], [1, 1]])
    candidate = torch.tensor([[2, 2], [1, 1]])
    target = torch.tensor([[2, 1], [1, 2]])
    row = utility_counts(base, candidate, target, torch.ones_like(base).bool())
    assert row["changed"] == 2
    assert row["corrected"] == 1
    assert row["harmed"] == 1
    assert row["net"] == 0

