import torch

from bda_benchmark.experiments.spi_s2_cluster_oracle_v1.core import (
    oracle_predictions,
    transition_clusters,
    utility_counts,
)


def test_clusters_are_separated_by_transition_and_connectivity():
    base = torch.ones((4, 5), dtype=torch.long)
    candidate = base.clone()
    building = torch.ones_like(base, dtype=torch.bool)
    candidate[0, 0] = 2
    candidate[1, 1] = 2  # diagonal: same 8-connected cluster
    candidate[3, 4] = 2  # disconnected cluster
    candidate[0, 1] = 3  # touching, but a different transition
    rows = transition_clusters(base, candidate, building)
    transitions = sorted(transition for transition, _ in rows)
    assert transitions == [6, 6, 7]
    assert sorted(int(mask.sum()) for _, mask in rows) == [1, 1, 2]


def test_utility_counts_corrected_harmed_and_neutral():
    base = torch.tensor([[1, 1, 1]])
    candidate = torch.tensor([[2, 2, 2]])
    target = torch.tensor([[2, 1, 3]])
    result = utility_counts(base, candidate, target, torch.ones_like(base, dtype=torch.bool))
    assert result == {"changed": 3, "corrected": 1, "harmed": 1, "neutral": 1, "net": 0}


def test_cluster_oracle_accepts_positive_cluster_and_rejects_negative_cluster():
    base = torch.ones((3, 5), dtype=torch.long)
    candidate = base.clone()
    candidate[0, 0:3] = 2
    candidate[2, 3:5] = 2
    target = base.clone()
    target[0, 0:3] = 2  # first cluster benefits from S2; building net is +1
    labels = torch.ones_like(base)
    building, cluster, pixel, counts = oracle_predictions(base, candidate, target, labels)
    assert torch.equal(building, candidate)  # building net is positive
    assert torch.equal(cluster[0, 0:3], candidate[0, 0:3])
    assert torch.equal(cluster[2, 3:5], base[2, 3:5])
    assert torch.equal(pixel, cluster)
    assert counts["clusters_total"] == 2
    assert counts["cluster_oracle_accepted"] == 1
    assert counts["cluster_oracle_rejected"] == 1
