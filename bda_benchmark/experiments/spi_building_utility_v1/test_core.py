from __future__ import annotations

import torch

from .core import building_feature_groups, oracle_predictions, utility_counts


def synthetic_case():
    base = torch.ones((2, 4), dtype=torch.long)
    candidate = torch.tensor([[2, 2, 2, 2], [1, 1, 2, 2]])
    target = torch.tensor([[2, 2, 1, 1], [1, 1, 2, 1]])
    labels = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
    return base, candidate, target, labels


def test_building_and_pixel_oracles_obey_selection_unit():
    base, candidate, target, labels = synthetic_case()
    first = utility_counts(base, candidate, target, labels == 1)
    second = utility_counts(base, candidate, target, labels == 2)
    assert first == {"changed": 2, "corrected": 2, "harmed": 0, "neutral": 0, "net": 2, "utility_ratio": 1.0}
    assert second["changed"] == 4
    assert second["corrected"] == 1
    assert second["harmed"] == 3
    assert second["net"] == -2
    building, pixel = oracle_predictions(base, candidate, target, labels)
    assert torch.equal(building[labels == 1], candidate[labels == 1])
    assert torch.equal(building[labels == 2], base[labels == 2])
    assert int((pixel == target).sum()) >= int((building == target).sum())


def test_building_feature_groups_are_finite_and_fixed_width():
    base, candidate, _, labels = synthetic_case()
    base_probabilities = torch.nn.functional.one_hot(base, num_classes=4).permute(2, 0, 1).float()
    candidate_probabilities = torch.nn.functional.one_hot(candidate, num_classes=4).permute(2, 0, 1).float()
    groups = building_feature_groups(
        mask=labels == 1,
        base_probabilities=base_probabilities,
        candidate_probabilities=candidate_probabilities,
        base_prediction=base,
        candidate_prediction=candidate,
        record={"purity": 0.75, "margin": 0.2},
        evidence={"direct": torch.rand(4, 2, 4)},
    )
    assert groups["shape"].shape == (4,)
    assert groups["semantic"].shape == (12,)
    assert groups["s2"].shape == (22,)
    assert groups["direct"].shape == (12,)
    assert all(torch.isfinite(values).all() for values in groups.values())
