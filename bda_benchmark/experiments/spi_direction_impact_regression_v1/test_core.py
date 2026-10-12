import torch

from .core import (
    direction_one_hot, impact_weights, risk_score, signed_log_transform,
    target_scale, uses_direction, uses_impact,
)


def test_factorial_variants_are_isolated():
    assert not uses_direction("c1_continuous")
    assert uses_direction("c2_direction")
    assert uses_impact("c3_impact")
    assert uses_direction("c4_direction_impact") and uses_impact("c4_direction_impact")


def test_direction_is_complete_one_hot():
    values = direction_one_hot(torch.tensor([0, 6, 15]))
    assert values.shape == (3, 16)
    assert torch.all(values.sum(1) == 1)
    assert values[1, 6] == 1


def test_signed_log_preserves_sign_and_zero():
    values = signed_log_transform(torch.tensor([-2., 0., 2.]), 1.0)
    assert values[0] < 0 and values[1] == 0 and values[2] > 0
    assert torch.isclose(values[0].abs(), values[2])


def test_impact_weights_emphasize_large_utility():
    metric = torch.tensor([.1, 10.])
    damaged = torch.tensor([.1, 1.])
    selected = torch.tensor([True, True])
    weights = impact_weights(metric, damaged, selected, 1., 1., 8.)
    assert weights[1] > weights[0]
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_risk_is_worst_predicted_utility():
    metric = torch.tensor([1., -2.])
    damaged = torch.tensor([-.5, 3.])
    assert torch.allclose(risk_score(metric, damaged), torch.tensor([.5, 2.]))
    assert target_scale(torch.tensor([0., -2., 4.]), torch.tensor([True, True, True])) == 3.0

