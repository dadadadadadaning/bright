import torch

from .core import (
    batched_metrics, direction_name, exact_keep_utilities, sign_agreement,
    weighted_binary_auroc,
)


def test_exact_keep_utility_rewards_correct_damaged_change():
    fixed = torch.eye(4, dtype=torch.float64) * 100
    fixed[2, 2] -= 1
    fixed[2, 1] += 1
    delta = torch.zeros(1, 4, 4)
    delta[0, 2, 1] = -1
    delta[0, 2, 2] = 1
    metric, damaged = exact_keep_utilities(fixed, delta)
    assert metric[0] > 0
    assert damaged[0] > 0


def test_batched_metrics_matches_scalar_shape():
    hist = torch.eye(4) * 10
    single = batched_metrics(hist)
    batch = batched_metrics(torch.stack((hist, hist)))
    assert single[0].ndim == 0
    assert torch.allclose(batch[0], single[0].repeat(2))
    assert torch.allclose(batch[1], single[1].repeat(2))


def test_weighted_auc_perfect_and_reversed():
    labels = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([1., 5., 2., 8.])
    assert weighted_binary_auroc(labels, torch.tensor([0., .1, .8, 1.]), weights) == 1.0
    assert weighted_binary_auroc(labels, torch.tensor([1., .8, .1, 0.]), weights) == 0.0


def test_sign_and_direction_contract():
    assert sign_agreement(torch.tensor([1., -1., 0.]), torch.tensor([2., -3., 4.])) == 1.0
    assert direction_name(1 * 4 + 2) == "intact_to_damaged"

