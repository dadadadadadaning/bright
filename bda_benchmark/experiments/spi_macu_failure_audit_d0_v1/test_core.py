import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from .core import (
    area_masks, class_f1_gradient, direction_name, linear_utility,
    mechanism_counts,
)


def test_damaged_f1_gradient_matches_finite_difference():
    histogram = torch.tensor(
        [[100., 5., 2., 1.], [4., 80., 6., 2.],
         [3., 7., 40., 5.], [1., 2., 4., 30.]],
        dtype=torch.float64,
    )
    gradient = class_f1_gradient(histogram, 2)
    epsilon = 1e-5
    changed = histogram.clone()
    changed[2, 1] += epsilon
    observed = (
        metrics_from_histogram(changed)["damaged_f1"]
        - metrics_from_histogram(histogram)["damaged_f1"]
    ) / epsilon
    assert abs(float(gradient[2, 1]) - observed) < 1e-8


def test_damaged_utility_rewards_correct_damaged_prediction():
    histogram = torch.eye(4, dtype=torch.float64) * 100
    delta = torch.zeros(1, 4, 4)
    delta[0, 2, 1] = -1
    delta[0, 2, 2] = 1
    assert linear_utility(delta, class_f1_gradient(histogram, 2))[0] > 0


def test_direction_and_area_groups_are_unambiguous():
    assert direction_name(2 * 4 + 1) == "damaged_to_intact"
    areas = torch.tensor([33, 64, 65, 128, 129, 256, 257, 512, 513])
    masks = area_masks(areas)
    assert torch.stack(tuple(masks.values())).sum(0).tolist() == [1] * len(areas)


def test_mechanism_counts_distinguish_accept_and_reject_errors():
    accept = torch.tensor([True, True, False, False])
    metric = torch.tensor([-1., 1., 1., -1.])
    damaged = torch.tensor([-1., 1., 1., -1.])
    result = mechanism_counts(accept, metric, damaged)
    assert result == {
        "accepted_metric_harm": 1,
        "rejected_metric_benefit": 1,
        "accepted_damaged_harm": 1,
        "rejected_damaged_benefit": 1,
    }
