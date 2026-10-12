import torch
import pytest

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from .core import (
    exact_marginal_utilities, metric_labels, metric_utilities, miou_gradient,
    selective_histogram,
)
from .train_verifier import matrix
from .evaluate_confirm import confirmation_checks


def test_miou_gradient_matches_small_finite_difference():
    # Finite differences on confusion counts must use float64.  With float32,
    # adding 1e-4 to a value around 100 loses enough significant digits to
    # make this derivative check fail even when the analytic gradient is right.
    hist = torch.tensor(
        [[100., 5., 2., 1.], [4., 80., 6., 2.],
         [3., 7., 40., 5.], [1., 2., 4., 30.]], dtype=torch.float64,
    )
    gradient = miou_gradient(hist)
    epsilon = 1e-5
    perturbed = hist.clone(); perturbed[2, 1] += epsilon
    observed = (
        metrics_from_histogram(perturbed)["valid_miou"]
        - metrics_from_histogram(hist)["valid_miou"]
    ) / epsilon
    assert abs(float(gradient[2, 1]) - observed) < 1e-8


def test_feature_matrix_repairs_float16_overflow_but_rejects_nan():
    cache = {"features": {"spatial": torch.tensor([[float("inf"), -float("inf")]])}}
    values = matrix(cache, ("spatial",), "test")
    assert torch.isfinite(values).all()
    assert values.tolist() == [[65504.0, -65504.0]]

    cache["features"]["spatial"][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="contains 1 NaN"):
        matrix(cache, ("spatial",), "test")


def test_metric_utility_prefers_correct_damaged_transition():
    hist = torch.eye(4) * 100
    gradient = miou_gradient(hist)
    # One true damaged pixel changes from predicted intact to predicted damaged.
    delta = torch.zeros(1, 4, 4)
    delta[0, 2, 1] = -1
    delta[0, 2, 2] = 1
    assert metric_utilities(delta, gradient)[0] > 0


def test_selective_histogram_reverts_rejected_cluster():
    fixed = torch.zeros(2, 4, 4); fixed[:, 0, 0] = 10
    delta = torch.zeros(2, 4, 4); delta[0, 0, 0] = 2; delta[1, 0, 0] = 3
    result = selective_histogram(fixed, delta, torch.tensor([True, False]))
    assert result[0, 0] == 17


def test_metric_labels_keep_ambiguous_band():
    utility = torch.tensor([-0.2, -0.01, 0.0, 0.02, 0.3])
    labels = metric_labels(utility, epsilon_pp=0.05)
    assert labels.tolist() == [0, -1, -1, -1, 1]


def test_exact_marginal_sign_for_beneficial_delta():
    fixed = torch.eye(4)[None] * 100
    delta = torch.zeros(1, 4, 4); delta[0, 2, 1] = -1; delta[0, 2, 2] = 1
    cache = {"fixed_s2_histograms": fixed, "delta_hist": delta}
    assert exact_marginal_utilities(cache)[0] > 0


def test_meta_confirm_requires_every_preregistered_check():
    passed = confirmation_checks(
        gain_pp=0.31, oracle_recovery=0.26, ci_low_pp=0.01,
        damaged_f1=0.71, fixed_damaged_f1=0.70,
        nonnegative_events=10, event_total=14,
    )
    assert all(passed.values())
    failed = confirmation_checks(
        gain_pp=0.29, oracle_recovery=0.26, ci_low_pp=0.01,
        damaged_f1=0.71, fixed_damaged_f1=0.70,
        nonnegative_events=10, event_total=14,
    )
    assert not failed["gain_vs_fixed_s2_ge_threshold"]
