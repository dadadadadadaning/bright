import torch

from .core import (
    ClassSafeLinearVerifier, apply_thresholds, cluster_event_mask,
    dual_targets, frozen_gradients, image_event_mask,
)


def _cache():
    fixed = torch.eye(4)[None].repeat(2, 1, 1) * 100
    delta = torch.zeros(2, 4, 4)
    delta[0, 2, 1] = -1
    delta[0, 2, 2] = 1
    delta[1, 2, 2] = -1
    delta[1, 2, 1] = 1
    return {
        "image_events": ["event_a", "event_b"],
        "image_index": torch.tensor([0, 1]),
        "fixed_s2_histograms": fixed,
        "delta_hist": delta,
    }


def test_event_masks_hold_out_exactly_one_event():
    cache = _cache()
    assert image_event_mask(cache, "event_a", equal=True).tolist() == [True, False]
    assert cluster_event_mask(cache, "event_a", equal=False).tolist() == [False, True]


def test_dual_targets_separate_metric_benefit_and_damaged_harm():
    cache = _cache()
    gradients = frozen_gradients(cache, torch.tensor([True, True]))
    metric, harm, metric_utility, damaged_utility = dual_targets(cache, *gradients)
    assert metric.tolist() == [1.0, 0.0]
    assert harm.tolist() == [0.0, 1.0]
    assert metric_utility[0] > 0 and metric_utility[1] < 0
    assert damaged_utility[0] > 0 and damaged_utility[1] < 0


def test_class_safe_acceptance_requires_benefit_and_low_harm():
    metric = torch.tensor([0.8, 0.8, 0.2])
    harm = torch.tensor([0.2, 0.8, 0.2])
    policy = {"metric_threshold": 0.5, "harm_threshold": 0.5}
    assert apply_thresholds(metric, harm, policy).tolist() == [True, False, False]


def test_verifier_has_only_two_linear_outputs():
    model = ClassSafeLinearVerifier(5)
    metric, harm = model(torch.zeros(3, 5))
    assert metric.shape == harm.shape == (3,)
    assert sum(parameter.numel() for parameter in model.parameters()) == 12
