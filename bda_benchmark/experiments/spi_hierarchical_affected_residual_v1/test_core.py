import torch

from .core import (
    AO_OUTCOME_CORRECTED, AO_OUTCOME_HARMED, affected_candidate_outcomes,
    apply_affected_oracle, building_core_and_ring, deterministic_subsample,
    equal_affected_residual, minimum_affected_residual,
)
from .model import (
    ABLATIONS, ExpectedConfusionConfidenceGate, HierarchicalAffectedResidual,
    region_relative,
)
from .build_evidence_cache import _probability_maps


def test_equal_residual_preserves_damaged_destroyed_difference():
    logits = torch.randn(2, 4, 8, 8)
    residual = torch.rand(2, 1, 8, 8)
    mask = torch.ones(2, 1, 8, 8)
    output = equal_affected_residual(logits, residual, mask)
    # In float32, (x+r)-(y+r) is algebraically equal to x-y but the two
    # addition roundings can leave an error of a few ULPs near zero.
    assert torch.allclose(
        output[:, 3] - output[:, 2],
        logits[:, 3] - logits[:, 2],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.equal(output[:, :2], logits[:, :2])


def test_zero_initialised_module_is_exact_identity():
    module = HierarchicalAffectedResidual(8, 8, hidden=8)
    logits = torch.randn(1, 4, 16, 16)
    rgb = torch.randn(1, 8, 8, 8); sar = torch.randn(1, 8, 8, 8)
    labels = torch.ones(1, 16, 16, dtype=torch.long)
    output, aux = module(logits, rgb, sar, labels, labels > 0)
    assert module.gamma.item() == 0
    assert torch.equal(output, logits)
    assert torch.equal(aux["scaled_occurrence_residual"], torch.zeros_like(aux["scaled_occurrence_residual"]))


def test_region_relative_has_zero_instance_mean():
    features = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    labels = torch.ones(1, 2, 2, dtype=torch.long)
    relative = region_relative(features, labels)
    assert torch.allclose(relative.mean((2, 3)), torch.zeros(1, 2))


def test_confidence_reduces_uncertain_gate():
    gate = ExpectedConfusionConfidenceGate()
    delta = torch.tensor([1.0, 1.0])
    probability = torch.tensor([[1/3, 1/3, 1/3], [.98, .01, .01]])
    values = gate(delta, delta, probability, use_confidence=True)
    assert values[0] < values[1]


def test_oracle_only_applies_reachable_correct_promotions():
    probabilities = torch.tensor([
        [[.1, .1]], [[.6, .6]], [[.2, .1]], [[.1, .2]],
    ])
    target = torch.tensor([[2, 2]])
    result, reachable = apply_affected_oracle(probabilities, torch.ones(1, 2), target)
    assert reachable.tolist() == [[True, False]]
    assert result.tolist() == [[2, 1]]


def test_ablation_registry_is_causally_nested():
    assert not ABLATIONS["a5_relative_rgbsar"]["expected_confusion"]
    assert ABLATIONS["a6_expected_confusion"]["expected_confusion"]
    assert not ABLATIONS["a6_expected_confusion"]["confidence"]
    assert ABLATIONS["a7_confidence"]["confidence"]


def test_candidate_outcomes_keep_neutral_separate():
    probabilities = torch.tensor([
        [[.1, .1, .1]], [[.6, .6, .6]], [[.2, .2, .1]], [[.1, .1, .2]],
    ])
    target = torch.tensor([[2, 1, 2]])
    candidate, outcome = affected_candidate_outcomes(
        probabilities, torch.ones(1, 3), target
    )
    assert candidate.all()
    assert outcome.tolist() == [[AO_OUTCOME_CORRECTED, AO_OUTCOME_HARMED, 0]]


def test_required_residual_uses_log_probability_gap():
    probabilities = torch.tensor([
        [[.1]], [[.6]], [[.2]], [[.1]],
    ])
    observed = minimum_affected_residual(probabilities)
    assert torch.allclose(observed, torch.tensor([[torch.log(torch.tensor(3.0))]]))


def test_core_ring_partition_and_subsample_are_deterministic():
    labels = torch.zeros(7, 7, dtype=torch.long); labels[1:6, 1:6] = 1
    core, ring = building_core_and_ring(labels, radius=1)
    assert int(core.sum()) == 9
    assert torch.equal(core | ring, labels > 0)
    first = deterministic_subsample(labels > 0, 5, 17)
    second = deterministic_subsample(labels > 0, 5, 17)
    assert int(first.sum()) == 5 and torch.equal(first, second)


def test_probability_evidence_maps_are_tensors_with_named_channels():
    p3 = torch.softmax(torch.randn(4, 3, 5), dim=0)
    fixed = torch.softmax(torch.randn(4, 3, 5), dim=0)
    maps, names = _probability_maps(p3, fixed)
    assert maps.shape == (1, 13, 3, 5)
    assert len(names) == maps.shape[1]
    assert torch.isfinite(maps).all()
