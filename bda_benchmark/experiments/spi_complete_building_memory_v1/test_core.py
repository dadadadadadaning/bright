import torch

from bda_benchmark.experiments.spi_complete_building_memory_v1.core import (
    balanced_memory_indices, building_role_tokens, geometry_vector, macro_f1,
    masked_mean_std,
)
from bda_benchmark.experiments.spi_complete_building_memory_v1.evaluate_writeback import (
    full_semantic_prediction, severity_only_prediction,
)


def test_geometry_is_finite_and_label_free():
    mask = torch.zeros(8, 8, dtype=torch.bool); mask[2:6, 1:7] = True
    value = geometry_vector(mask)
    assert value.shape == (8,) and torch.isfinite(value).all()


def test_role_tokens_have_fixed_contract():
    token = torch.randn(6, 4, 4); energy = torch.arange(16).reshape(4, 4).float()
    mask = torch.zeros(8, 8, dtype=torch.bool); mask[1:7, 1:7] = True
    value, valid, fraction = building_role_tokens(token, energy, mask, tokens_per_role=2)
    assert value.shape == (5, 2, 6)
    assert valid.shape == (5, 2) and valid.any(1).all()
    assert fraction.shape == (5,) and torch.isfinite(value).all()


def test_balanced_memory_caps_each_event_class():
    labels = torch.tensor([0, 0, 0, 1, 1, 0, 0, 1])
    events = ["a"] * 5 + ["b"] * 3
    rows = balanced_memory_indices(labels, events, maximum_per_event_class=2, seed=7)
    for event in ("a", "b"):
        for label in (0, 1):
            count = sum(events[i] == event and int(labels[i]) == label for i in rows.tolist())
            assert count <= 2


def test_macro_f1_is_one_for_exact_prediction():
    target = torch.tensor([0, 1, 2, 3])
    assert macro_f1(target, target) == 1.0


def test_modality_pooling_covers_complete_mask():
    feature = torch.arange(2 * 4 * 4).reshape(2, 4, 4).float()
    mask = torch.zeros(8, 8, dtype=torch.bool); mask[:4, :4] = True
    value = masked_mean_std(feature, mask)
    assert value.shape == (4,) and torch.isfinite(value).all()


def test_severity_writeback_preserves_foreground_exactly():
    probability = torch.tensor([
        [[.8, .1], [.1, .1]], [[.1, .7], [.2, .2]],
        [[.05, .1], [.6, .2]], [[.05, .1], [.1, .5]],
    ])
    regions = torch.tensor([[0, 1], [1, 1]])
    posterior = torch.tensor([[0., 0., 0., 1.]])
    result = severity_only_prediction(probability, regions, posterior, .25)
    assert torch.equal(result > 0, probability.argmax(0) > 0)
    assert result[0, 0] == 0


def test_full_semantic_writeback_can_reject_a_false_building():
    probability = torch.tensor([
        [[.1, .1], [.1, .1]], [[.9, .9], [.9, .9]],
        [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
    ])
    regions = torch.ones(2, 2, dtype=torch.long)
    posterior = torch.tensor([[1., 0., 0., 0.]])
    result = full_semantic_prediction(probability, regions, posterior, .90)
    assert not bool((result > 0).any())
