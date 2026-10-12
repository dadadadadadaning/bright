import torch

from bda_benchmark.experiments.spi_memory_dependency_audit_v1.core import (
    p3_uncertainty_features, random_cross_event_support, route_probability,
    shuffle_within_event,
)


def test_random_support_excludes_query_event():
    query = ["a", "b", "a"]
    memory = ["a", "b", "c", "c"]
    selected = random_cross_event_support(query, memory, topk=4, seed=7)
    for row, event in enumerate(query):
        assert all(memory[index] != event for index in selected[row].tolist())


def test_shuffle_preserves_event_group_values():
    value = torch.arange(12).view(6, 2)
    events = ["a", "a", "a", "b", "b", "b"]
    shuffled = shuffle_within_event(value, events, seed=9)
    assert set(map(tuple, shuffled[:3].tolist())) == set(map(tuple, value[:3].tolist()))
    assert set(map(tuple, shuffled[3:].tolist())) == set(map(tuple, value[3:].tolist()))


def test_uncertainty_features_are_finite():
    probability = torch.softmax(torch.randn(5, 4), 1)
    feature = p3_uncertainty_features(probability)
    assert feature.shape == (5, 9)
    assert torch.isfinite(feature).all()


def test_router_uses_expert_only_above_threshold():
    p3 = torch.tensor([[1., 0.], [1., 0.]])
    expert = torch.tensor([[0., 1.], [0., 1.]])
    output, routed = route_probability(p3, expert, torch.tensor([.2, .8]), .5)
    assert torch.equal(routed, torch.tensor([False, True]))
    assert torch.equal(output, torch.tensor([[1., 0.], [0., 1.]]))
