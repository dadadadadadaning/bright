from __future__ import annotations

import torch

from .core import (Retrieval, bounded_observable_memory_rows,
                   correction_direction, memory_probability, retrieve,
                   sample_metric_pairs)


def _cache():
    # Four events, two P3 sources and two correction outcomes per source.
    return {
        "image_index": torch.arange(8),
        "image_events": ["a", "b", "c", "d", "a", "b", "c", "d"],
        "p3_predictions": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
        "labels": torch.tensor([1, 2, 1, 2, 2, 3, 2, 3]),
        "label_purity": torch.ones(8),
    }


def test_correction_direction_encodes_observable_source_and_target():
    cache = _cache()
    assert torch.equal(correction_direction(cache), cache["p3_predictions"] * 4 + cache["labels"])


def test_metric_pairs_are_cross_event_and_balanced():
    cache = _cache(); generator = torch.Generator().manual_seed(3)
    left, right, label = sample_metric_pairs(cache, 20, generator)
    events = cache["image_events"]
    direction = correction_direction(cache)
    for a, b, positive in zip(left.tolist(), right.tolist(), label.tolist()):
        assert events[a] != events[b]
        assert int(cache["p3_predictions"][a]) == int(cache["p3_predictions"][b])
        assert (int(direction[a]) == int(direction[b])) == bool(positive)
    assert int(label.sum()) == 10


def test_memory_capacity_is_bounded_per_event_and_observable_source():
    cache = _cache()
    rows = bounded_observable_memory_rows(cache, maximum_per_event_source=1, seed=7)
    seen = set()
    events = cache["image_events"]
    for row in rows.tolist():
        key = (events[row], int(cache["p3_predictions"][row]))
        assert key not in seen
        seen.add(key)


def test_retrieval_excludes_query_event_and_respects_route_source():
    qs = torch.tensor([[1., 0.], [0., 1.]])
    ms = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    quality_q = torch.zeros(2, 1); quality_m = torch.zeros(4, 1)
    found = retrieve(
        qs, ms, qs, ms, quality_q, quality_m,
        ["a", "b"], ["a", "b", "a", "c"],
        torch.tensor([1, 2]), torch.tensor([1, 1, 2, 2]),
        torch.ones(2), torch.ones(4), torch.zeros(2), torch.zeros(4),
        mode="route", coarse_k=4, topk=1, maximum_per_event=1,
        quality_limit=1, area_ratio_limit=2,
    )
    assert found.coverage.all()
    memory_events = ["a", "b", "a", "c"]
    memory_sources = torch.tensor([1, 1, 2, 2])
    for row, index in enumerate(found.indices[:, 0].tolist()):
        assert memory_events[index] != ["a", "b"][row]
        assert int(memory_sources[index]) == [1, 2][row]


def test_zero_memory_control_is_bitwise_equal_to_p3():
    base = torch.softmax(torch.randn(3, 4), 1)
    retrieval = Retrieval(torch.zeros(3, 1, dtype=torch.long), torch.ones(3, 1),
                          torch.zeros(3, 1), torch.ones(3, dtype=torch.bool))
    memory = {"labels": torch.zeros(1, dtype=torch.long)}
    assert torch.equal(memory_probability(base, retrieval, memory, "zero", .25), base)
