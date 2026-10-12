from __future__ import annotations
from collections import Counter
import torch

from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import Retrieval
from .core import (dual_bank_error_score, granularity_targets,
                   sample_hierarchical_pairs, weighted_neighbour_distribution)


def _cache():
    events, source, labels = [], [], []
    for event in ("a", "b", "c", "d"):
        # same observable P3 source: correct, affected miss, and other error
        events.extend([event] * 3); source.extend([1, 1, 1]); labels.extend([1, 2, 0])
    return {"image_index": torch.arange(12), "image_events": events,
            "p3_predictions": torch.tensor(source), "labels": torch.tensor(labels),
            "label_purity": torch.ones(12)}


def test_granularity_targets_form_expected_hierarchy():
    target = granularity_targets(_cache())
    assert target["p3_error"].tolist()[:3] == [0, 1, 1]
    assert target["affected"].tolist()[:3] == [0, 1, 0]
    assert target["error_family"].tolist()[:3] == [0, 1, 4]
    assert target["full_direction"].tolist()[:3] == [5, 6, 4]


def test_hierarchical_pair_sampler_has_registered_composition():
    cache = _cache(); generator = torch.Generator().manual_seed(17)
    left, right, label, kinds = sample_hierarchical_pairs(cache, 100, generator)
    count = Counter(kinds)
    assert count == {"error_positive": 40, "error_hard_negative": 30,
                     "correct_positive": 20, "state_negative": 10}
    events = cache["image_events"]
    assert all(events[a] != events[b] for a, b in zip(left.tolist(), right.tolist()))
    assert int(label.sum()) == 60


def test_weighted_distribution_uses_only_retrieved_values():
    retrieval = Retrieval(torch.tensor([[0, 1]]), torch.tensor([[1., 1.]]),
                          torch.zeros(1, 2), torch.ones(1, dtype=torch.bool))
    probability = weighted_neighbour_distribution(retrieval, torch.tensor([0, 1]), 2)
    assert torch.allclose(probability, torch.tensor([[.5, .5]]))


def test_dual_bank_score_increases_when_residual_is_more_similar():
    correct = Retrieval(torch.tensor([[0]]), torch.tensor([[.2]]), torch.zeros(1, 1), torch.ones(1, dtype=torch.bool))
    residual = Retrieval(torch.tensor([[0]]), torch.tensor([[.8]]), torch.zeros(1, 1), torch.ones(1, dtype=torch.bool))
    score, difference = dual_bank_error_score(correct, residual, .1)
    assert float(difference) > 0
    assert float(score) > .5

