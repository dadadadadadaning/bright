from __future__ import annotations

import torch

from .core import (
    build_memory,
    event_label_balanced_indices,
    evaluate_scores,
    fit_robust_normalizer,
    retrieve_topk_by_utility_label,
    scores_from_neighbors,
)


def test_robust_normalizer_drops_constant_dimensions() -> None:
    values = torch.tensor(
        [[0.0, 4.0, 2.0], [1.0, 4.0, 4.0], [2.0, 4.0, 8.0], [100.0, 4.0, 10.0]]
    )
    normalizer = fit_robust_normalizer(values)
    transformed = normalizer.transform(values)
    assert transformed.shape == (4, 2)
    assert torch.isfinite(transformed).all()
    norms = transformed.norm(dim=1)
    assert bool((norms <= 1.0 + 1e-5).all())
    assert int((norms > 0.99).sum()) >= 3


def test_event_label_balancing_is_deterministic_and_capped() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    events = ("a",) * 6 + ("b",) * 4
    first = event_label_balanced_indices(labels, events, 2, seed=7)
    second = event_label_balanced_indices(labels, events, 2, seed=7)
    assert torch.equal(first, second)
    for event in ("a", "b"):
        for label in (0, 1):
            count = sum(events[int(i)] == event and int(labels[int(i)]) == label for i in first)
            assert count <= 2


def test_cross_event_retrieval_predicts_utility_without_self_neighbor() -> None:
    # Every event contains a negative key near [-1,0] and a positive key near [1,0].
    keys = torch.tensor(
        [
            [-1.0, 0.00], [1.0, 0.00],
            [-1.0, 0.05], [1.0, 0.05],
            [-1.0, -0.05], [1.0, -0.05],
        ]
    )
    keys = torch.nn.functional.normalize(keys, dim=1)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    utilities = torch.tensor([-0.8, 0.8, -0.7, 0.7, -0.9, 0.9])
    events = ("a", "a", "b", "b", "c", "c")
    ids = tuple(f"item-{i}" for i in range(6))
    memory = build_memory(keys, labels, utilities, events, ids, torch.arange(6))
    retrieval = retrieve_topk_by_utility_label(
        keys, events, ids, memory,
        max_k=2,
        max_neighbors_per_event=1,
        exclude_same_event=True,
        chunk_size=3,
    )
    neighbor_indices = retrieval["indices"]
    for row, event in enumerate(events):
        for index in neighbor_indices[row].flatten().tolist():
            assert memory.events[index] != event
            assert memory.instance_ids[index] != ids[row]
    signals = scores_from_neighbors(retrieval, k=2, temperature=0.1)
    report = evaluate_scores(labels, signals["utility_vote"], events)
    assert report["auroc"] == 1.0
    assert bool(signals["available"].all())
