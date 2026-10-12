import torch

from bda_benchmark.experiments.spi_cross_event_metric_memory_v2.audit_m2 import (
    hierarchical_probabilities, reliability_gate,
)
from bda_benchmark.experiments.spi_cross_event_metric_memory_v2.core import (
    class_similarity_scores, cross_event_supervised_contrastive,
    selective_memory_probability, truth_distribution,
)


def test_truth_distribution_reports_soft_label_and_purity():
    delta = torch.zeros(1, 4, 4, dtype=torch.long)
    delta[0, 1, 2] = 2; delta[0, 2, 2] = 6; delta[0, 3, 2] = 2
    soft, hard, purity, foreground, keep = truth_distribution(torch.tensor([6]), delta)
    assert torch.allclose(soft, torch.tensor([[.2, .6, .2]]))
    assert int(hard[0]) == 1 and bool(keep[0])
    assert torch.allclose(purity, torch.tensor([.6]))
    assert torch.allclose(foreground, torch.ones(1))


def test_cross_event_retrieval_excludes_same_event():
    memory = torch.tensor([[1., 0.], [.8, .2], [0., 1.]])
    query = torch.tensor([[1., 0.]])
    score, available = class_similarity_scores(
        query, memory, torch.tensor([0, 0, 1]), torch.tensor([7]),
        torch.tensor([7, 8, 9]), topk=1, exclude_same_event=True,
    )
    assert int(available[0, 0]) == 1
    assert torch.allclose(score[0, 0], torch.tensor(.8 / (.8 ** 2 + .2 ** 2) ** .5))


def test_cross_event_contrast_requires_different_event_positive():
    embedding = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    loss = cross_event_supervised_contrastive(
        embedding, torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 1, 2])
    )
    assert torch.isfinite(loss) and float(loss) > 0


def test_hierarchical_occurrence_preserves_parametric_severity_ratio():
    base = torch.tensor([[.5, .3, .2]])
    retrieved = torch.tensor([[.2, .2, .6]])
    occurrence, severity, dual = hierarchical_probabilities(base, retrieved)
    assert torch.allclose(occurrence[:, 1] / occurrence[:, 2], base[:, 1] / base[:, 2])
    assert torch.allclose(severity[:, 1:].sum(1), base[:, 1:].sum(1))
    assert torch.equal(dual, retrieved)


def test_reliability_gate_is_zero_without_full_coverage():
    probability = torch.tensor([[.9, .05, .05], [.9, .05, .05]])
    availability = torch.tensor([[4, 4, 4], [4, 3, 4]])
    gate = reliability_gate(probability, availability, topk=4)
    assert float(gate[0]) > 0
    assert float(gate[1]) == 0


def test_selective_memory_only_changes_confident_disagreements():
    parametric = torch.tensor([[.70, .20, .10], [.40, .35, .25]])
    memory = torch.tensor([[.20, .70, .10], [.10, .20, .70]])
    availability = torch.full((2, 3), 4)
    output, intervene, _ = selective_memory_probability(
        parametric,
        memory,
        availability,
        topk=4,
        minimum_confidence_advantage=.05,
        minimum_memory_margin=.20,
    )
    assert not bool(intervene[0])
    assert bool(intervene[1])
    assert torch.equal(output[0], parametric[0])
    assert torch.equal(output[1], memory[1])
