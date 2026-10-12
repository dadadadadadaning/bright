import torch

from .core import (ResidualCalibrator, bounded_event_class_rows,
                   hierarchical_cross_event_topk, memory_residual_probability,
                   p3_diagnostics)


def test_zero_initialized_calibrator_is_exact_probability_identity():
    p = torch.softmax(torch.randn(5, 4), 1)
    model = ResidualCalibrator(p3_diagnostics(p).shape[1], 16)
    assert torch.allclose(model(p3_diagnostics(p), p).softmax(1), p, atol=1e-6)


def test_bounded_rows_respect_each_event_class_capacity():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    events = ["a"] * 6
    rows = bounded_event_class_rows(labels, events, 2, 7)
    assert int((labels[rows] == 0).sum()) == 2
    assert int((labels[rows] == 1).sum()) == 2


def test_hierarchical_retrieval_excludes_query_event():
    q = torch.tensor([[1., 0.]])
    m = torch.tensor([[1., 0.], [.9, .1]])
    indices, _ = hierarchical_cross_event_topk(q, m, q, m, q, m, ["a"], ["a", "b"], coarse_k=2, topk=1)
    assert int(indices[0, 0]) == 1


def test_zero_beta_keeps_base_probability():
    p = torch.tensor([[.1, .2, .3, .4]])
    out = memory_residual_probability(p, torch.tensor([[0]]), torch.tensor([[1.]]), p, p, beta=0)
    assert torch.allclose(out, p)

