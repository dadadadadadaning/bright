from __future__ import annotations
import torch

from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import Retrieval
from .core import (family_probability, high_risk_mask, source_quantile_thresholds,
                   uncertainty)


def test_uncertainty_is_one_minus_largest_p3_probability():
    probability = torch.tensor([[.1, .8, .05, .05], [.25, .25, .25, .25]])
    assert torch.allclose(uncertainty(probability), torch.tensor([.2, .75]))


def test_source_thresholds_are_fit_without_labels():
    probability = torch.tensor([[.9, .05, .03, .02], [.5, .45, .03, .02], [.1, .8, .05, .05], [.25, .25, .25, .25]])
    source = torch.tensor([0, 0, 1, 1]); threshold = source_quantile_thresholds(probability, source, .5)
    mask = high_risk_mask(probability, source, threshold)
    assert mask.tolist() == [False, True, False, True]


def test_family_probability_is_weighted_neighbour_vote():
    retrieval = Retrieval(torch.tensor([[0, 1]]), torch.tensor([[1., 1.]]), torch.zeros(1, 2), torch.ones(1, dtype=torch.bool))
    probability = family_probability(retrieval, torch.tensor([1, 3]))
    assert torch.allclose(probability[0, torch.tensor([1, 3])], torch.tensor([.5, .5]))
    assert float(probability.sum()) == 1.

