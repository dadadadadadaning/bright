# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.pairwise import CMRBoundedPairwiseScorer, multi_positive_candidate_loss
from script.standard_ML.cmr_pairwise_utils import (
    candidate_metrics,
    frozen_topk_candidates,
    oracle_gate,
    oracle_ranking,
)


class CMRM3PairwiseTests(unittest.TestCase):
    def test_zero_initialisation_exactly_preserves_raw_score(self):
        model = CMRBoundedPairwiseScorer(4, hidden_dim=8, maximum_residual=0.05)
        q = torch.randn(3, 5, 4)
        c = torch.randn(3, 5, 4)
        residual = model(q, c)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_pairwise_residual_is_strictly_bounded(self):
        model = CMRBoundedPairwiseScorer(2, hidden_dim=4, maximum_residual=0.03)
        with torch.no_grad():
            model.network[-1].bias.fill_(100.0)
        residual = model(torch.randn(8, 2), torch.randn(8, 2))
        self.assertTrue(bool((residual <= 0.03).all()))
        self.assertTrue(bool((residual >= -0.03).all()))

    def test_same_event_is_removed_before_topk(self):
        memory = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        indices, _ = frozen_topk_candidates(
            memory,
            torch.tensor([0, 1, 2]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            topk=2,
            query_chunk=1,
            device="cpu",
            print_progress=False,
        )
        self.assertNotIn(0, indices[0].tolist())

    def test_oracle_promotes_existing_correct_candidate_only(self):
        labels = torch.tensor([[0, 1, 2], [0, 2, 0]])
        truth = torch.tensor([1, 1])
        ranked = oracle_ranking(labels, truth)
        self.assertEqual(int(ranked[0, 0]), 1)
        self.assertEqual(ranked[1].tolist(), labels[1].tolist())

    def test_oracle_gate_requires_damaged_headroom_and_coverage(self):
        baseline = candidate_metrics(
            torch.tensor([[1, 0], [0, 1], [1, 0], [2, 0]]),
            torch.tensor([0, 1, 1, 2]),
            k_values=(1, 2), ap_k=2,
        )
        oracle = candidate_metrics(
            torch.tensor([[0, 1], [1, 0], [1, 0], [2, 0]]),
            torch.tensor([0, 1, 1, 2]),
            k_values=(1, 2), ap_k=2,
        )
        checks = oracle_gate(
            baseline, oracle, damaged_headroom=0.4, minimum_damaged_coverage=0.9
        )
        self.assertTrue(all(checks.values()))

    def test_listwise_loss_is_finite_and_differentiable(self):
        score = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.1, 0.2]], requires_grad=True)
        labels = torch.tensor([[0, 1, 2], [2, 0, 1]])
        truth = torch.tensor([1, 2])
        loss = multi_positive_candidate_loss(score, labels, truth)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(score.grad)


if __name__ == "__main__":
    unittest.main()
