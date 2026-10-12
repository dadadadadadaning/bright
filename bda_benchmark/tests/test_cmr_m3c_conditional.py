# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.pairwise import CMRCandidateConditionalPairwiseScorer
from script.standard_ML.cmr_pairwise_utils import per_event_r1_diagnostics
from script.standard_ML.train_CMR_M3C_conditional import admission_checks


class CMRM3CConditionalTests(unittest.TestCase):
    def test_zero_initialisation_preserves_every_candidate_class(self):
        model = CMRCandidateConditionalPairwiseScorer(4, hidden_dim=8)
        query = torch.randn(2, 3, 4)
        candidate = torch.randn(2, 3, 4)
        labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
        residual = model(query, candidate, labels)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_candidate_label_routes_to_its_declared_head(self):
        model = CMRCandidateConditionalPairwiseScorer(
            2, hidden_dim=4, maximum_residual=0.05
        )
        with torch.no_grad():
            model.class_heads[0].bias.fill_(-2.0)
            model.class_heads[1].bias.fill_(0.0)
            model.class_heads[2].bias.fill_(2.0)
        query = torch.randn(1, 3, 2)
        candidate = torch.randn(1, 3, 2)
        residual = model(query, candidate, torch.tensor([[0, 1, 2]]))
        self.assertLess(float(residual[0, 0]), float(residual[0, 1]))
        self.assertLess(float(residual[0, 1]), float(residual[0, 2]))

    def test_query_labels_are_not_a_forward_argument(self):
        model = CMRCandidateConditionalPairwiseScorer(2, hidden_dim=4)
        with self.assertRaises(ValueError):
            model(
                torch.randn(1, 2),
                torch.randn(1, 2),
                torch.tensor([3]),
            )

    def test_damaged_event_metric_is_not_hidden_by_i_x_gain(self):
        # Each event has I/D/X. I and X improve, while D is harmed.
        truth = torch.tensor([0, 1, 2, 0, 1, 2])
        events = torch.tensor([0, 0, 0, 1, 1, 1])
        raw = torch.tensor([
            [1, 0], [1, 0], [0, 2],
            [1, 0], [1, 0], [0, 2],
        ])
        new = torch.tensor([
            [0, 1], [0, 1], [2, 0],
            [0, 1], [0, 1], [2, 0],
        ])
        result = per_event_r1_diagnostics(raw, new, truth, events)
        self.assertEqual(result["macro_nonnegative_events"], 2)
        self.assertEqual(result["damaged_nonnegative_events"], 0)

    def test_gate_uses_damaged_specific_event_count(self):
        baseline = {
            "macro_mAP@10": 0.80,
            "per_class": {
                0: {"R@1": 0.75, "R@5": 0.90},
                1: {"R@1": 0.73, "R@5": 0.91},
                2: {"R@1": 0.82, "R@5": 0.93},
            },
        }
        candidate = {
            "macro_mAP@10": 0.81,
            "per_class": {
                0: {"R@1": 0.76, "R@5": 0.91},
                1: {"R@1": 0.75, "R@5": 0.92},
                2: {"R@1": 0.82, "R@5": 0.93},
            },
        }
        checks = admission_checks(
            candidate,
            baseline,
            {"damaged_nonnegative_events": 9},
            minimum_damaged_gain=0.01,
            maximum_destroyed_drop=0.01,
            minimum_damaged_events=10,
        )
        self.assertFalse(checks["damaged_nonnegative_events"])


if __name__ == "__main__":
    unittest.main()

