# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.pairwise import class_balanced_hard_top1_loss
from script.standard_ML.cmr_pairwise_utils import per_event_r1_diagnostics


class CMRM3TTop1Tests(unittest.TestCase):
    def test_loss_prefers_best_positive_above_hardest_negative(self):
        labels = torch.tensor([[1, 0, 1]])
        truth = torch.tensor([1])
        good = class_balanced_hard_top1_loss(
            torch.tensor([[0.9, 0.2, 0.1]]), labels, truth
        )
        bad = class_balanced_hard_top1_loss(
            torch.tensor([[0.1, 0.9, 0.2]]), labels, truth
        )
        self.assertLess(float(good), float(bad))

    def test_loss_is_finite_and_differentiable(self):
        score = torch.tensor(
            [[0.1, 0.2, 0.0], [0.3, 0.1, 0.2], [0.0, 0.4, 0.2]],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1, 2], [2, 0, 1], [0, 1, 2]])
        truth = torch.tensor([1, 2, 0])
        loss = class_balanced_hard_top1_loss(score, labels, truth)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(score.grad)

    def test_event_details_report_sample_count_and_net_correct(self):
        truth = torch.tensor([1, 1, 1, 1])
        events = torch.tensor([0, 0, 0, 1])
        raw = torch.tensor([[1, 0], [0, 1], [0, 1], [0, 1]])
        new = torch.tensor([[1, 0], [1, 0], [0, 1], [1, 0]])
        result = per_event_r1_diagnostics(raw, new, truth, events)
        event0 = result["damaged_event_details"]["0"]
        event1 = result["damaged_event_details"]["1"]
        self.assertEqual(event0["queries"], 3)
        self.assertEqual(event0["net_correct"], 1)
        self.assertEqual(event1["queries"], 1)
        self.assertEqual(event1["net_correct"], 1)


if __name__ == "__main__":
    unittest.main()

