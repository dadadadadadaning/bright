# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_asymmetric_rescue_utils import (
    one_way_accept_mask,
    promote_selected_to_front,
    rescue_complement,
    select_locked_threshold,
)


class CMRM4AAsymmetricTests(unittest.TestCase):
    def test_raw_damaged_prediction_can_never_be_changed(self):
        raw = torch.tensor([[1, 0, 2], [0, 1, 2]])
        model = torch.tensor([[0, 1, 2], [1, 0, 2]])
        accept = one_way_accept_mask(
            raw, model, torch.tensor([1.0, 1.0]), threshold=0.0
        )
        self.assertEqual(accept.tolist(), [False, True])

    def test_only_accepted_row_promotes_declared_d_candidate(self):
        raw = torch.tensor([[0, 2, 1], [2, 1, 0]])
        result = promote_selected_to_front(
            raw, torch.tensor([2, 1]), torch.tensor([True, False])
        )
        self.assertEqual(result[0].tolist(), [1, 0, 2])
        self.assertEqual(result[1].tolist(), raw[1].tolist())

    def test_complement_counts_rescue_and_harm_against_raw(self):
        raw = torch.tensor([[0, 1], [0, 1], [1, 0]])
        final = torch.tensor([[1, 0], [1, 0], [1, 0]])
        truth = torch.tensor([1, 0, 1])
        events = torch.tensor([0, 0, 1])
        audit = rescue_complement(
            raw, final, truth, events, torch.tensor([True, True, False])
        )
        self.assertEqual(audit["rescued"], 1)
        self.assertEqual(audit["harmed"], 1)
        self.assertEqual(audit["net_correction"], 0)
        self.assertEqual(audit["damaged_rescued"], 1)
        self.assertEqual(audit["intact_harmed"], 1)

    def test_threshold_selector_prefers_passing_train_point(self):
        candidates = [
            {
                "threshold": 0.0,
                "passed": False,
                "complement": {
                    "net_correction": 20,
                    "damaged_rescued": 30,
                    "accepted_precision": 0.4,
                    "accepted": 80,
                },
            },
            {
                "threshold": 0.01,
                "passed": True,
                "complement": {
                    "net_correction": 5,
                    "damaged_rescued": 12,
                    "accepted_precision": 0.6,
                    "accepted": 20,
                },
            },
        ]
        selected = select_locked_threshold(candidates)
        self.assertEqual(selected["threshold"], 0.01)


if __name__ == "__main__":
    unittest.main()

