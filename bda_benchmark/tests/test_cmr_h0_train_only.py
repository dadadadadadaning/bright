# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_hard_mining_utils import (
    event_diverse_cross_event_topk,
    hard_mining_audit,
    hard_mining_gate,
    select_missed_damaged_anchors,
)


class CMRH0TrainOnlyTests(unittest.TestCase):
    def test_anchor_contract_selects_only_p3_missed_damaged(self):
        labels = torch.tensor([0, 1, 1, 2, 1])
        prediction = torch.tensor([0, 0, 1, 2, 0])
        selected = select_missed_damaged_anchors(labels, prediction)
        self.assertEqual(selected.tolist(), [1, 4])

    def test_cross_event_topk_excludes_query_event_and_caps_sources(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        query_events = torch.tensor([0, 1])
        reference = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [0.8, 0.2], [0.1, 0.9],
            [0.7, 0.3], [0.2, 0.8],
            [0.6, 0.4], [0.3, 0.7],
        ])
        reference_events = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        reference_indices = torch.arange(10, 18)
        indices, scores, unique = event_diverse_cross_event_topk(
            query,
            query_events,
            reference,
            reference_events,
            reference_indices,
            topk=3,
            max_per_event=1,
            device="cpu",
        )
        self.assertEqual(indices.shape, (2, 3))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertEqual(unique.tolist(), [3, 3])
        local = indices - 10
        for row, event in zip(local, query_events):
            self.assertFalse(bool((reference_events[row] == event).any()))

    def test_audit_rejects_label_or_same_event_leakage(self):
        feature = torch.eye(6)
        labels = torch.tensor([1, 1, 1, 0, 0, 0])
        prediction = torch.tensor([0, 1, 1, 0, 0, 0])
        events = torch.tensor([0, 1, 2, 1, 2, 3])
        with self.assertRaises(RuntimeError):
            hard_mining_audit(
                feature,
                labels,
                prediction,
                events,
                torch.tensor([0]),
                torch.tensor([[0, 2]]),  # first positive leaks anchor event
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([2]),
                torch.tensor([[3, 4]]),
                torch.tensor([[0.0, 0.0]]),
                torch.tensor([2]),
                max_per_event=1,
                seed=3,
            )

    def test_pair_margin_is_diagnostic_not_a_gate(self):
        audit = {
            "missed_damaged_anchors": 200,
            "anchor_events": 12,
            "complete_positive_rows_fraction": 1.0,
            "complete_negative_rows_fraction": 1.0,
            "positive_unique_events_mean": 5.0,
            "positive_similarity_gain_over_random": 0.05,
            "maximum_anchor_event_share": 0.20,
            "top1_positive_beats_hard_intact_fraction": 0.10,
        }
        self.assertTrue(all(hard_mining_gate(audit).values()))


if __name__ == "__main__":
    unittest.main()

