# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_event_capped_sampler import (
    audit_event_capped_schedules,
    build_event_capped_epoch,
    event_occurrence_statistics,
    h0b_gate,
)


class CMRH0BEventCapTests(unittest.TestCase):
    def test_epoch_caps_large_events_without_repeating_rows(self):
        events = torch.tensor([0] * 10 + [1] * 4 + [2] * 2)
        rows = build_event_capped_epoch(
            events, cap_per_event=3, seed=7, epoch=0
        )
        selected_events = events[rows]
        self.assertEqual(rows.numel(), 8)
        self.assertEqual(torch.unique(rows).numel(), rows.numel())
        self.assertEqual(int((selected_events == 0).sum()), 3)
        self.assertEqual(int((selected_events == 1).sum()), 3)
        self.assertEqual(int((selected_events == 2).sum()), 2)

    def test_large_event_rotates_to_unseen_rows_across_epochs(self):
        events = torch.tensor([0] * 10 + [1] * 3)
        first = build_event_capped_epoch(events, cap_per_event=4, seed=11, epoch=0)
        second = build_event_capped_epoch(events, cap_per_event=4, seed=11, epoch=1)
        first_large = set(first[events[first] == 0].tolist())
        second_large = set(second[events[second] == 0].tolist())
        self.assertEqual(len(first_large), 4)
        self.assertEqual(len(second_large), 4)
        self.assertTrue(bool(second_large - first_large))

    def test_occurrence_rate_uses_event_damaged_denominator(self):
        labels = torch.tensor([1, 1, 1, 0, 1, 1, 0])
        predictions = torch.tensor([0, 1, 0, 0, 0, 1, 0])
        events = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        anchors = torch.tensor([0, 2, 4])
        stats = event_occurrence_statistics(
            labels,
            predictions,
            events,
            {"a": 0, "b": 1},
            anchors,
            cap_per_event=1,
        )
        self.assertAlmostEqual(stats["per_event"]["a"]["damaged_to_intact_rate"], 2 / 3)
        self.assertAlmostEqual(stats["per_event"]["b"]["damaged_to_intact_rate"], 1 / 2)
        self.assertEqual(stats["effective_epoch_anchor_count"], 2)

    def test_schedule_audit_reports_no_replacement(self):
        events = torch.tensor([0] * 12 + [1] * 8 + [2] * 5)
        audit = audit_event_capped_schedules(
            events, cap_per_event=4, seed=19, audit_epochs=3
        )
        self.assertTrue(audit["no_replacement_within_every_epoch"])
        self.assertEqual(audit["anchors_per_epoch"], 12)
        self.assertGreater(audit["unique_anchors_seen_across_audit_epochs"], 12)

    def test_gate_requires_core_pairs_and_capped_distribution(self):
        checks = h0b_gate(
            {"positive": True, "negative": True},
            {"effective_training_events": 14, "effective_maximum_event_share": 0.12},
            {"no_replacement_within_every_epoch": True, "anchors_per_epoch": 563},
        )
        self.assertTrue(all(checks.values()))
        failed = h0b_gate(
            {"positive": True, "negative": False},
            {"effective_training_events": 14, "effective_maximum_event_share": 0.12},
            {"no_replacement_within_every_epoch": True, "anchors_per_epoch": 563},
        )
        self.assertFalse(failed["all_core_h0_pair_checks_pass"])


if __name__ == "__main__":
    unittest.main()

