# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_global_graph_utils import (
    build_event_diverse_graph,
    damaged_anchor_nce_loss,
    graph_audit,
)
from script.standard_ML.train_CMR_M1BG_graph import sample_cross_event_negatives


class CMRM1BGGlobalGraphTests(unittest.TestCase):
    def test_graph_excludes_same_event_and_respects_event_cap(self):
        feature = torch.tensor([
            [1.00, 0.00], [0.99, 0.01],
            [0.95, 0.05], [0.94, 0.06],
            [0.00, 1.00], [0.01, 0.99],
        ])
        events = torch.tensor([0, 0, 1, 1, 2, 2])
        neighbours, scores = build_event_diverse_graph(
            feature,
            events,
            topk=2,
            max_per_event=1,
            chunk_size=3,
            device="cpu",
        )
        self.assertEqual(neighbours.shape, (6, 2))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        for anchor, row in enumerate(neighbours):
            self.assertFalse(bool((events[row] == events[anchor]).any()))
            self.assertEqual(torch.unique(events[row]).numel(), 2)

    def test_graph_audit_detects_nearest_over_random_separation(self):
        torch.manual_seed(7)
        feature = []
        events = []
        for event in range(6):
            for mode in range(4):
                centre = torch.zeros(8)
                centre[mode] = 1.0
                feature.append(centre + 0.01 * torch.randn(8))
                events.append(event)
        feature = torch.stack(feature)
        events = torch.tensor(events)
        neighbours, scores = build_event_diverse_graph(
            feature, events, topk=5, max_per_event=1, device="cpu"
        )
        audit = graph_audit(
            feature,
            events,
            neighbours,
            scores,
            max_per_event=1,
            seed=7,
        )
        self.assertGreater(audit["similarity_gap"], 0.2)
        self.assertGreaterEqual(audit["anchors_with_at_least_5_events"], 0.95)

    def test_damaged_anchor_nce_is_finite_and_differentiable(self):
        torch.manual_seed(9)
        anchor = torch.randn(4, 8, requires_grad=True)
        positive = torch.randn(4, 8, requires_grad=True)
        negatives = torch.randn(4, 6, 8, requires_grad=True)
        loss = damaged_anchor_nce_loss(anchor, positive, negatives)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(anchor.grad.abs().sum()), 0.0)

    def test_negative_sampler_excludes_anchor_event_and_balances_classes(self):
        labels = torch.tensor([0, 0, 0, 2, 2, 2, 1, 1])
        events = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        anchor_events = torch.tensor([0, 1])
        indices = sample_cross_event_negatives(
            labels,
            events,
            anchor_events,
            negatives_per_class=2,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(indices.shape, (2, 4))
        for row, anchor_event in zip(indices, anchor_events):
            self.assertFalse(bool((events[row] == anchor_event).any()))
            self.assertEqual(int((labels[row] == 0).sum()), 2)
            self.assertEqual(int((labels[row] == 2).sum()), 2)


if __name__ == "__main__":
    unittest.main()
