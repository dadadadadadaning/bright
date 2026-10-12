# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.metric import (
    CMRResidualMetric,
    selective_damaged_positive_mask,
    selective_damaged_supcon_loss,
)
from script.standard_ML.train_CMR_M1B_selective import selective_metric_gate


class CMRM1BSelectiveTests(unittest.TestCase):
    def test_damaged_uses_only_raw_topk_cross_event_positive(self):
        # Four Damaged samples.  For anchor 0, sample 1 is the closest eligible
        # cross-event sample; sample 2 is same-event and sample 3 must be ignored.
        raw = torch.tensor([
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.00, 1.00],
            [-1.00, 0.00],
            [-0.99, 0.01],
        ])
        labels = torch.tensor([1, 1, 1, 1, 0, 0])
        events = torch.tensor([0, 1, 0, 2, 0, 1])
        positive = selective_damaged_positive_mask(
            raw, labels, events, damaged_topk=1
        )
        self.assertTrue(bool(positive[0, 1]))
        self.assertFalse(bool(positive[0, 2]))
        self.assertFalse(bool(positive[0, 3]))

    def test_nonselected_damaged_pair_is_not_a_negative(self):
        raw = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
            [-1.0, 0.0], [-0.9, 0.1], [0.0, -1.0],
        ])
        labels = torch.tensor([1, 1, 1, 0, 0, 0])
        events = torch.tensor([0, 1, 2, 0, 1, 2])
        positive = selective_damaged_positive_mask(
            raw, labels, events, damaged_topk=1
        )
        # Exactly one positive is selected for every Damaged anchor.
        self.assertEqual(positive[:3].sum(dim=1).tolist(), [1, 1, 1])
        # The loss remains finite even though other same-class pairs are ignored.
        loss = selective_damaged_supcon_loss(
            raw.clone().requires_grad_(True),
            raw,
            labels,
            events,
            damaged_topk=1,
        )
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_identity_adapter_receives_selective_loss_gradient(self):
        torch.manual_seed(3407)
        raw = torch.randn(12, 8)
        labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        events = torch.tensor([0, 1, 2, 3] * 3)
        model = CMRResidualMetric(8, hidden_dim=6)
        loss = selective_damaged_supcon_loss(
            model(raw), raw, labels, events, damaged_topk=2
        )
        loss.backward()
        self.assertGreater(float(model.adapter[-1].weight.grad.abs().sum()), 0.0)

    def test_gate_protects_intact_and_damaged_recall(self):
        raw = {
            "macro_mAP@10": 0.80,
            "per_class": {
                0: {"R@1": 0.75},
                1: {"R@1": 0.73, "R@5": 0.91},
                2: {"R@1": 0.83},
            },
        }
        candidate = {
            "macro_mAP@10": 0.81,
            "per_class": {
                0: {"R@1": 0.75},
                1: {"R@1": 0.75, "R@5": 0.92},
                2: {"R@1": 0.82},
            },
        }
        checks = selective_metric_gate(
            candidate,
            raw,
            {"delta_cross_event": 0.16},
            {"delta_cross_event": 0.15},
            ap_k=10,
            minimum_damaged_gain=0.01,
            maximum_destroyed_drop=0.02,
        )
        self.assertTrue(all(checks.values()))
        candidate["per_class"][0]["R@1"] = 0.74
        checks = selective_metric_gate(
            candidate,
            raw,
            {"delta_cross_event": 0.16},
            {"delta_cross_event": 0.15},
            ap_k=10,
            minimum_damaged_gain=0.01,
            maximum_destroyed_drop=0.02,
        )
        self.assertFalse(checks["intact_r1_nondecreasing"])


if __name__ == "__main__":
    unittest.main()
