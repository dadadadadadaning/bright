# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_metric_utils import cross_event_retrieval_eval
from script.standard_ML.train_CMR_M1BG_graph import (
    sample_cross_event_negatives,
    severity_metric_gate,
)


class CMRM1BGSSeverityTests(unittest.TestCase):
    def test_intact_only_negative_sampler_never_draws_destroyed(self):
        labels = torch.tensor([0, 0, 0, 2, 2, 2, 1, 1])
        events = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        indices = sample_cross_event_negatives(
            labels,
            events,
            torch.tensor([0, 1]),
            negatives_per_class=2,
            negative_classes=(0,),
            generator=torch.Generator().manual_seed(4),
        )
        self.assertEqual(indices.shape, (2, 2))
        self.assertTrue(bool((labels[indices] == 0).all()))
        for row, anchor_event in zip(indices, (0, 1)):
            self.assertFalse(bool((events[row] == anchor_event).any()))

    def test_retrieval_reports_per_event_macro_map(self):
        reference = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 2])
        events = torch.tensor([0, 1, 0, 1])
        result = cross_event_retrieval_eval(
            reference,
            labels,
            events,
            reference,
            labels,
            events,
            k_values=(1, 2),
            ap_k=2,
            query_chunk=2,
            device="cpu",
            print_progress=False,
        )
        self.assertEqual(set(result["per_event"]), {"0", "1"})
        self.assertIn("macro_mAP@2", result["per_event"]["0"])

    def test_severity_gate_uses_event_stability_not_delta_ce(self):
        baseline = {
            "macro_mAP@10": 0.80,
            "per_class": {
                0: {"R@1": 0.75},
                1: {"R@1": 0.73, "R@5": 0.91},
                2: {"R@1": 0.83},
            },
            "per_event": {
                str(event): {"macro_mAP@10": 0.50} for event in range(14)
            },
        }
        candidate = {
            "macro_mAP@10": 0.81,
            "per_class": {
                0: {"R@1": 0.76},
                1: {"R@1": 0.75, "R@5": 0.92},
                2: {"R@1": 0.82},
            },
            "per_event": {
                str(event): {"macro_mAP@10": 0.51 if event < 10 else 0.49}
                for event in range(14)
            },
        }
        checks, diagnostics = severity_metric_gate(
            candidate,
            baseline,
            ap_k=10,
            minimum_damaged_gain=0.01,
            maximum_destroyed_drop=0.01,
            minimum_nonnegative_events=10,
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(diagnostics["nonnegative_events"], 10)


if __name__ == "__main__":
    unittest.main()
