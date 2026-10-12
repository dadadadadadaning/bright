# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_hierarchy_runtime import binary_metrics, event_binary_metrics


class CMRE0HierarchyTests(unittest.TestCase):
    def test_perfect_binary_scores_have_unit_metrics(self):
        result = binary_metrics(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1]))
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_event_macro_does_not_weight_large_event_more(self):
        score = torch.tensor([0.1, 0.9, 0.9, 0.1])
        target = torch.tensor([0, 1, 0, 1])
        result = event_binary_metrics(score, target, ["a", "a", "b", "b"])
        self.assertEqual(result["comparable_events"], 2)
        self.assertAlmostEqual(result["event_macro_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()

