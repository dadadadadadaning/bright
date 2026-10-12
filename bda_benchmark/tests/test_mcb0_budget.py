# -*- coding: utf-8 -*-
import unittest

import torch

from script.standard_ML.mcb0_budget_runtime import (
    blended_budget_indices,
    candidate_table,
    evaluate_budgets,
    selection_complement,
)


class MCB0BudgetTests(unittest.TestCase):
    def test_candidate_pool_is_p3_intact_only(self):
        cache = {
            "labels": torch.tensor([0, 1, 2, 0]),
            "base_predictions": torch.tensor([0, 0, 1, 0]),
            "base_probabilities": torch.tensor([
                [0.8, 0.1, 0.1], [0.7, 0.2, 0.1],
                [0.1, 0.1, 0.8], [0.6, 0.3, 0.1],
            ]),
            "events": ["a", "a", "b", "b"],
            "image_ids": ["i0", "i1", "i2", "i3"],
            "key_variants": {"mean": torch.eye(4)},
        }
        table = candidate_table(cache)
        self.assertEqual(table["indices"].tolist(), [0, 1, 3])
        self.assertEqual(table["truth"].tolist(), [0, 1, 0])

    def test_blend_preserves_exact_budget_and_uses_unique_memory_rows(self):
        blend = blended_budget_indices(
            torch.tensor([0.9, 0.8, 0.7, 0.1, 0.0]),
            torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8]),
            budget=5,
            p3_fraction=0.8,
        )
        self.assertEqual(blend["p3_selected"].tolist(), [0, 1, 2, 3])
        self.assertEqual(blend["memory_unique_selected"].tolist(), [4])
        self.assertEqual(torch.unique(blend["selected"]).numel(), 5)

    def test_complement_counts_unique_damaged_and_event_spread(self):
        report = selection_complement(
            torch.tensor([0, 2, 3]),
            torch.tensor([0, 1, 4]),
            torch.tensor([0, 0, 1, 1, 0]),
            ["a", "a", "b", "c", "d"],
        )
        self.assertEqual(report["overlap"], 1)
        self.assertEqual(report["unique_damaged_rescued"], 2)
        self.assertEqual(report["unique_damaged_events"], 2)

    def test_every_method_has_same_workload(self):
        result = evaluate_budgets(
            torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
            torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8]),
            torch.tensor([1, 0, 0, 1, 2]),
            ["a", "a", "b", "b", "c"],
            budgets=(5,),
            p3_fraction=0.8,
        )["5"]
        self.assertEqual(
            {method["workload"] for method in result["methods"].values()}, {5}
        )


if __name__ == "__main__":
    unittest.main()
