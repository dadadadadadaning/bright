import unittest

import numpy as np
import torch

from bda_benchmark.script.standard_ML.q1_r0_runtime import (
    RegionProtocolAccumulator,
    aggregate_probabilities,
    connected_components,
    decide_r0,
)


class Q1R0Tests(unittest.TestCase):
    def test_connected_components_separates_diagonal_under_four_connectivity(self):
        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        _, four = connected_components(mask, connectivity=4)
        _, eight = connected_components(mask, connectivity=8)
        self.assertEqual(four, 2)
        self.assertEqual(eight, 1)

    def test_mean_max_and_topk_capture_different_evidence(self):
        probability = torch.tensor([
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.90, 0.05, 0.05],
        ])
        mean = aggregate_probabilities(probability, "mean")
        maximum = aggregate_probabilities(probability, "max")
        top = aggregate_probabilities(probability, "top20")
        self.assertEqual(int(mean.argmax()), 0)
        self.assertGreater(float(maximum[1]), float(mean[1]))
        self.assertGreater(float(top[1]), float(mean[1]))

    def test_lse_is_between_mean_and_max(self):
        probability = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        mean = aggregate_probabilities(probability, "mean")
        maximum = aggregate_probabilities(probability, "max")
        lse = aggregate_probabilities(probability, "lse", lse_beta=5.0)
        self.assertTrue(torch.all(lse >= mean - 1e-6))
        self.assertTrue(torch.all(lse <= maximum + 1e-6))

    def test_mixed_region_is_excluded(self):
        accumulator = RegionProtocolAccumulator()
        probability = torch.full((3, 2, 2), 1 / 3)
        damage = torch.tensor([[1, 1], [2, 2]])
        support = torch.ones(2, 2, dtype=torch.bool)
        accumulator.add_image(
            probability, damage, support, "event-a_0001",
            minimum_pixels=1, minimum_building_fraction=1.0,
            minimum_class_purity=0.75, lse_beta=10, area_threshold=0.5,
        )
        result = accumulator.compute()
        self.assertEqual(result["eligible_regions"], 0)
        self.assertEqual(result["mixed_or_background_regions"], 1)

    def test_decision_requires_sparse_gain_and_event_consistency(self):
        def entry(damaged, macro, event):
            return {
                "region": {"damaged_recall": damaged, "macro_f1": macro},
                "events": {f"e{i}": {"macro_f1": event} for i in range(14)},
                "event_macro_f1": event,
            }
        methods = {
            "mean": entry(0.30, 0.60, 0.50),
            "max": entry(0.31, 0.59, 0.49),
            "top10": entry(0.34, 0.60, 0.51),
            "top20": entry(0.33, 0.60, 0.50),
            "lse": entry(0.32, 0.60, 0.50),
            "area": entry(0.31, 0.60, 0.50),
        }
        decision = decide_r0({"methods": methods})
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["best_local_method"], "top10")


if __name__ == "__main__":
    unittest.main()
