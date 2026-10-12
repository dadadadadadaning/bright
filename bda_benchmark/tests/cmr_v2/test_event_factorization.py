import unittest

import torch

from model.cmr_v2.event_factorization import (
    event_expert_deltas,
    fit_event_factorization,
    frozen_topk_delta,
    transform_features,
)


class EventFactorizationTests(unittest.TestCase):
    def test_event_center_is_class_balanced(self):
        # Intact is deliberately over-represented.  The event center must be
        # mean(class means)=(0+3+6)/3=3, not the sample-weighted mean.
        features = torch.tensor([[0.0], [0.0], [0.0], [0.0], [3.0], [6.0]])
        labels = torch.tensor([0, 0, 0, 0, 1, 2])
        factor = fit_event_factorization(features, labels, ["a"] * 6)
        self.assertAlmostEqual(float(factor["centers"][0, 0]), 3.0, places=6)
        self.assertTrue(bool(factor["reliable"][0]))

    def test_missing_class_event_uses_global_fallback(self):
        features = torch.tensor([[0.0], [2.0], [4.0], [10.0], [12.0]])
        labels = torch.tensor([0, 1, 2, 0, 1])
        events = ["complete", "complete", "complete", "partial", "partial"]
        factor = fit_event_factorization(features, labels, events)
        partial = factor["event_order"].index("partial")
        self.assertFalse(bool(factor["reliable"][partial]))
        transformed = transform_features(features, events, factor, "centered")
        self.assertEqual(transformed.shape, features.shape)

    def test_same_and_excluded_event_protocols(self):
        memory = torch.tensor([
            [1.0, 0.0], [0.0, 1.0],  # A: I, D
            [1.0, 0.0], [0.0, 1.0],  # B: I, D
        ])
        labels = torch.tensor([0, 1, 0, 1])
        events = ["a", "a", "b", "b"]
        query = torch.tensor([[0.0, 1.0]])
        same = event_expert_deltas(
            query, ["a"], torch.tensor([100]), memory, labels, events,
            torch.arange(4), protocol="same_event", case_k=1,
        )
        excluded = event_expert_deltas(
            query, ["a"], torch.tensor([100]), memory, labels, events,
            torch.arange(4), protocol="exclude_same_event", case_k=1,
        )
        self.assertGreater(float(same["mean_delta"][0]), 0.0)
        self.assertGreater(float(excluded["mean_delta"][0]), 0.0)
        self.assertEqual(int(same["expert_count"][0]), 1)
        self.assertEqual(int(excluded["expert_count"][0]), 1)

    def test_frozen_topk_counterfactual_delta(self):
        scores = torch.tensor([[0.9, 0.8, 0.7]])
        labels = torch.tensor([[1, 1, 0]])
        result = frozen_topk_delta(scores, labels, case_k=1)
        self.assertAlmostEqual(float(result["delta"][0]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
