# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np
import torch

from script.standard_ML.p3_direct_occurrence_runtime import (
    hard_region_boundary_loss,
    region_occurrence_margin,
    select_hard_region_indices,
)


class P3O2HardBoundaryTests(unittest.TestCase):
    def test_region_margin_uses_only_exact_component(self):
        logits = torch.zeros(1, 4, 2, 2, requires_grad=True)
        logits.data[:, 1] = 2.0
        logits.data[:, 2, 0, 0] = 6.0
        mask = np.zeros((2, 2), dtype=np.bool_)
        mask[0, 0] = True
        margin = region_occurrence_margin(logits, mask, batch_index=0)
        margin.backward()
        self.assertGreater(float(margin), 0.0)
        self.assertGreater(float(logits.grad[:, :, 0, 0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, :, 1, 1].abs().sum()), 0.0)

    def test_boundary_loss_pushes_classes_in_opposite_directions(self):
        damaged = torch.tensor([-1.0], requires_grad=True)
        intact = torch.tensor([1.0], requires_grad=True)
        loss, _ = hard_region_boundary_loss(damaged, intact)
        loss.backward()
        self.assertLess(float(damaged.grad), 0.0)
        self.assertGreater(float(intact.grad), 0.0)

    def test_hard_pool_uses_no_feature_similarity_and_caps_events(self):
        labels = torch.tensor([1, 1, 1, 0, 0, 0, 1, 0])
        predictions = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        probabilities = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.5, 0.4, 0.1],
                [0.6, 0.3, 0.1],
                [0.8, 0.1, 0.1],
                [0.4, 0.5, 0.1],
            ]
        )
        events = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        positive, negative, audit = select_hard_region_indices(
            labels,
            predictions,
            probabilities,
            events,
            cap_per_event=2,
            seed=3,
        )
        self.assertEqual(int((events[positive] == 0).sum()), 2)
        self.assertEqual(int((events[negative] == 0).sum()), 2)
        # Highest Affected probabilities for event 0 are rows 4 and 5.
        self.assertEqual(set(negative[events[negative] == 0].tolist()), {4, 5})
        self.assertFalse(audit["uses_feature_similarity"])
        self.assertFalse(audit["validation_accessed"])
        self.assertFalse(audit["test_accessed"])


if __name__ == "__main__":
    unittest.main()

