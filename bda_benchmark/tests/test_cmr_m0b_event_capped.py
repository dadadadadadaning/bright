# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.multiprototype import assign_to_centres, event_capped_indices


class CMRM0BEventCappedTests(unittest.TestCase):
    def test_event_cap_never_repeats_small_events(self):
        events = torch.tensor([0] * 10 + [1] * 3 + [2])
        indices = event_capped_indices(events, cap_per_event=4, seed=3)
        selected = events[indices]
        self.assertEqual(torch.bincount(selected, minlength=3).tolist(), [4, 3, 1])
        self.assertEqual(torch.unique(indices).numel(), indices.numel())

    def test_assignment_uses_cosine_nearest_centre(self):
        feature = torch.tensor([[1., 0.], [0., 1.], [.9, .1]])
        centres = torch.tensor([[1., 0.], [0., 1.]])
        self.assertEqual(assign_to_centres(feature, centres).tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()

