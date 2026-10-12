# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.multiprototype import (
    adjusted_rand_index,
    cross_event_similarity_gain,
    normalised_mutual_information,
    prototype_event_diagnostics,
    spherical_kmeans,
)


class CMRM0MultiPrototypeTests(unittest.TestCase):
    def test_spherical_kmeans_recovers_two_modes(self):
        torch.manual_seed(7)
        first = torch.randn(40, 4) * 0.02 + torch.tensor([1.0, 0, 0, 0])
        second = torch.randn(40, 4) * 0.02 + torch.tensor([0.0, 1, 0, 0])
        assignment, centres = spherical_kmeans(torch.cat((first, second)), 2)
        self.assertEqual(tuple(centres.shape), (2, 4))
        self.assertGreater(adjusted_rand_index(assignment, torch.tensor([0] * 40 + [1] * 40)), 0.95)

    def test_event_nmi_detects_event_only_clusters(self):
        event = torch.tensor([0] * 10 + [1] * 10)
        self.assertGreater(normalised_mutual_information(event, event), 0.99)
        mixed = torch.tensor([0, 1] * 10)
        self.assertLess(normalised_mutual_information(mixed, event), 0.05)

    def test_prototype_event_contract_reports_dominance(self):
        assignment = torch.tensor([0, 0, 0, 1, 1, 1])
        event = torch.tensor([0, 0, 1, 0, 1, 2])
        rows = prototype_event_diagnostics(assignment, event, clusters=2)
        self.assertAlmostEqual(rows[0]["max_event_share"], 2 / 3)
        self.assertEqual(rows[1]["events"], 3)

    def test_cross_event_gain_is_positive_for_real_modes(self):
        features = torch.tensor([
            [1.0, 0.0], [0.99, 0.01], [1.0, 0.02],
            [0.0, 1.0], [0.01, 0.99], [0.02, 1.0],
        ])
        assignment = torch.tensor([0, 0, 0, 1, 1, 1])
        events = torch.tensor([0, 1, 2, 0, 1, 2])
        result = cross_event_similarity_gain(features, assignment, events, pairs=20, seed=3)
        self.assertGreater(result["similarity_gain"], 0.1)


if __name__ == "__main__":
    unittest.main()

