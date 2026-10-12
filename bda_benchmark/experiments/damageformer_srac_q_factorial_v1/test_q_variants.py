"""Synthetic cue-partition shape and gradient checks; no BRIGHT data is read."""

from __future__ import annotations

import unittest

import torch

from .q_variants import CuePartition


class CuePartitionTests(unittest.TestCase):
    def test_candidate_probabilities_normalize_and_cue_receives_gradient(self):
        torch.manual_seed(23)
        partition = CuePartition(in_dim=4, attn_dim=4, stride=2,
                                 iterations=2, cue_gain=0.1)
        structural = torch.randn(1, 4, 4, 4)
        building = torch.sigmoid(torch.randn(1, 1, 4, 4))
        cue = torch.randn(1, 512, 4, 4)
        result = partition(structural, building, cue)
        q = result["q"]
        self.assertEqual(tuple(q.shape), (1, 9, 16))
        self.assertEqual(result["count"], 4)
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.allclose(q.sum(dim=1), torch.ones_like(q[:, 0]), atol=1e-6))
        q[:, 4].square().sum().backward()
        gradient = partition.cue_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
