import unittest

import torch

from model.cmr_v1.set_matcher import (
    dn4_score,
    frn_reconstruction_score,
    sinkhorn_emd_score,
    symmetric_chamfer_score,
)


class TestCMRV1SetMatcher(unittest.TestCase):
    def setUp(self):
        self.q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.same = self.q.clone()
        self.other = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.mask = torch.tensor([True, True])

    def test_same_scores_higher(self):
        methods = [
            lambda a, b: symmetric_chamfer_score(a, self.mask, b, self.mask),
            lambda a, b: dn4_score(a, self.mask, b, self.mask, neighbor_k=1),
            lambda a, b: sinkhorn_emd_score(a, self.mask, b, self.mask, epsilon=0.1),
            lambda a, b: frn_reconstruction_score(a, self.mask, b, self.mask, ridge=1e-2),
        ]
        for method in methods:
            self.assertGreater(float(method(self.q, self.same)), float(method(self.q, self.other)))

    def test_padding_is_ignored(self):
        q = torch.cat([self.q, torch.tensor([[99.0, 99.0]])], dim=0)
        s = torch.cat([self.same, torch.tensor([[-99.0, -99.0]])], dim=0)
        mask = torch.tensor([True, True, False])
        score = symmetric_chamfer_score(q, mask, s, mask)
        self.assertAlmostEqual(float(score), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
