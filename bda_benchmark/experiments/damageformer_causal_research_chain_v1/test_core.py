from __future__ import annotations

import tempfile
import unittest

import torch

from .cache_features import building_feature_from_regions, wrong_region_feature
from .core import binary_auroc, coverage_curve, deterministic_partition, rescue_harm_from_scores, separation_ratio
from .prototype_utils import build_class_prototypes, score_prototypes
from .tail_losses import balanced_ce, blv_loss, logit_adjust_ce, seesaw_ce
from .tail_models import MultiPrototypeTail


class ContractTests(unittest.TestCase):
    def test_building_feature_shape_and_wrong_region_control(self):
        t = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        c = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.3, 0.4])
        good = building_feature_from_regions(t, c)
        bad = wrong_region_feature(t, c)
        self.assertEqual(tuple(good.shape), (16,))
        self.assertFalse(torch.equal(good, bad))

    def test_binary_auroc(self):
        s = torch.tensor([0.1, 0.2, 0.8, 0.9])
        y = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(binary_auroc(s, y), 1.0, places=6)

    def test_tail_losses_finite(self):
        logits = torch.randn(16, 3, requires_grad=True)
        labels = torch.tensor([0] * 10 + [1] * 4 + [2] * 2)
        counts = torch.tensor([100.0, 20.0, 10.0])
        losses = [
            balanced_ce(logits, labels, counts),
            blv_loss(logits, labels, counts, 1.0),
            logit_adjust_ce(logits, labels, counts, 1.0),
            seesaw_ce(logits, labels, counts),
        ]
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))
            loss.backward(retain_graph=True)

    def test_multi_prototype_contract(self):
        model = MultiPrototypeTail(32, 16, 2, 0.1)
        score, h = model(torch.randn(7, 32))
        self.assertEqual(tuple(score.shape), (7, 3))
        self.assertEqual(tuple(h.shape), (7, 16))
        self.assertTrue(torch.allclose(h.norm(dim=1), torch.ones(7), atol=1e-5))

    def test_prototype_scores(self):
        x = torch.randn(30, 8)
        y = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        p = build_class_prototypes(x, y, 2)
        s = score_prototypes(x, p)
        self.assertEqual(tuple(s.shape), (30, 3))

    def test_coverage_curve(self):
        score = torch.tensor([0.9, 0.8, 0.7, 0.1])
        cand = torch.ones(4, dtype=torch.bool)
        pos = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        neg = ~pos
        rows = coverage_curve(score, cand, pos, neg, (0.5,))
        self.assertEqual(rows[0]["selected"], 2)
        self.assertEqual(rows[0]["rescue"], 1)
        self.assertEqual(rows[0]["harm"], 1)

    def test_deterministic_partition_disjoint(self):
        names = [f"x{i}" for i in range(30)]
        events = ["e1"] * 15 + ["e2"] * 15
        labels = torch.tensor(([0, 1, 2] * 10))
        fit, cal = deterministic_partition(names, events, labels, 0.2, 3407)
        self.assertFalse(bool((fit & cal).any()))
        self.assertTrue(bool((fit | cal).all()))

    def test_rescue_harm(self):
        labels = torch.tensor([1, 0, 1, 0])
        base = torch.tensor([[3., 1., 0.], [3., 1., 0.], [3., 1., 0.], [3., 1., 0.]])
        new = base.clone(); new[0] = torch.tensor([1., 3., 0.]); new[1] = torch.tensor([1., 3., 0.])
        m = rescue_harm_from_scores(base, new, labels)
        self.assertEqual(m["rescue"], 1)
        self.assertEqual(m["harm"], 1)


if __name__ == "__main__":
    unittest.main()
