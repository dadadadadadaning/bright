"""Synthetic, differentiable Q loss checks; no BRIGHT files are read."""

from __future__ import annotations

import unittest

import torch

from .q_losses import soft_label_reconstruction


class QLossTests(unittest.TestCase):
    @staticmethod
    def sample():
        logits = torch.linspace(-0.6, 0.6, 36).reshape(1, 9, 4).requires_grad_()
        q = logits.softmax(dim=1)
        ids = torch.tensor([[0, 0, 1, 1] if route % 2 == 0 else [1, 1, 0, 0]
                            for route in range(9)], dtype=torch.long)
        return logits, q, ids

    def test_four_class_and_building_losses_backpropagate_to_q(self):
        gt = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
        for classes in (2, 4):
            logits, q, ids = self.sample()
            loss, detail = soft_label_reconstruction(q, ids, (2, 2), 2, gt, classes)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(loss.detach()), 0)
            self.assertGreater(int(detail["labelled_regions"]), 0)
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_ignore_255_has_zero_supervision(self):
        logits, q, ids = self.sample()
        gt = torch.full((1, 2, 2), 255, dtype=torch.long)
        loss, detail = soft_label_reconstruction(q, ids, (2, 2), 2, gt, 4)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(detail["valid_grid_mass"]), 0.0)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
