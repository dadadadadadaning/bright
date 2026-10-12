# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.cmr_global_graph_utils import cosine_preservation_loss


class CMRM1BGSPPreserveTests(unittest.TestCase):
    def test_identity_has_zero_preservation_loss(self):
        torch.manual_seed(11)
        raw = torch.randn(8, 16)
        loss = cosine_preservation_loss(raw, raw)
        self.assertLess(abs(float(loss)), 1e-6)

    def test_rotated_embedding_has_positive_loss_and_gradient(self):
        raw = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        adapted = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], requires_grad=True)
        loss = cosine_preservation_loss(adapted, raw)
        loss.backward()
        self.assertGreater(float(loss), 0.9)
        self.assertGreater(float(adapted.grad.abs().sum()), 0.0)

    def test_raw_target_is_detached(self):
        raw = torch.randn(4, 6, requires_grad=True)
        adapted = (raw.detach() + 0.2 * torch.randn(4, 6)).requires_grad_(True)
        loss = cosine_preservation_loss(adapted, raw)
        loss.backward()
        self.assertIsNone(raw.grad)
        self.assertIsNotNone(adapted.grad)


if __name__ == "__main__":
    unittest.main()
