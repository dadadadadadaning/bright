from __future__ import annotations

import unittest

import torch

from .common import five_features


class ContractTests(unittest.TestCase):
    def test_five_features(self):
        current = torch.tensor([1.0, -1.0])
        history = torch.tensor([0.5, 2.0])
        area = torch.tensor([9.0, 99.0])
        value = five_features(current, history, area)
        self.assertEqual(tuple(value.shape), (2, 5))
        self.assertTrue(torch.allclose(value[:, 2], torch.minimum(current, history)))
        self.assertTrue(torch.allclose(value[:, 3], (current - history).abs()))


if __name__ == "__main__":
    unittest.main()

