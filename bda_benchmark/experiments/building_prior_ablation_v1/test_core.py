import unittest

import torch

from .model import apply_soft_building_prior


class BuildingPriorCoreTests(unittest.TestCase):
    def test_zero_transition_is_exact_identity(self):
        feature = torch.randn(2, 8, 7, 9)
        probability = torch.rand(2, 1, 7, 9)
        output, weight = apply_soft_building_prior(
            feature, probability, torch.zeros(()), roi_floor=0.10
        )
        self.assertTrue(torch.equal(output, feature))
        self.assertTrue(torch.equal(weight, torch.ones_like(weight)))

    def test_probability_is_detached(self):
        feature = torch.randn(1, 4, 3, 3, requires_grad=True)
        probability = torch.rand(1, 1, 3, 3, requires_grad=True)
        transition = torch.tensor(0.3, requires_grad=True)
        output, _ = apply_soft_building_prior(
            feature, probability, transition, roi_floor=0.10
        )
        output.sum().backward()
        self.assertIsNone(probability.grad)
        self.assertIsNotNone(feature.grad)
        self.assertIsNotNone(transition.grad)


if __name__ == "__main__":
    unittest.main()
