import unittest

import torch

from bda_benchmark.experiments.spi_risk_gate_v1.core import (
    RiskGate,
    apply_gate,
    boundary_depth_weight,
    fixed_uncertainty_gate,
    normalized_entropy,
    selector_targets,
)


class RiskGateCoreTests(unittest.TestCase):
    def test_uniform_entropy_is_one(self):
        probability = torch.full((1, 4, 2, 2), 0.25)
        self.assertTrue(torch.allclose(normalized_entropy(probability), torch.ones(1, 1, 2, 2)))

    def test_fixed_gate_never_leaks_outside_support(self):
        logits = torch.zeros(1, 4, 3, 3)
        support = torch.zeros(1, 1, 3, 3)
        support[:, :, 1, 1] = 1
        gate = fixed_uncertainty_gate(logits, support)
        self.assertEqual(int((gate > 0).sum()), 1)

    def test_boundary_weight_protects_edge(self):
        support = torch.ones(1, 1, 5, 5)
        weight = boundary_depth_weight(support, radius=2, boundary_floor=0.2)
        self.assertAlmostEqual(float(weight[0, 0, 0, 0]), 0.2)
        self.assertAlmostEqual(float(weight[0, 0, 2, 2]), 1.0)

    def test_zero_and_one_gate_recover_endpoints(self):
        base = torch.randn(1, 4, 2, 2)
        candidate = torch.randn(1, 4, 2, 2)
        self.assertTrue(torch.equal(apply_gate(base, candidate, torch.zeros(1, 1, 2, 2)), base))
        self.assertTrue(torch.allclose(apply_gate(base, candidate, torch.ones(1, 1, 2, 2)), candidate))

    def test_learned_gate_is_zero_outside_support(self):
        model = RiskGate()
        inputs = torch.randn(1, 10, 3, 3)
        support = torch.zeros(1, 1, 3, 3)
        support[:, :, 1, 1] = 1
        gate = model(inputs, support)
        self.assertTrue(torch.equal(gate[support == 0], torch.zeros_like(gate[support == 0])))

    def test_selector_targets_only_decisive_pixels(self):
        base = torch.tensor([[[[0.0]], [[3.0]], [[0.0]], [[0.0]]]])
        candidate = torch.tensor([[[[0.0]], [[0.0]], [[3.0]], [[0.0]]]])
        target = torch.tensor([[[2]]])
        selector, decisive, positive, negative = selector_targets(base, candidate, target)
        self.assertTrue(bool(decisive.item()))
        self.assertTrue(bool(positive.item()))
        self.assertFalse(bool(negative.item()))
        self.assertEqual(float(selector.item()), 1.0)


if __name__ == "__main__":
    unittest.main()

