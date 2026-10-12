import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H7_CER import (
    DamagedCounterfactualRouter,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H7_CER import (
    counterfactual_router_loss,
    keep_frozen_experts_in_eval,
)


class H7CERTests(unittest.TestCase):
    def test_router_is_exact_p3_at_initialization(self):
        router = DamagedCounterfactualRouter()
        p3 = torch.randn(2, 4, 8, 8)
        a2 = torch.randn_like(p3)
        output, diagnostics = router(p3, a2, torch.randn(2, 1, 8, 8))
        self.assertTrue(torch.equal(output, p3))
        self.assertEqual(tuple(diagnostics["cer_route_gate"].shape), (2, 1, 8, 8))

    def test_only_damaged_a2_disagreements_are_eligible(self):
        router = DamagedCounterfactualRouter()
        p3 = torch.full((1, 4, 2, 2), -4.0)
        a2 = torch.full_like(p3, -4.0)
        p3[:, 1] = 4.0
        a2[:, 2] = 4.0
        _, diagnostics = router(p3, a2, torch.zeros(1, 1, 2, 2))
        self.assertTrue(torch.all(diagnostics["cer_eligible_mask"] == 1))
        a2[:, 3] = 8.0
        _, diagnostics = router(p3, a2, torch.zeros(1, 1, 2, 2))
        self.assertTrue(torch.all(diagnostics["cer_eligible_mask"] == 0))

    def test_transition_cannot_reverse_a2_evidence(self):
        router = DamagedCounterfactualRouter()
        router.transition.data.fill_(-2.0)
        p3 = torch.full((1, 4, 1, 1), -4.0)
        a2 = torch.full_like(p3, -4.0)
        p3[:, 1] = 4.0
        a2[:, 2] = 4.0
        output, diagnostics = router(p3, a2, torch.zeros(1, 1, 1, 1))
        self.assertGreaterEqual(float(diagnostics["cer_transition"]), 0.0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, p3))

    def test_counterfactual_loss_separates_helpful_and_harmful_routes(self):
        route_logit = torch.zeros(1, 1, 1, 2, requires_grad=True)
        outputs = {
            "cer_route_logit": route_logit,
            "cer_eligible_mask": torch.ones(1, 1, 1, 2),
            "cer_p3_prediction": torch.tensor([[[[1, 1]]]]),
            "cer_native_prediction": torch.tensor([[[[2, 2]]]]),
        }
        # First pixel: A2 correct. Second pixel: P3 correct.
        damage = torch.tensor([[[2, 1]]])
        values = counterfactual_router_loss(outputs, damage, 255)
        self.assertTrue(torch.isfinite(values["extra_supervised_loss"]))
        values["extra_supervised_loss"].backward()
        self.assertLess(float(route_logit.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(route_logit.grad[0, 0, 0, 1]), 0.0)

    def test_only_router_remains_in_train_mode(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = torch.nn.Dropout()
                self.safe_residual_router = torch.nn.Dropout()

        model = Toy().train()
        keep_frozen_experts_in_eval(model)
        self.assertFalse(model.expert.training)
        self.assertTrue(model.safe_residual_router.training)


if __name__ == "__main__":
    unittest.main()
