import unittest

import torch
from torch import nn

from bda_benchmark.model.DamageFormer_RGBSAR_OA_HMD import (
    OA_HMD_VARIANTS,
    DamageFormerRGBSAROAHMD,
    Stage3ExplicitDifferenceHead,
)
from bda_benchmark.script.standard_ML.train_OA_HMD import (
    configure_freeze_policy,
)


class OAHMDTests(unittest.TestCase):
    def test_variants_form_minimal_control_chain(self):
        self.assertEqual(tuple(OA_HMD_VARIANTS), ("c0", "c1"))
        self.assertFalse(OA_HMD_VARIANTS["c0"].stage3_difference)
        self.assertTrue(OA_HMD_VARIANTS["c1"].stage3_difference)

    def test_stage3_projections_are_not_shared(self):
        head = Stage3ExplicitDifferenceHead()
        rgb_ids = {id(parameter) for parameter in head.rgb_projection.parameters()}
        sar_ids = {id(parameter) for parameter in head.sar_projection.parameters()}
        self.assertTrue(rgb_ids)
        self.assertTrue(sar_ids)
        self.assertTrue(rgb_ids.isdisjoint(sar_ids))

    def test_difference_head_is_blocked_outside_buildings(self):
        head = Stage3ExplicitDifferenceHead()
        rgb = torch.randn(2, 384, 8, 8)
        sar = torch.randn_like(rgb)
        building = torch.zeros(2, 1, 32, 32)
        candidate, diagnostics = head(rgb, sar, building)
        self.assertTrue(torch.equal(candidate, torch.zeros_like(candidate)))
        self.assertTrue(
            torch.equal(
                diagnostics["oa_building_support"],
                torch.zeros_like(candidate),
            )
        )

    def test_difference_head_uses_declared_four_relations(self):
        head = Stage3ExplicitDifferenceHead(hidden_channels=16)
        first = head.mixer[0][0]
        self.assertIsInstance(first, nn.Conv2d)
        self.assertEqual(first.in_channels, 4 * 16)

    def test_candidate_is_finite_and_bounded(self):
        head = Stage3ExplicitDifferenceHead(candidate_bound=3.0)
        rgb = 100.0 * torch.randn(2, 384, 8, 8)
        sar = 100.0 * torch.randn_like(rgb)
        candidate, _ = head(rgb, sar, torch.ones(2, 1, 32, 32))
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertLessEqual(float(candidate.abs().max()), 3.0 + 1e-6)

    def test_c1_declares_no_forbidden_alignment_or_3d_module(self):
        model = DamageFormerRGBSAROAHMD(variant="c1", pretrained_rgb=False)
        head = model.stage3_difference_head
        self.assertFalse(any(isinstance(module, nn.Conv3d) for module in head.modules()))
        names = " ".join(name.lower() for name, _ in head.named_modules())
        self.assertNotIn("router", names)
        self.assertNotIn("flow", names)
        self.assertNotIn("fca", names)

    def test_freeze_policy_trains_only_stage3_branch(self):
        model = DamageFormerRGBSAROAHMD(variant="c1", pretrained_rgb=False)
        configure_freeze_policy(model)
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(("stage3_difference_head", "affected_transition"))
                for name in trainable
            )
        )

    def test_zero_transition_has_first_step_gradient(self):
        head = Stage3ExplicitDifferenceHead()
        transition = nn.Parameter(torch.tensor(0.0))
        rgb = torch.randn(2, 384, 8, 8)
        sar = torch.randn_like(rgb)
        candidate, _ = head(rgb, sar, torch.ones(2, 1, 32, 32))
        residual = torch.tanh(transition) * candidate
        base = torch.randn(2, 1, 8, 8)
        output = base + residual
        self.assertTrue(torch.equal(output, base))
        output.square().mean().backward()
        self.assertIsNotNone(transition.grad)
        self.assertTrue(torch.isfinite(transition.grad))
        self.assertGreater(float(transition.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
