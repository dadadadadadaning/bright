import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H4 import (
    BoundedFlowAlignment,
    H4_VARIANTS,
    SARMaskedAutoencoder,
    build_h4_damageformer,
    hierarchical_probabilities,
)
from bda_benchmark.model.DamageFormer_RGBSAR_Heterogeneous import (
    build_heterogeneous_damageformer,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (
    balanced_binary_ce,
    configure_freeze_policy,
    flow_regularization,
)


class H4DamageFormerTests(unittest.TestCase):
    def test_a0_is_exact_h3_checkpoint_view(self):
        torch.manual_seed(10)
        h3 = build_heterogeneous_damageformer(
            "h3", pretrained_rgb=False
        ).eval()
        a0 = build_h4_damageformer("a0", pretrained_rgb=False).eval()
        a0.load_state_dict(h3.state_dict(), strict=True)
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = h3(rgb, sar, return_dict=True)
            actual = a0(rgb, sar, return_dict=True)
        torch.testing.assert_close(
            actual["output_loc"], expected["output_loc"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["output_dam"], expected["output_dam"], rtol=0, atol=0
        )

    def test_a1_a2_zero_transition_preserves_h3_output(self):
        torch.manual_seed(11)
        h3 = build_heterogeneous_damageformer(
            "h3", pretrained_rgb=False
        ).eval()
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = h3(rgb, sar, return_dict=True)
        for variant in ("a1", "a2"):
            model = build_h4_damageformer(
                variant, pretrained_rgb=False
            ).eval()
            missing, unexpected = model.load_state_dict(
                h3.state_dict(), strict=False
            )
            self.assertFalse(unexpected)
            self.assertTrue(missing)
            self.assertTrue(
                all(
                    "gate_predictor" in key
                    or "gated_transition" in key
                    or "flow_alignment" in key
                    for key in missing
                )
            )
            with torch.no_grad():
                actual = model(rgb, sar, return_dict=True)
            torch.testing.assert_close(
                actual["output_dam"],
                expected["output_dam"],
                rtol=1e-6,
                atol=1e-6,
            )
            for stage in range(1, 5):
                gate = actual[f"stage{stage}_sar_gate"]
                self.assertTrue(torch.isfinite(gate).all())
                self.assertGreaterEqual(float(gate.min()), 0.0)
                self.assertLessEqual(float(gate.max()), 1.0)

    def test_bounded_flow_is_identity_at_initialization(self):
        module = BoundedFlowAlignment(16, max_offset=2.0).eval()
        rgb = torch.randn(2, 16, 7, 5)
        sar = torch.randn(2, 16, 7, 5)
        with torch.no_grad():
            aligned, flow = module(rgb, sar)
        torch.testing.assert_close(flow, torch.zeros_like(flow), rtol=0, atol=0)
        torch.testing.assert_close(aligned, sar, rtol=1e-5, atol=1e-5)
        self.assertLessEqual(float(flow.abs().max()), 2.0)

    def test_flow_regularization_ignores_changed_pixels(self):
        outputs = {
            "output_dam": torch.randn(1, 4, 16, 16),
            "stage1_flow": torch.randn(1, 2, 4, 4),
        }
        changed = torch.full((1, 16, 16), 2, dtype=torch.long)
        losses = flow_regularization(outputs, changed, ignore_index=255)
        self.assertEqual(float(losses["flow_regularization_loss"]), 0.0)

    def test_zero_flow_regularization_has_finite_backward(self):
        flow = torch.zeros(1, 2, 4, 4, requires_grad=True)
        outputs = {
            "output_dam": torch.randn(1, 4, 16, 16, requires_grad=True),
            "stage1_flow": flow,
        }
        unchanged = torch.ones(1, 16, 16, dtype=torch.long)
        losses = flow_regularization(outputs, unchanged, ignore_index=255)
        loss = losses["flow_regularization_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(flow.grad)
        self.assertTrue(torch.isfinite(flow.grad).all())
        torch.testing.assert_close(flow.grad, torch.zeros_like(flow.grad))

    def test_hierarchical_probabilities_are_normalized(self):
        probabilities = hierarchical_probabilities(
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 2, 8, 8),
        )
        self.assertEqual(tuple(probabilities.shape), (2, 4, 8, 8))
        torch.testing.assert_close(
            probabilities.sum(1),
            torch.ones(2, 8, 8),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertTrue((probabilities >= 0).all())

    def test_balanced_binary_ce_is_finite_when_crop_has_no_target(self):
        logits = torch.randn(1, 2, 8, 8, requires_grad=True)
        targets = torch.full((1, 8, 8), 255, dtype=torch.long)
        loss = balanced_binary_ce(
            logits, targets, (100.0, 2.0), ignore_index=255, prior_strength=0.5
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)
        loss.backward()

    def test_sar_mae_shapes_and_masked_loss_contract(self):
        model = SARMaskedAutoencoder(depths=(1, 1, 1, 1)).eval()
        sar = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            outputs = model(sar, mask_ratio=0.5, patch_size=8)
        self.assertEqual(tuple(outputs["reconstruction"].shape), tuple(sar.shape))
        self.assertEqual(tuple(outputs["mask"].shape), tuple(sar.shape))
        self.assertTrue(torch.isfinite(outputs["reconstruction"]).all())
        self.assertGreater(float(outputs["mask"].sum()), 0.0)

    def test_freeze_policy_keeps_only_late_sar_and_damage_path_trainable(self):
        model = build_h4_damageformer("a2", pretrained_rgb=False)
        configure_freeze_policy(model)
        parameters = dict(model.named_parameters())
        self.assertFalse(parameters["rgb_encoder.features.0.0.weight"].requires_grad)
        self.assertFalse(parameters["sar_encoder.stem.0.0.weight"].requires_grad)
        self.assertTrue(
            parameters["sar_encoder.stages.2.0.depthwise.weight"].requires_grad
        )
        self.assertTrue(
            parameters["joint_stages.0.gated_transition"].requires_grad
        )

    def test_variants_do_not_add_a_teacher(self):
        for variant in H4_VARIANTS.values():
            self.assertNotIn("teacher", variant.name.lower())


if __name__ == "__main__":
    unittest.main()
