import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H14_UWDC import (
    H14_VARIANTS,
    CommonUniqueEvidenceStage,
    DamageFormerRGBSARH14UWDC,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from bda_benchmark.script.standard_ML.h14_uwdc_runtime import (
    error_conditioned_occurrence_bce,
)


class H14UWDCTests(unittest.TestCase):
    def test_variants_form_controlled_chain(self):
        self.assertEqual(H14_VARIANTS["u0"].evidence_stages, ())
        self.assertEqual(H14_VARIANTS["u1"].evidence_stages, ("stage2",))
        self.assertFalse(H14_VARIANTS["u1"].error_conditioned_loss)
        self.assertEqual(H14_VARIANTS["u2"].evidence_stages, ("stage2",))
        self.assertTrue(H14_VARIANTS["u2"].error_conditioned_loss)
        self.assertEqual(H14_VARIANTS["u3"].evidence_stages, ("stage2", "stage3"))

    def test_common_unique_stage_is_finite(self):
        stage = CommonUniqueEvidenceStage(8, hidden_channels=8)
        values = [torch.randn(2, 8, 12, 12) for _ in range(4)]
        feature, diagnostics = stage(*values)
        self.assertEqual(feature.shape, (2, 8, 12, 12))
        self.assertTrue(torch.isfinite(feature).all())
        self.assertIn("sar_unique_abs", diagnostics)

    def test_output_head_is_zero_but_receives_gradient(self):
        stage = CommonUniqueEvidenceStage(8, hidden_channels=8)
        values = [torch.randn(1, 8, 8, 8, requires_grad=True) for _ in range(4)]
        feature, _ = stage(*values)
        head = torch.nn.Conv2d(8, 1, 1)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        output = head(feature)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        output.sum().backward()
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

    def test_error_conditioned_loss_upweights_missed_damaged(self):
        labels = torch.tensor([[[2, 3], [1, 1]]])
        base = torch.tensor([[[[-2.0, -2.0], [2.0, -2.0]]]])
        updated = base.clone().requires_grad_(True)
        outputs = {
            "h10_affected_logits": updated,
            "h10_base_affected_logits": base,
        }
        result = error_conditioned_occurrence_bce(outputs, labels, 255)
        self.assertTrue(torch.isfinite(result["extra_supervised_loss"]))
        self.assertEqual(float(result["h14_missed_damaged_pixels"]), 1.0)
        self.assertEqual(float(result["h14_missed_destroyed_pixels"]), 1.0)
        self.assertEqual(float(result["h14_intact_false_alarm_pixels"]), 1.0)
        result["extra_supervised_loss"].backward()
        self.assertTrue(torch.isfinite(updated.grad).all())

    def test_occurrence_update_preserves_severity_odds(self):
        logits = torch.randn(2, 4, 8, 8)
        residual = torch.randn(2, 1, 8, 8)
        before_occurrence, before_severity = factorize_building_logits(logits)
        updated = apply_hierarchical_residuals(
            logits, occurrence_residual=residual
        )
        after_occurrence, after_severity = factorize_building_logits(updated)
        self.assertTrue(torch.allclose(after_severity, before_severity))
        self.assertTrue(
            torch.allclose(after_occurrence, before_occurrence + residual)
        )

    def test_unique_channels_are_not_subtracted(self):
        stage = CommonUniqueEvidenceStage(8, hidden_channels=8).eval()
        common = torch.zeros(1, 8, 8, 8)
        rgb_unique = torch.randn(1, 8, 8, 8)
        sar_unique = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            first, _ = stage(common, common, rgb_unique, sar_unique)
            second, _ = stage(common, common, rgb_unique, -sar_unique)
        self.assertFalse(torch.equal(first, second))

    def test_model_declares_no_superpixel_modules(self):
        model = DamageFormerRGBSARH14UWDC("u1", pretrained_rgb=False)
        names = " ".join(name.lower() for name, _ in model.named_modules())
        self.assertNotIn("superpixel", names)
        self.assertNotIn("region_graph", names)


if __name__ == "__main__":
    unittest.main()
