import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (
    H10_VARIANTS,
    ConditionalSeverityHead,
    DamageFormerRGBSARH10H3D,
    Lite3DSTAAffectedHead,
    Matched2DAffectedHead,
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (
    DamageFormerRGBSARH6SDCR,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (
    hierarchical_damage_nll,
)
from bda_benchmark.script.standard_ML.train_H10_H3D import (
    configure_h10_freeze_policy,
)


class H10H3DTests(unittest.TestCase):
    def test_variants_form_the_controlled_chain(self):
        self.assertEqual(tuple(H10_VARIANTS), ("h0", "h1", "h2", "h3", "h4"))
        self.assertEqual(H10_VARIANTS["h2"].occurrence_head, "2d")
        self.assertEqual(H10_VARIANTS["h3"].occurrence_head, "3d")
        self.assertTrue(H10_VARIANTS["h4"].conditional_severity)

    def test_h0_h1_have_exact_h6_a2_state_contract(self):
        for variant in ("h0", "h1"):
            torch.manual_seed(13)
            source = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
            torch.manual_seed(13)
            target = DamageFormerRGBSARH10H3D(variant=variant, pretrained_rgb=False)
            self.assertEqual(set(source.state_dict()), set(target.state_dict()))
            target.load_state_dict(source.state_dict(), strict=True)

    def test_factorisation_matches_building_probabilities(self):
        logits = torch.randn(3, 4, 9, 7)
        occurrence, severity = factorize_building_logits(logits)
        probability = logits.softmax(1)
        building_mass = probability[:, 1:].sum(1, keepdim=True)
        expected_occurrence = (probability[:, 2:4].sum(1, keepdim=True) / building_mass).clamp(0, 1)
        expected_severity = probability[:, 3:4] / probability[:, 2:4].sum(1, keepdim=True)
        self.assertTrue(torch.allclose(occurrence.sigmoid(), expected_occurrence, atol=1e-6))
        self.assertTrue(torch.allclose(severity.sigmoid(), expected_severity, atol=1e-6))

    def test_zero_residual_is_bitwise_safe(self):
        logits = torch.randn(2, 4, 8, 8)
        zero = logits[:, :1] * 0.0
        output = apply_hierarchical_residuals(logits, zero, zero)
        self.assertTrue(torch.equal(output, logits))

    def test_occurrence_residual_preserves_severity_boundary(self):
        logits = torch.randn(2, 4, 8, 8)
        residual = torch.randn(2, 1, 8, 8)
        output = apply_hierarchical_residuals(logits, occurrence_residual=residual)
        self.assertTrue(
            torch.allclose(
                output[:, 3] - output[:, 2],
                logits[:, 3] - logits[:, 2],
                atol=2e-6,
            )
        )
        self.assertTrue(torch.equal(output[:, 0], logits[:, 0]))
        self.assertTrue(torch.equal(output[:, 1], logits[:, 1]))

    def test_severity_residual_preserves_affected_mass(self):
        logits = torch.randn(2, 4, 8, 8)
        residual = 0.2 * torch.randn(2, 1, 8, 8)
        output = apply_hierarchical_residuals(logits, severity_residual=residual)
        before = torch.logsumexp(logits[:, 2:4], dim=1)
        after = torch.logsumexp(output[:, 2:4], dim=1)
        self.assertTrue(torch.allclose(before, after, atol=2e-6))
        self.assertTrue(
            torch.allclose(
                output[:, 3:4] - output[:, 2:3],
                logits[:, 3:4] - logits[:, 2:3] + residual,
                atol=2e-6,
            )
        )

    def test_building_prior_blocks_nonbuilding_residual(self):
        rgb = torch.randn(2, 160, 6, 6)
        sar = torch.randn_like(rgb)
        building = torch.zeros(2, 1, 24, 24)
        for head in (Matched2DAffectedHead(), Lite3DSTAAffectedHead()):
            residual, diagnostics = head(rgb, sar, building)
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
            self.assertTrue(torch.equal(diagnostics["h10_building_support"], torch.zeros_like(residual)))

    def test_2d_and_3d_controls_are_parameter_matched(self):
        two_d = sum(parameter.numel() for parameter in Matched2DAffectedHead().parameters())
        three_d = sum(parameter.numel() for parameter in Lite3DSTAAffectedHead().parameters())
        ratio = max(two_d, three_d) / min(two_d, three_d)
        self.assertLess(ratio, 1.35, (two_d, three_d, ratio))

    def test_zero_transition_is_safe_and_has_first_step_gradient(self):
        head = Lite3DSTAAffectedHead()
        transition = torch.nn.Parameter(torch.tensor(0.0))
        rgb = torch.randn(2, 160, 6, 6)
        sar = torch.randn_like(rgb)
        building = torch.ones(2, 1, 24, 24)
        candidate, _ = head(rgb, sar, building)
        base = torch.randn(2, 4, 6, 6)
        residual = torch.tanh(transition) * candidate
        output = apply_hierarchical_residuals(base, occurrence_residual=residual)
        self.assertTrue(torch.equal(output, base))
        output.square().mean().backward()
        self.assertIsNotNone(transition.grad)
        self.assertTrue(torch.isfinite(transition.grad))
        self.assertGreater(float(transition.grad.abs()), 0.0)

    def test_h3_full_forward_is_exact_a2_at_safe_initialization(self):
        model = DamageFormerRGBSARH10H3D(variant="h3", pretrained_rgb=False)
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 1, 64, 64),
                return_dict=True,
            )
        self.assertTrue(torch.equal(outputs["output_dam"], outputs["native_output_dam"]))
        self.assertEqual(tuple(outputs["h10_affected_logits"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(outputs["h10_severity_logits"].shape), (1, 1, 64, 64))

    def test_severity_head_is_building_and_affected_conditioned(self):
        head = ConditionalSeverityHead()
        rgb = torch.randn(2, 160, 6, 6)
        sar = torch.randn_like(rgb)
        zero = torch.zeros(2, 1, 24, 24)
        one = torch.ones_like(zero)
        residual, diagnostics = head(rgb, sar, zero, one)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(diagnostics["h10_severity_support"], torch.zeros_like(residual)))

    def test_hierarchical_nll_uses_only_building_damage_labels(self):
        occurrence = torch.zeros(1, 1, 2, 2, requires_grad=True)
        severity = torch.zeros(1, 1, 2, 2, requires_grad=True)
        labels = torch.tensor([[[0, 1], [2, 3]]])
        result = hierarchical_damage_nll(
            {
                "h10_affected_logits": occurrence,
                "h10_severity_logits": severity,
            },
            labels,
            255,
        )
        # Three building pixels: one term for intact and two terms for each
        # affected pixel => 5*log(2)/3.
        expected = 5.0 * torch.log(torch.tensor(2.0)) / 3.0
        self.assertAlmostEqual(float(result["extra_supervised_loss"]), float(expected), places=6)

    def test_freeze_policy_trains_only_the_declared_decision(self):
        h3 = DamageFormerRGBSARH10H3D(variant="h3", pretrained_rgb=False)
        configure_h10_freeze_policy(h3)
        trainable = [name for name, value in h3.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("affected_head", "affected_transition")) for name in trainable))

        h4 = DamageFormerRGBSARH10H3D(variant="h4", pretrained_rgb=False)
        configure_h10_freeze_policy(h4)
        trainable = [name for name, value in h4.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("severity_head", "severity_transition")) for name in trainable))


if __name__ == "__main__":
    unittest.main()
