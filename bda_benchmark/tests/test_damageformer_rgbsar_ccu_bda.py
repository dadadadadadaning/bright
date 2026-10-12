from __future__ import annotations

import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_CCU_BDA import (
    CCU_VARIANTS,
    VALID_CCU_VARIANTS,
    BuildingLocalCorrelation,
    BuildingLocalCorrelationStage,
    ChangeConditionedUnificationStage,
    DamageFormerRGBSARCCUBDA,
    DamageFormerRGBSARCCUControlled,
    ControlledStage4Difference,
    SpatioTemporalDifference3D,
    zero_extra_regularization,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H4 import hierarchical_probabilities


class CCUBDATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3407)

    def test_variants_form_the_declared_control_chain(self) -> None:
        self.assertEqual(
            VALID_CCU_VARIANTS,
            ("c0", "c1", "c2", "c3", "c4", "e1", "e2", "e3"),
        )
        self.assertFalse(CCU_VARIANTS["c0"].local_correlation)
        self.assertTrue(CCU_VARIANTS["c1"].global_unification)
        self.assertTrue(CCU_VARIANTS["c2"].local_correlation)
        self.assertFalse(CCU_VARIANTS["c2"].change_conditioning)
        self.assertTrue(CCU_VARIANTS["c3"].change_conditioning)
        self.assertTrue(CCU_VARIANTS["c4"].hierarchical_severity)
        self.assertFalse(CCU_VARIANTS["e1"].local_correlation)
        self.assertFalse(CCU_VARIANTS["e2"].local_correlation)
        self.assertTrue(CCU_VARIANTS["e3"].local_correlation)
        description = " ".join(value.name.lower() for value in CCU_VARIANTS.values())
        for forbidden in ("teacher", "distill", "prototype", "translation"):
            self.assertNotIn(forbidden, description)

    def test_e1_e3_have_identical_trainable_parameter_contracts(self) -> None:
        e1 = ControlledStage4Difference(16, local_matching=False)
        e3 = ControlledStage4Difference(16, local_matching=True)
        e1_contract = {
            name: tuple(parameter.shape)
            for name, parameter in e1.named_parameters()
        }
        e3_contract = {
            name: tuple(parameter.shape)
            for name, parameter in e3.named_parameters()
        }
        self.assertEqual(e1_contract, e3_contract)
        self.assertEqual(
            sum(parameter.numel() for parameter in e1.parameters()),
            sum(parameter.numel() for parameter in e3.parameters()),
        )

    def test_controlled_stage_is_exact_base_at_safe_initialization(self) -> None:
        base = torch.randn(2, 16, 8, 8)
        rgb = torch.randn(2, 16, 8, 8)
        sar = torch.randn(2, 16, 8, 8)
        building = torch.rand(2, 1, 32, 32)
        for local_matching in (False, True):
            stage = ControlledStage4Difference(
                16, local_matching=local_matching
            ).eval()
            with torch.no_grad():
                output, diagnostics = stage(base, rgb, sar, building)
            self.assertTrue(torch.equal(output, base))
            self.assertEqual(float(diagnostics["controlled_transition"]), 0.0)
        e2 = ControlledStage4Difference(
            16,
            local_matching=False,
            spatiotemporal_difference=True,
        ).eval()
        with torch.no_grad():
            output, diagnostics = e2(base, rgb, sar, building)
        self.assertTrue(torch.equal(output, base))
        self.assertEqual(float(diagnostics["controlled_transition"]), 0.0)

    def test_e2_3d_difference_has_signed_multiscale_initialization(self) -> None:
        module = SpatioTemporalDifference3D(4).eval()
        rgb = torch.zeros(1, 4, 9, 9)
        sar = torch.ones_like(rgb)
        with torch.no_grad():
            difference = module(rgb, sar)
        self.assertEqual(tuple(difference.shape), (1, 4, 9, 9))
        self.assertTrue(torch.isfinite(difference).all())
        # Away from padding, all three initialized branches compute SAR-RGB.
        interior = difference[:, :, 2:-2, 2:-2]
        self.assertTrue(torch.allclose(interior, torch.ones_like(interior)))

    def test_controlled_transition_is_safe_but_not_gradient_dead(self) -> None:
        stage = ControlledStage4Difference(16, local_matching=True)
        output, diagnostics = stage(
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 16, 8, 8),
            torch.ones(2, 1, 8, 8),
        )
        output.square().mean().backward()
        self.assertIsNotNone(stage.transition.grad)
        self.assertTrue(torch.isfinite(stage.transition.grad))
        for value in diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_soft_local_matching_recovers_known_shift_better_than_e1(self) -> None:
        stage = ControlledStage4Difference(
            8, local_matching=True, window_size=5, temperature=0.02
        ).eval()
        rgb = torch.randn(1, 8, 9, 9)
        # Shift one feature cell right.  Ignore the two-cell boundary where
        # padding necessarily removes correspondence candidates.
        sar = torch.zeros_like(rgb)
        sar[:, :, :, 1:] = rgb[:, :, :, :-1]
        with torch.no_grad():
            matched, _ = stage._soft_local_match(rgb, sar)
        local_error = (matched[:, :, 2:-2, 2:-2] - rgb[:, :, 2:-2, 2:-2]).abs().mean()
        same_position_error = (
            sar[:, :, 2:-2, 2:-2] - rgb[:, :, 2:-2, 2:-2]
        ).abs().mean()
        self.assertLess(float(local_error), float(same_position_error))

    def test_controlled_model_freezes_everything_except_stage4_adapter(self) -> None:
        model = DamageFormerRGBSARCCUControlled(
            "e3", pretrained_rgb=False
        )
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("stage4_control.") for name in trainable)
        )
        self.assertNotIn("cycle", " ".join(trainable).lower())

    def test_local_correlation_is_bounded_finite_and_spatial(self) -> None:
        matcher = BuildingLocalCorrelation(16, window_size=5, topk=9)
        aligned, diagnostics = matcher(
            torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8)
        )
        self.assertEqual(tuple(aligned.shape), (2, 16, 8, 8))
        for key in (
            "reliability_raw",
            "correlation_entropy",
            "correlation_peak",
            "correlation_margin",
            "offset_magnitude",
            "cycle_error",
            "cycle_consistency",
        ):
            value = diagnostics[key]
            self.assertEqual(tuple(value.shape), (2, 1, 8, 8))
            self.assertTrue(torch.isfinite(value).all())
        reliability = diagnostics["reliability_raw"]
        self.assertGreaterEqual(float(reliability.min()), 0.0)
        self.assertLessEqual(float(reliability.max()), 1.0)

    def test_zero_offset_geometry_has_finite_backward(self) -> None:
        # Constant interior features yield a symmetric local correlation and
        # therefore an exact zero expected offset.  This is the regression
        # case that previously produced NaN gradients after optimizer step 1.
        matcher = BuildingLocalCorrelation(8, window_size=3, topk=None)
        rgb = torch.ones(1, 8, 7, 7, requires_grad=True)
        sar = torch.ones(1, 8, 7, 7, requires_grad=True)
        _, diagnostics = matcher(rgb, sar)
        interior = diagnostics["offset_magnitude"][:, :, 2:-2, 2:-2]
        self.assertTrue(torch.isfinite(interior).all())
        interior.mean().backward()
        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(sar.grad)
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue(torch.isfinite(sar.grad).all())
        for parameter in matcher.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_building_prior_blocks_nonbuilding_sar(self) -> None:
        stage = BuildingLocalCorrelationStage(32, 16).eval()
        _, diagnostics = stage(
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 8, 8),
            building_prior=torch.zeros(1, 1, 8, 8),
        )
        self.assertTrue(torch.equal(diagnostics["reliability"], torch.zeros_like(
            diagnostics["reliability"]
        )))
        self.assertTrue(torch.equal(diagnostics["changed_gate"], torch.zeros_like(
            diagnostics["changed_gate"]
        )))

    def test_c3_is_exactly_c2_at_safe_initialization(self) -> None:
        c2 = BuildingLocalCorrelationStage(32, 16).eval()
        c3 = ChangeConditionedUnificationStage(32, 16).eval()
        missing, unexpected = c3.load_state_dict(c2.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any("conditional" in key or "change_predictor" in key for key in missing))
        rgb = torch.randn(1, 32, 8, 8)
        sar = torch.randn(1, 32, 8, 8)
        prior = torch.rand(1, 1, 8, 8)
        with torch.no_grad():
            c2_output, _ = c2(rgb, sar, prior)
            c3_output, _ = c3(rgb, sar, prior)
        self.assertTrue(torch.equal(c2_output, c3_output))

    def test_safe_transition_receives_gradient(self) -> None:
        stage = ChangeConditionedUnificationStage(32, 16)
        output, diagnostics = stage(
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 32, 8, 8),
            torch.ones(2, 1, 8, 8),
        )
        (output.square().mean()).backward()
        self.assertIsNotNone(stage.conditional_transition.grad)
        self.assertTrue(torch.isfinite(stage.conditional_transition.grad))
        for key in ("unchanged_gate", "changed_gate", "rejected_gate"):
            self.assertTrue(torch.isfinite(diagnostics[key]).all())

    def test_rgb_localization_does_not_depend_on_sar(self) -> None:
        model = DamageFormerRGBSARCCUBDA("c3", pretrained_rgb=False).eval()
        rgb = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            first = model(rgb, torch.randn(1, 1, 64, 64), return_dict=True)
            second = model(rgb, torch.randn(1, 1, 64, 64), return_dict=True)
        self.assertTrue(torch.equal(first["output_loc"], second["output_loc"]))
        with self.assertRaisesRegex(ValueError, "native SAR"):
            model(rgb, torch.randn(1, 3, 64, 64))

    def test_hierarchical_probabilities_are_normalized(self) -> None:
        probability = hierarchical_probabilities(
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 2, 8, 8),
        )
        self.assertEqual(tuple(probability.shape), (2, 4, 8, 8))
        self.assertTrue(torch.allclose(probability.sum(1), torch.ones(2, 8, 8)))

    def test_ccu_declares_no_extra_supervised_regularizer(self) -> None:
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        values = zero_extra_regularization(
            {"output_dam": logits}, torch.zeros(2, 8, 8, dtype=torch.long)
        )
        self.assertEqual(float(values["representation_loss"]), 0.0)
        values["representation_loss"].backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
