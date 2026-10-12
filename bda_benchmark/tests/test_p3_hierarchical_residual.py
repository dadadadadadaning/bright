# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.DamageFormer_P3_HierarchicalResidual import (
    DamageFormerP3HierarchicalResidual,
    Stage23OccurrencePyramid,
    VALID_P3_HR_VARIANTS,
    ZeroBoundedHead,
    hierarchical_residual_update,
    occurrence_log_odds,
    severity_log_odds,
)
from script.standard_ML.p3_hierarchical_residual_runtime import (
    hierarchical_residual_loss,
    hr3o_region_occurrence_loss,
    hr_validation_checks,
    severity_validation_checks,
)


class P3HierarchicalResidualTests(unittest.TestCase):
    def test_zero_residual_is_bitwise_exact_p3(self):
        base = torch.randn(2, 4, 7, 9)
        zero = torch.zeros_like(base[:, :1])
        output = hierarchical_residual_update(base, zero, zero)
        self.assertTrue(torch.equal(output, base))

    def test_update_preserves_background_and_total_building_mass(self):
        base = torch.randn(2, 4, 5, 6)
        occurrence = 0.2 * torch.randn_like(base[:, :1])
        severity = 0.2 * torch.randn_like(base[:, :1])
        output = hierarchical_residual_update(base, occurrence, severity)
        self.assertTrue(torch.equal(output[:, 0], base[:, 0]))
        self.assertTrue(
            torch.allclose(
                torch.logsumexp(output[:, 1:4], dim=1),
                torch.logsumexp(base[:, 1:4], dim=1),
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                occurrence_log_odds(output),
                occurrence_log_odds(base) + occurrence,
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                severity_log_odds(output),
                severity_log_odds(base) + severity,
                atol=2e-6,
                rtol=2e-6,
            )
        )

    def test_hr1_heads_are_zero_initialised_and_building_gated(self):
        head = ZeroBoundedHead(32, hidden_channels=16)
        feature = torch.randn(2, 32, 8, 8)
        support = torch.ones(2, 1, 16, 16)
        output = head(feature, support)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        with torch.no_grad():
            head.output.bias.fill_(1.0)
        output = head(feature, torch.zeros_like(support))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_hr2_stage23_pyramid_is_zero_initialised(self):
        module = Stage23OccurrencePyramid(channels=16)
        output, aux = module(
            torch.randn(2, 192, 12, 10),
            torch.randn(2, 192, 12, 10),
            torch.randn(2, 384, 6, 5),
            torch.randn(2, 384, 6, 5),
            torch.ones(2, 1, 24, 20),
        )
        self.assertEqual(tuple(output.shape), (2, 1, 12, 10))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertEqual(tuple(aux["hr_occurrence_tokens"].shape), (2, 16, 12, 10))

    def test_hr3o_is_a_declared_isolated_variant(self):
        self.assertIn("hr3o", VALID_P3_HR_VARIANTS)

    def test_single_head_ablation_variants_are_structurally_isolated(self):
        occurrence_only = DamageFormerP3HierarchicalResidual("hr1o")
        severity_only = DamageFormerP3HierarchicalResidual("hr1s")
        self.assertIsNotNone(occurrence_only.occurrence_head)
        self.assertIsNone(occurrence_only.severity_head)
        self.assertIsNone(severity_only.occurrence_head)
        self.assertIsNotNone(severity_only.severity_head)

    def test_hr3o_region_loss_repairs_d_and_protects_correct_i_d_x(self):
        target = torch.zeros(1, 8, 12, dtype=torch.long)
        regions = {
            "hard_i": (slice(1, 3), slice(1, 3), 1),
            "keep_i": (slice(5, 7), slice(1, 3), 1),
            "repair_d": (slice(1, 3), slice(5, 7), 2),
            "keep_d": (slice(5, 7), slice(5, 7), 2),
            "keep_x": (slice(1, 3), slice(9, 11), 3),
            "unreachable_d": (slice(5, 7), slice(9, 11), 2),
        }
        for y, x, class_id in regions.values():
            target[0, y, x] = class_id

        base = torch.full((1, 4, 8, 12), -10.0)

        def set_logits(name, intact, damaged, destroyed):
            y, x, _ = regions[name]
            base[0, 1, y, x] = intact
            base[0, 2, y, x] = damaged
            base[0, 3, y, x] = destroyed

        set_logits("hard_i", 0.2, 0.0, -10.0)
        set_logits("keep_i", 2.0, 0.0, -10.0)
        set_logits("repair_d", 1.0, 0.0, -10.0)
        set_logits("keep_d", 0.0, 2.0, -10.0)
        set_logits("keep_x", 0.0, -10.0, 2.0)
        set_logits("unreachable_d", 3.0, 0.0, -10.0)

        delta = torch.full((1, 1, 8, 12), 0.1, requires_grad=True)
        base_occurrence = occurrence_log_odds(base)
        new_occurrence = base_occurrence + delta
        severity = torch.zeros_like(delta)
        loss, diagnostics = hr3o_region_occurrence_loss(
            base,
            new_occurrence,
            base_occurrence,
            delta,
            severity,
            target,
            minimum_component_area=1,
        )
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertEqual(diagnostics["repairable_damaged_regions"], 1)
        self.assertEqual(diagnostics["unreachable_damaged_regions"], 1)
        self.assertEqual(diagnostics["selected_repair_regions"], 1)
        self.assertEqual(diagnostics["selected_hard_intact_regions"], 1)
        self.assertEqual(diagnostics["protected_intact_regions"], 1)
        self.assertEqual(diagnostics["protected_damaged_regions"], 1)
        self.assertEqual(diagnostics["protected_destroyed_regions"], 1)
        self.assertGreater(diagnostics["protection_loss"], 0.0)
        self.assertTrue(diagnostics["severity_delta_exact_zero"])
        loss.backward()
        repair_y, repair_x, _ = regions["repair_d"]
        intact_y, intact_x, _ = regions["hard_i"]
        self.assertLess(float(delta.grad[0, 0, repair_y, repair_x].mean()), 0.0)
        self.assertGreater(float(delta.grad[0, 0, intact_y, intact_x].mean()), 0.0)

    def test_hr3o_rejects_any_severity_residual(self):
        base = torch.randn(1, 4, 3, 3)
        occurrence = occurrence_log_odds(base)
        target = torch.ones(1, 3, 3, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "severity residual"):
            hr3o_region_occurrence_loss(
                base,
                occurrence,
                occurrence,
                torch.zeros(1, 1, 3, 3),
                torch.ones(1, 1, 3, 3),
                target,
                minimum_component_area=1,
            )

    def test_loss_requires_damaged_and_intact(self):
        base = torch.randn(1, 4, 3, 3)
        zero = torch.zeros(1, 1, 3, 3)
        target = torch.full((1, 3, 3), 2)
        loss, diagnostics = hierarchical_residual_loss(
            base,
            base,
            occurrence_log_odds(base),
            occurrence_log_odds(base),
            severity_log_odds(base),
            severity_log_odds(base),
            zero,
            zero,
            target,
            minimum_hard_pixels=1,
        )
        self.assertIsNone(loss)
        self.assertFalse(diagnostics["usable"])

    def test_loss_backpropagates_to_both_conditional_residuals(self):
        base = torch.randn(1, 4, 2, 4)
        occurrence_delta = torch.zeros(1, 1, 2, 4, requires_grad=True)
        severity_delta = torch.zeros(1, 1, 2, 4, requires_grad=True)
        output = hierarchical_residual_update(base, occurrence_delta, severity_delta)
        target = torch.tensor([[[1, 2, 3, 1], [2, 3, 1, 2]]])
        loss, diagnostics = hierarchical_residual_loss(
            output,
            base,
            occurrence_log_odds(output),
            occurrence_log_odds(base),
            severity_log_odds(output),
            severity_log_odds(base),
            occurrence_delta,
            severity_delta,
            target,
            minimum_hard_pixels=1,
        )
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertGreater(float(occurrence_delta.grad.abs().sum()), 0.0)
        self.assertGreater(float(severity_delta.grad.abs().sum()), 0.0)
        self.assertTrue(diagnostics["usable"])

    def test_gate_requires_declared_occurrence_improvements(self):
        base = {
            "event_miou_by_name": {str(i): 0.5 for i in range(14)},
            "valid_miou": 0.6,
            "damaged_iou": 0.4,
            "intact_iou": 0.7,
            "destroyed_iou": 0.6,
            "damaged_to_intact": 0.3,
            "damaged_gate_miss": 0.4,
            "affected_recall": 0.7,
            "event_mean_official_style_4class_miou": 0.5,
            "event_worst_official_style_4class_miou": 0.5,
        }
        candidate = dict(base)
        candidate.update(
            {
                "event_miou_by_name": {str(i): 0.501 for i in range(14)},
                "valid_miou": 0.601,
                "damaged_iou": 0.404,
                "damaged_to_intact": 0.29,
                "damaged_gate_miss": 0.394,
                "affected_recall": 0.706,
                "event_mean_official_style_4class_miou": 0.501,
                "event_worst_official_style_4class_miou": 0.501,
            }
        )
        checks, _ = hr_validation_checks(candidate, base)
        self.assertTrue(all(checks.values()))

    def test_severity_gate_does_not_require_occurrence_improvement(self):
        base = {
            "event_miou_by_name": {str(i): 0.5 for i in range(14)},
            "valid_miou": 0.6,
            "damage_macro_f1": 0.7,
            "intact_iou": 0.7,
            "damaged_iou": 0.4,
            "destroyed_iou": 0.5,
            "affected_recall": 0.7,
            "event_worst_official_style_4class_miou": 0.4,
        }
        candidate = dict(base)
        candidate.update({
            "event_miou_by_name": {str(i): 0.501 for i in range(14)},
            "valid_miou": 0.601,
            "damage_macro_f1": 0.701,
            "damaged_iou": 0.401,
            "destroyed_iou": 0.501,
            "event_worst_official_style_4class_miou": 0.401,
        })
        checks, diagnostics = severity_validation_checks(candidate, base)
        self.assertTrue(all(checks.values()))
        self.assertGreater(diagnostics["severity_mean_iou_gain_pp"], 0.0)


if __name__ == "__main__":
    unittest.main()
