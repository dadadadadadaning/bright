# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

BDA_ROOT = Path(__file__).resolve().parents[1]
if str(BDA_ROOT) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT))
TRAIN_SCRIPT_ROOT = BDA_ROOT / "script" / "standard_ML"
if str(TRAIN_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPT_ROOT))

from model.reliable_cross_modal.blocks import (  # noqa: E402
    BoundedSpatialReliabilityGate,
    ChangeRelationPyramid,
    ConservativeStage3Rectifier,
    LocalStructuralRelationDifference,
    PixelDamageHierarchyHead,
    ResidualSARAdapter,
    build_hierarchical_targets,
    reliable_auxiliary_losses,
)
from train_S5_S8_GPU import lovasz_grad  # noqa: E402


class S8ReliableCrossModalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_relation_is_invariant_to_positive_feature_scaling(self) -> None:
        block = LocalStructuralRelationDifference(
            in_channels=8, projection_channels=4, out_channels=8
        ).eval()
        rgb = torch.randn(2, 8, 9, 11)
        with torch.no_grad():
            _, energy, _ = block(rgb, rgb * 3.5)
        self.assertLess(float(energy.abs().max().item()), 2e-6)

    def test_relation_pyramid_shapes(self) -> None:
        pyramid = ChangeRelationPyramid(
            stage_channels=(12, 20),
            projection_channels=4,
            relation_channels=8,
            out_channels=8,
        ).eval()
        with torch.no_grad():
            relation, logits, stats = pyramid(
                torch.randn(2, 12, 12, 10),
                torch.randn(2, 12, 12, 10),
                torch.randn(2, 20, 6, 5),
                torch.randn(2, 20, 6, 5),
                target_size=(16, 14),
            )
        self.assertEqual(tuple(relation.shape), (2, 8, 16, 14))
        self.assertEqual(tuple(logits.shape), (2, 2, 16, 14))
        self.assertIn("relation_energy_s3", stats)

    def test_sar_adapter_is_exact_identity_at_initialization(self) -> None:
        adapter = ResidualSARAdapter(3, 8).eval()
        sar = torch.randn(2, 3, 13, 15)
        with torch.no_grad():
            output = adapter(sar)
        torch.testing.assert_close(output, sar, rtol=0.0, atol=0.0)

    def test_spatial_gate_is_nonnegative_and_bounded(self) -> None:
        gate = BoundedSpatialReliabilityGate(
            base_channels=16, relation_channels=8, alpha_max=0.1
        ).eval()
        base = torch.rand(2, 16, 8, 8) + 0.25
        with torch.no_grad():
            output, stats = gate(
                base,
                torch.randn(2, 8, 8, 8),
                torch.rand(2, 1, 8, 8),
                torch.randn(2, 16, 8, 8),
                torch.randn(2, 16, 8, 8),
            )
        multiplier = output / base
        self.assertGreaterEqual(float(multiplier.min().item()), 1.0)
        self.assertLessEqual(float(multiplier.max().item()), 1.1 + 1e-6)
        self.assertGreaterEqual(float(stats["reliability_alpha"].item()), 0.0)
        self.assertLessEqual(float(stats["reliability_alpha"].item()), 0.1)
        self.assertIn("change_attention_logits", stats)

    def test_actual_change_attention_receives_direct_supervision(self) -> None:
        gate = BoundedSpatialReliabilityGate(
            base_channels=16, relation_channels=8, alpha_max=0.1
        )
        base = torch.rand(1, 16, 8, 8) + 0.25
        output, stats = gate(
            base,
            torch.randn(1, 8, 8, 8),
            torch.rand(1, 1, 8, 8),
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 16, 8, 8),
        )
        losses = reliable_auxiliary_losses(
            {
                "output_dam": torch.randn(1, 4, 8, 8),
                "change_attention_logits": stats["change_attention_logits"],
            },
            torch.full((1, 8, 8), 2, dtype=torch.long),
        )
        losses["loss_aux_total"].backward()
        gradient = gate.gate[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_hierarchical_targets_mask_background_and_intact_severity(self) -> None:
        labels = torch.tensor([[[0, 1, 2, 3, 255]]])
        exist, severity = build_hierarchical_targets(labels)
        self.assertEqual(exist.tolist(), [[[255, 0, 1, 1, 255]]])
        self.assertEqual(severity.tolist(), [[[255, 255, 0, 1, 255]]])

    def test_auxiliary_losses_are_finite_with_missing_destroyed_pixels(self) -> None:
        hierarchy = PixelDamageHierarchyHead(16, 8)
        feature = torch.randn(1, 16, 7, 9)
        exist_logits, severity_logits = hierarchy(feature)
        outputs = {
            "output_dam": torch.randn(1, 4, 14, 18),
            "change_logits_feature": torch.randn(1, 2, 7, 9, requires_grad=True),
            "pixel_exist_logits_feature": exist_logits,
            "pixel_severity_logits_feature": severity_logits,
        }
        labels = torch.ones(1, 14, 18, dtype=torch.long)
        losses = reliable_auxiliary_losses(outputs, labels)
        for value in losses.values():
            self.assertTrue(bool(torch.isfinite(value).item()))
        self.assertEqual(float(losses["loss_pixel_severity"].item()), 0.0)

    def test_stage3_rectifier_is_bounded_and_finite(self) -> None:
        rectifier = ConservativeStage3Rectifier(
            channels=8, hidden_channels=4, alpha_max=0.05
        ).eval()
        rgb = torch.randn(1, 8, 6, 6)
        sar = torch.randn(1, 8, 6, 6)
        with torch.no_grad():
            rgb_out, sar_out, stats = rectifier(
                rgb,
                sar,
                torch.ones(1, 1, 12, 12),
                torch.zeros(1, 1, 6, 6),
            )
        self.assertTrue(bool(torch.isfinite(rgb_out).all().item()))
        self.assertTrue(bool(torch.isfinite(sar_out).all().item()))
        self.assertLessEqual(float(stats["rectifier_alpha"].item()), 0.05)
        # At initialization the gate is 0.5 and the learned alpha is tiny.
        self.assertLess(float((rgb_out - rgb).abs().mean().item()), 0.01)

    def test_lovasz_gradient_has_no_overlapping_inplace_update(self) -> None:
        target = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        gradient = lovasz_grad(target)
        self.assertEqual(tuple(gradient.shape), (5,))
        self.assertTrue(bool(torch.isfinite(gradient).all().item()))


if __name__ == "__main__":
    unittest.main()
