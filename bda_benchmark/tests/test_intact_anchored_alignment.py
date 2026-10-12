# -*- coding: utf-8 -*-
import unittest

import torch

from model.IntactAnchoredAlignment import (
    IntactAnchoredMultiScaleAlignment,
    LocalDeformableAligner,
    alignment_training_loss,
    local_correlation_volume,
    warp_with_offsets,
)
from script.standard_ML.al_alignment_runtime import (
    evidence_audit,
    occurrence_evidence_audit,
    scene_relative_residual,
)


class IntactAnchoredAlignmentTests(unittest.TestCase):
    def test_local_correlation_finds_declared_translation(self):
        generator = torch.Generator().manual_seed(17)
        rgb = torch.randn(1, 16, 9, 9, generator=generator)
        sar = torch.zeros_like(rgb)
        sar[:, :, :, 1:] = rgb[:, :, :, :-1]
        correlation, _ = local_correlation_volume(rgb, sar, radius=2)
        centre = correlation[0, :, 4, 4].argmax()
        self.assertEqual(int(centre), 13)  # dy=0, dx=+1 in a 5x5 window.

    def test_warp_uses_pixel_offsets(self):
        value = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
        offsets = torch.zeros(1, 2, 5, 5)
        offsets[:, 0] = 1.0
        warped = warp_with_offsets(value, offsets)
        self.assertAlmostEqual(float(warped[0, 0, 2, 2]), float(value[0, 0, 2, 3]))

    def test_alignment_first_offset_refiner_is_exactly_zero(self):
        model = LocalDeformableAligner(8, descriptor_channels=8, radius=1)
        rgb = torch.randn(2, 8, 7, 7)
        sar = torch.randn(2, 8, 7, 7)
        output = model(rgb, sar)
        self.assertTrue(torch.equal(output["offset_correction"], torch.zeros_like(output["offset_correction"])))

    def test_al1_loss_ignores_non_intact_pixels(self):
        model = LocalDeformableAligner(8, descriptor_channels=8, radius=1)
        rgb = torch.randn(1, 8, 7, 7)
        sar = torch.randn(1, 8, 7, 7)
        output = model(rgb, sar)
        mask = torch.zeros(1, 1, 7, 7)
        mask[:, :, 2:5, 2:5] = 1
        loss = alignment_training_loss(output, mask)
        self.assertTrue(torch.isfinite(loss["total"]))
        self.assertEqual(int(loss["intact_pixels"]), 9)

    def test_small_intact_support_survives_feature_grid_downsampling(self):
        model = LocalDeformableAligner(8, descriptor_channels=8, radius=1)
        output = model(torch.randn(1, 8, 7, 7), torch.randn(1, 8, 7, 7))
        mask = torch.zeros(1, 1, 28, 28)
        mask[:, :, 13, 13] = 1.0
        loss = alignment_training_loss(output, mask)
        self.assertTrue(torch.isfinite(loss["total"]))
        self.assertGreater(float(loss["intact_pixels"]), 0.0)

    def test_al3_freezes_alignment_but_keeps_head_trainable(self):
        model = IntactAnchoredMultiScaleAlignment(
            descriptor_channels=8, radius=1, occurrence_head=True
        )
        model.freeze_alignment()
        self.assertFalse(any(parameter.requires_grad for parameter in model.aligners.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.occurrence_head.parameters()))

    def test_evidence_gate_requires_all_predeclared_checks(self):
        events = [f"event-{index}" for index in range(10)]
        aligned = []
        unaligned = []
        for event in events:
            aligned.extend(
                [
                    {"score": 0.1, "label": 1, "event": event},
                    {"score": 0.9, "label": 2, "event": event},
                ]
            )
            unaligned.extend(
                [
                    {"score": 0.5, "label": 1, "event": event},
                    {"score": 0.5, "label": 2, "event": event},
                ]
            )
        result = evidence_audit(aligned, unaligned)
        self.assertTrue(result["passed"])
        self.assertEqual(result["damaged_vs_intact"]["direction_consistent_events"], 10)

    def test_scene_relative_residual_removes_positive_affine_scene_shift(self):
        residual = torch.tensor(
            [[[[0.1, 0.2, 0.4], [0.3, 0.7, 0.9], [0.2, 0.5, 0.8]]]],
            dtype=torch.float32,
        )
        building = torch.ones_like(residual)
        first = scene_relative_residual(
            residual, building, minimum_support=4
        )["normalised"]
        second = scene_relative_residual(
            3.0 * residual + 5.0, building, minimum_support=4
        )["normalised"]
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))

    def test_scene_relative_residual_uses_probability_not_target_labels(self):
        residual = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        probability = torch.zeros_like(residual)
        probability[:, :, :2, :2] = 1.0
        output = scene_relative_residual(
            residual, probability, minimum_support=4
        )
        self.assertEqual(int(output["support_count"].item()), 4)
        self.assertTrue(torch.isfinite(output["normalised"]).all())

    def test_occurrence_gate_uses_affected_not_damaged(self):
        candidate = []
        baseline = []
        for event_index in range(10):
            event = f"event-{event_index}"
            # Damaged alone ranks below Intact, while the combined Affected
            # population is dominated by correctly ranked Destroyed regions.
            candidate.extend(
                [{"score": 0.2, "label": 1, "event": event},
                 {"score": 0.1, "label": 2, "event": event}]
                + [{"score": 0.9, "label": 3, "event": event} for _ in range(4)]
            )
            baseline.extend(
                [{"score": 0.5, "label": label, "event": event}
                 for label in (1, 2, 3, 3, 3, 3)]
            )
        result = occurrence_evidence_audit(candidate, baseline)
        self.assertTrue(result["passed"])
        self.assertLess(result["damaged_vs_intact_diagnostic"]["global_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()
