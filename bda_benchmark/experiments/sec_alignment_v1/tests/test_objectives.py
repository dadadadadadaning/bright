# -*- coding: utf-8 -*-
import random
import unittest

import torch

from bda_benchmark.experiments.sec_alignment_v1.objectives import (
    SEC_VARIANTS,
    building_prediction_consistency_loss,
    reference_validity_for_shift,
    relative_offset_equivariance_loss,
    sample_axis_shift,
    select_alignment_support,
    shift_without_wrap,
)


class SECObjectiveTests(unittest.TestCase):
    def test_s2r_is_an_alignment_only_variant(self):
        self.assertTrue(SEC_VARIANTS["s2r"].offset_delta)
        self.assertTrue(SEC_VARIANTS["s2r"].alignment_only)
        self.assertFalse(SEC_VARIANTS["s2r"].prediction_consistency)
        self.assertEqual(SEC_VARIANTS["s2r0"].support_mode, "all")
        self.assertEqual(SEC_VARIANTS["s2rp"].support_mode, "predicted")
        self.assertEqual(SEC_VARIANTS["s2rr"].support_mode, "predicted_ring")
        self.assertTrue(SEC_VARIANTS["s2rcf"].coarse_to_fine)

    def test_building_support_variants(self):
        location = torch.tensor([[[0, 1], [0, 0]]])
        clean = {
            "localization_probability_low": torch.tensor(
                [[[[0.0, 0.5], [0.25, 1.0]]]]
            )
        }
        all_support = select_alignment_support(clean, location, "all")
        gt_support = select_alignment_support(clean, location, "gt")
        predicted = select_alignment_support(clean, location, "predicted")
        dilated = select_alignment_support(
            clean, location, "predicted_ring", ring_radius_pixels=1
        )
        self.assertTrue(torch.equal(all_support, torch.ones_like(all_support)))
        self.assertEqual(float(gt_support.sum()), 1.0)
        self.assertTrue(torch.equal(predicted, clean["localization_probability_low"]))
        self.assertTrue(torch.all(dilated >= predicted))

    def test_shift_has_no_wraparound_and_reports_validity(self):
        value = torch.arange(20).reshape(1, 1, 4, 5).float()
        shifted, valid = shift_without_wrap(value, dx=2)
        self.assertTrue(torch.equal(shifted[..., :2], torch.zeros_like(shifted[..., :2])))
        self.assertTrue(torch.equal(shifted[..., 2:], value[..., :-2]))
        self.assertTrue(torch.equal(valid[..., :2], torch.zeros_like(valid[..., :2])))
        self.assertTrue(torch.equal(valid[..., 2:], torch.ones_like(valid[..., 2:])))

    def test_sampler_changes_only_one_axis(self):
        rng = random.Random(3407)
        for _ in range(20):
            dx, dy = sample_axis_shift((4, 8), rng)
            self.assertIn(abs(dx) + abs(dy), (4, 8))
            self.assertTrue((dx == 0) ^ (dy == 0))

    def test_reference_validity_masks_the_opposite_border(self):
        value = torch.zeros(1, 1, 3, 6)
        validity = reference_validity_for_shift(value, dx=2)
        self.assertTrue(torch.equal(validity[..., -2:], torch.zeros_like(validity[..., -2:])))
        self.assertTrue(torch.equal(validity[..., :-2], torch.ones_like(validity[..., :-2])))

    def test_relative_delta_loss_is_zero_at_declared_response(self):
        clean = {"output_dam": torch.zeros(1, 4, 8, 8)}
        shifted = {"output_dam": torch.zeros(1, 4, 8, 8)}
        for stage, stride, size in ((2, 8, 4), (3, 16, 2)):
            clean[f"stage{stage}_offset_x"] = torch.zeros(1, 1, size, size)
            clean[f"stage{stage}_offset_y"] = torch.zeros(1, 1, size, size)
            shifted[f"stage{stage}_offset_x"] = torch.full((1, 1, size, size), 8.0 / stride)
            shifted[f"stage{stage}_offset_y"] = torch.zeros(1, 1, size, size)
        mask = torch.ones(1, 1, 32, 32)
        valid = torch.ones_like(mask)
        result = relative_offset_equivariance_loss(
            clean, shifted, mask, valid, dx=8, dy=0
        )
        self.assertAlmostEqual(float(result["offset_delta_loss"]), 0.0, places=7)

    def test_prediction_consistency_is_zero_for_equal_logits(self):
        logits = torch.randn(2, 4, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        loss = building_prediction_consistency_loss(
            logits, logits.clone(), mask, mask
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
