# -*- coding: utf-8 -*-
import unittest

import torch
import torch.nn.functional as F

from script.standard_ML.im4cd_matching_runtime import (
    binary_auroc,
    build_augmented_keys,
    hard_local_matching_maps,
)


class IM4CDRegionMatchingTests(unittest.TestCase):
    def test_hard_peak_recovers_and_removes_a_global_translation(self):
        generator = torch.Generator().manual_seed(7)
        rgb = torch.randn(1, 12, 11, 11, generator=generator)
        sar = torch.zeros_like(rgb)
        sar[:, :, :, 1:] = rgb[:, :, :, :-1]
        maps = hard_local_matching_maps(rgb, sar, window_size=5)
        dominant = maps["dominant_offset"][0]
        self.assertEqual((float(dominant[0]), float(dominant[1])), (1.0, 0.0))
        interior = maps["hard_residual"][0, 0, 2:-2, 2:-2]
        self.assertLess(float(interior.mean()), 0.05)

    def test_changed_cell_has_nonzero_residual_after_global_motion_removal(self):
        generator = torch.Generator().manual_seed(11)
        rgb = torch.randn(1, 8, 11, 11, generator=generator)
        sar = torch.zeros_like(rgb)
        sar[:, :, :, 1:] = rgb[:, :, :, :-1]
        # Give the centre query a distinctive match at a different offset.
        sar[:, :, 5, 4] = rgb[:, :, 5, 5]
        maps = hard_local_matching_maps(rgb, sar, window_size=5)
        self.assertGreater(float(maps["hard_residual"][0, 0, 5, 5]), 0.1)

    def test_zero_auxiliary_weight_is_exact_visual_key(self):
        visual = torch.randn(5, 7)
        auxiliary = torch.randn(5, 3)
        normalizer = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        observed = build_augmented_keys(visual, auxiliary, normalizer, 0.0)
        self.assertTrue(torch.equal(observed, F.normalize(visual.float(), dim=1)))

    def test_binary_auroc_handles_ties(self):
        labels = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(binary_auroc(labels, torch.tensor([0.0, 0.0, 1.0, 1.0])), 1.0)
        self.assertAlmostEqual(binary_auroc(labels, torch.ones(4)), 0.5)


if __name__ == "__main__":
    unittest.main()

