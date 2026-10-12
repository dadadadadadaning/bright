from __future__ import annotations

import unittest

import torch

from .losses import _shared_probability
from .variants import VARIANTS


class ContractTests(unittest.TestCase):
    def test_main_and_controls_exist(self):
        self.assertEqual(
            set(VARIANTS),
            {
                "p2_rgb_structure",
                "p5_anchored_change",
                "p6_shuffled_sar",
                "p7_no_confidence",
                "p8_no_damage_boundary",
            },
        )

    def test_main_is_structure_plus_change(self):
        spec = VARIANTS["p5_anchored_change"]
        self.assertTrue(spec.use_structure_logits)
        self.assertTrue(spec.use_change_refinement)
        self.assertTrue(spec.use_alignment_confidence)
        self.assertTrue(spec.use_damage_boundary)

    def test_p2_cannot_change_geometry(self):
        self.assertFalse(VARIANTS["p2_rgb_structure"].use_change_refinement)

    def test_shared_assignment_probability(self):
        p = torch.tensor([[[1.0], [0.0]]])
        ids_same = torch.tensor([[[4], [5]]])
        ids_other = torch.tensor([[[7], [8]]])
        self.assertAlmostEqual(float(_shared_probability(p, ids_same, p, ids_same)), 1.0 - 1e-6, places=5)
        self.assertAlmostEqual(float(_shared_probability(p, ids_same, p, ids_other)), 1e-6, places=7)

    def test_test_split_is_never_a_variant(self):
        self.assertFalse(any("test" in name.lower() for name in VARIANTS))


if __name__ == "__main__":
    unittest.main()
