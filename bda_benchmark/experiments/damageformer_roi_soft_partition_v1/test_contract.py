from __future__ import annotations
import unittest

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.variants import get_variant
from .model import FIXED_VARIANT


class ContractTests(unittest.TestCase):
    def test_fixed_variant_name(self):
        self.assertEqual(FIXED_VARIANT, "c5_aligned_assoc")

    def test_only_designed_partition_contract(self):
        spec = get_variant(FIXED_VARIANT)
        self.assertTrue(spec.use_lca)
        self.assertEqual(spec.partition, "learnable")
        self.assertEqual(spec.sar_source, "aligned")
        self.assertTrue(spec.use_alignment_confidence)
        self.assertFalse(spec.shuffle_sar)
        self.assertFalse(spec.use_damage_hint)
        self.assertFalse(spec.use_boundary_loss)

    def test_not_color_slico(self):
        self.assertNotEqual(get_variant(FIXED_VARIANT).partition, "slico")


if __name__ == "__main__":
    unittest.main()
