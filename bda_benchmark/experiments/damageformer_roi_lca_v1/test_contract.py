import unittest

import torch

from bda_benchmark.model.DamageFormer_ROI_LCA import LCA_VARIANTS, ROILocalCorrelationResidual


class ContractTests(unittest.TestCase):
    def test_variants_isolate_one_factor(self):
        self.assertEqual(LCA_VARIANTS["a2_lca3"].window_size, 3)
        self.assertFalse(LCA_VARIANTS["a2_lca3"].confidence_gate)
        self.assertTrue(LCA_VARIANTS["a3_conf3"].confidence_gate)
        self.assertTrue(LCA_VARIANTS["a4_orient3"].orientation_key)
        self.assertEqual(LCA_VARIANTS["a5_conf5"].window_size, 5)

    def test_zero_residual_is_identity(self):
        module = ROILocalCorrelationResidual(channels=8, match_channels=4, value_channels=4)
        rgb = torch.randn(1, 8, 8, 8)
        sar = torch.randn_like(rgb)
        parent = torch.randn_like(rgb)
        building = torch.rand(1, 1, 8, 8)
        result = module(rgb, sar, parent, building)
        self.assertTrue(torch.equal(result["feature"], parent))

    def test_local_weights_produce_bounded_confidence(self):
        module = ROILocalCorrelationResidual(channels=8, match_channels=4, value_channels=4)
        result = module(
            torch.randn(1, 8, 7, 7), torch.randn(1, 8, 7, 7),
            torch.randn(1, 8, 7, 7), torch.rand(1, 1, 7, 7),
        )
        self.assertGreaterEqual(float(result["alignment_confidence"].min()), 0.0)
        self.assertLessEqual(float(result["alignment_confidence"].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
