import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_Heterogeneous import (
    LocalSparseCorrespondence,
    NativeSARPyramid,
    VARIANTS,
    build_heterogeneous_damageformer,
    common_unique_regularization,
)


class HeterogeneousDamageFormerTests(unittest.TestCase):
    def test_native_sar_pyramid_is_one_channel_and_hierarchical(self):
        model = NativeSARPyramid(depths=(1, 1, 1, 1)).eval()
        outputs = model(torch.randn(2, 1, 64, 64))
        self.assertEqual(
            [tuple(value.shape) for value in outputs],
            [
                (2, 96, 16, 16),
                (2, 192, 8, 8),
                (2, 384, 4, 4),
                (2, 768, 2, 2),
            ],
        )
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 64, 64))

    def test_local_sparse_correspondence_is_bounded_and_finite(self):
        module = LocalSparseCorrespondence(32, heads=4, topk=3).eval()
        aligned, reliability = module(
            torch.randn(2, 32, 8, 6), torch.randn(2, 32, 8, 6)
        )
        self.assertEqual(tuple(aligned.shape), (2, 32, 8, 6))
        self.assertEqual(tuple(reliability.shape), (2, 1, 8, 6))
        self.assertTrue(torch.isfinite(aligned).all())
        self.assertTrue(torch.isfinite(reliability).all())
        self.assertGreaterEqual(float(reliability.min()), 0.0)
        self.assertLessEqual(float(reliability.max()), 1.0)

    def test_h3_rgb_localization_does_not_depend_on_sar(self):
        torch.manual_seed(7)
        model = build_heterogeneous_damageformer(
            "h3", pretrained_rgb=False
        ).eval()
        rgb = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            first = model(rgb, torch.randn(1, 1, 64, 64), return_dict=True)
            second = model(rgb, torch.randn(1, 1, 64, 64), return_dict=True)
        torch.testing.assert_close(first["output_loc"], second["output_loc"])
        self.assertEqual(tuple(first["output_dam"].shape), (1, 4, 64, 64))
        self.assertTrue(torch.isfinite(first["output_dam"]).all())

    def test_h0_and_h3_declare_different_sensor_contracts(self):
        rgb = torch.randn(1, 3, 64, 64)
        h0 = build_heterogeneous_damageformer("h0", pretrained_rgb=False).eval()
        h3 = build_heterogeneous_damageformer("h3", pretrained_rgb=False).eval()
        with torch.no_grad():
            self.assertEqual(
                tuple(h0(rgb, torch.randn(1, 3, 64, 64))[1].shape),
                (1, 4, 64, 64),
            )
        with self.assertRaises(ValueError):
            h0(rgb, torch.randn(1, 1, 64, 64))
        with self.assertRaises(ValueError):
            h3(rgb, torch.randn(1, 3, 64, 64))

    def test_changed_pixels_are_not_forced_into_common_alignment(self):
        outputs = {}
        for stage in range(1, 5):
            outputs[f"stage{stage}_rgb_common"] = torch.randn(1, 8, 4, 4)
            outputs[f"stage{stage}_sar_common"] = torch.randn(1, 8, 4, 4)
            outputs[f"stage{stage}_rgb_unique"] = torch.randn(1, 8, 4, 4)
            outputs[f"stage{stage}_sar_unique"] = torch.randn(1, 8, 4, 4)
        labels = torch.full((1, 16, 16), 2, dtype=torch.long)
        losses = common_unique_regularization(outputs, labels)
        self.assertEqual(float(losses["common_unchanged_loss"]), 0.0)
        self.assertTrue(torch.isfinite(losses["representation_loss"]))

    def test_no_variant_declares_a_teacher(self):
        for specification in VARIANTS.values():
            self.assertNotIn("teacher", specification.name.lower())


if __name__ == "__main__":
    unittest.main()
