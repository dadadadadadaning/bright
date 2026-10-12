import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H4 import build_h4_damageformer
from bda_benchmark.model.DamageFormer_RGBSAR_H5_BC import (
    H5_BC_VARIANTS,
    JointRGBSARFeaturePretrainer,
    build_h5_bc_damageformer,
)


class H5BCTests(unittest.TestCase):
    def test_b0_is_exact_h4_a4_checkpoint_view(self):
        torch.manual_seed(31)
        a4 = build_h4_damageformer("a4", pretrained_rgb=False).eval()
        b0 = build_h5_bc_damageformer("b0", pretrained_rgb=False).eval()
        b0.load_state_dict(a4.state_dict(), strict=True)
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = a4(rgb, sar, return_dict=True)
            actual = b0(rgb, sar, return_dict=True)
        torch.testing.assert_close(actual["output_loc"], expected["output_loc"])
        torch.testing.assert_close(actual["output_dam"], expected["output_dam"])

    def test_b1_sparse_zero_residual_preserves_a4_and_has_no_hard_decode(self):
        torch.manual_seed(32)
        a4 = build_h4_damageformer("a4", pretrained_rgb=False).eval()
        b1 = build_h5_bc_damageformer("b1", pretrained_rgb=False).eval()
        missing, unexpected = b1.load_state_dict(a4.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(
            all(key.startswith("building_sparse_fusions") for key in missing)
        )
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = a4(rgb, sar, return_dict=True)
            actual = b1(rgb, sar, return_dict=True)
        torch.testing.assert_close(
            actual["output_loc"],
            expected["output_loc"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual["output_dam"],
            expected["output_dam"],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertNotIn("instance_query_output_dam", actual)
        self.assertEqual(
            set(b1.building_sparse_fusions.keys()),
            {"stage3", "stage4"},
        )

    def test_c1_zero_adapters_preserve_a4_and_localization_is_untouched(self):
        torch.manual_seed(33)
        a4 = build_h4_damageformer("a4", pretrained_rgb=False).eval()
        c1 = build_h5_bc_damageformer("c1", pretrained_rgb=False).eval()
        missing, unexpected = c1.load_state_dict(a4.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(
            all(key.startswith("cross_modal_adapters") for key in missing)
        )
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = a4(rgb, sar, return_dict=True)
            actual = c1(rgb, sar, return_dict=True)
        torch.testing.assert_close(
            actual["output_loc"],
            expected["output_loc"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual["output_dam"],
            expected["output_dam"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_joint_pretraining_losses_are_finite_and_group_severity(self):
        torch.manual_seed(34)
        model = build_h5_bc_damageformer("c1", pretrained_rgb=False)
        pretrainer = JointRGBSARFeaturePretrainer(model).eval()
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        damage = torch.zeros(1, 64, 64, dtype=torch.long)
        damage[:, 8:32, 8:32] = 1
        damage[:, 32:48, 8:32] = 2
        damage[:, 32:48, 32:56] = 3
        losses = pretrainer(rgb, sar, damage, mask_ratio=0.75)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertTrue(
            losses["joint_pretrain_loss"].requires_grad
        )
        self.assertFalse(
            H5_BC_VARIANTS["c1"].extra_supervised_losses
        )
        self.assertFalse(H5_BC_VARIANTS["c1"].hard_instance_decode)


if __name__ == "__main__":
    unittest.main()
