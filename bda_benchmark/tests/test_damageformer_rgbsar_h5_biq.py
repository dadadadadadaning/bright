import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H4 import build_h4_damageformer
from bda_benchmark.model.DamageFormer_RGBSAR_H5_BIQ import (
    BuildingInstanceQueryHead,
    H5_VARIANTS,
    InstanceSparseDeformableFusion,
    build_h5_biq_damageformer,
    connected_building_instances,
)


class H5BIQTests(unittest.TestCase):
    def test_connected_instances_are_deterministic_and_area_filtered(self):
        building = torch.zeros(1, 1, 12, 12)
        building[0, 0, 1:5, 1:5] = 1
        building[0, 0, 7:10, 7:10] = 1
        building[0, 0, 0, 11] = 1
        first = connected_building_instances(building, min_area=4)
        second = connected_building_instances(building, min_area=4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(int(first.max()), 2)
        self.assertEqual(int(first[0, 2, 2]), 1)
        self.assertEqual(int(first[0, 8, 8]), 2)
        self.assertEqual(int(first[0, 0, 11]), 0)

    def test_q0_is_exact_h4_a4_checkpoint_view(self):
        torch.manual_seed(21)
        a4 = build_h4_damageformer("a4", pretrained_rgb=False).eval()
        q0 = build_h5_biq_damageformer("q0", pretrained_rgb=False).eval()
        q0.load_state_dict(a4.state_dict(), strict=True)
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = a4(rgb, sar, return_dict=True)
            actual = q0(rgb, sar, return_dict=True)
        torch.testing.assert_close(
            actual["output_loc"], expected["output_loc"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["output_dam"], expected["output_dam"], rtol=0, atol=0
        )

    def test_q1_q2_zero_residuals_preserve_h4_a4_output(self):
        torch.manual_seed(22)
        a4 = build_h4_damageformer("a4", pretrained_rgb=False).eval()
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = a4(rgb, sar, return_dict=True)
        for variant in ("q1", "q2"):
            model = build_h5_biq_damageformer(
                variant,
                pretrained_rgb=False,
            ).eval()
            missing, unexpected = model.load_state_dict(
                a4.state_dict(),
                strict=False,
            )
            self.assertFalse(unexpected)
            self.assertTrue(missing)
            self.assertTrue(
                all(
                    key.startswith("instance_query_head")
                    or key.startswith("instance_sparse_fusions")
                    for key in missing
                )
            )
            with torch.no_grad():
                actual = model(rgb, sar, return_dict=True)
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

    def test_instance_query_broadcast_is_constant_within_instance(self):
        torch.manual_seed(23)
        head = BuildingInstanceQueryHead(max_instances=8).eval()
        torch.nn.init.normal_(head.severity_delta.weight)
        instance_map = torch.zeros(1, 16, 16, dtype=torch.long)
        instance_map[:, 2:8, 2:8] = 1
        instance_map[:, 9:14, 9:15] = 2
        damage = torch.randn(1, 256, 16, 16)
        base_logits = torch.randn(1, 4, 16, 16)
        rgb = [
            torch.randn(1, 96, 16, 16),
            torch.randn(1, 192, 8, 8),
            torch.randn(1, 384, 4, 4),
            torch.randn(1, 768, 2, 2),
        ]
        sar = [value.clone() for value in rgb]
        with torch.no_grad():
            delta, instance_logits, diagnostics = head(
                damage,
                base_logits,
                rgb,
                sar,
                instance_map,
            )
        self.assertEqual(tuple(delta.shape), (1, 4, 16, 16))
        self.assertEqual(int(diagnostics["instance_query_count"][0]), 2)
        for instance_id in (1, 2):
            mask = instance_map[0] == instance_id
            values = delta[0, 1:4, mask]
            torch.testing.assert_close(
                values,
                values[:, :1].expand_as(values),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            delta[0, :, instance_map[0] == 0],
            torch.zeros_like(delta[0, :, instance_map[0] == 0]),
            rtol=0,
            atol=0,
        )
        for instance_id in (1, 2):
            mask = instance_map[0] == instance_id
            predictions = instance_logits[0, :, mask].argmax(0)
            self.assertEqual(int(torch.unique(predictions).numel()), 1)

    def test_sparse_deformable_fusion_is_exact_identity_at_initialization(self):
        torch.manual_seed(24)
        module = InstanceSparseDeformableFusion(384, 128).eval()
        joint = torch.randn(1, 128, 8, 8)
        rgb = torch.randn(1, 384, 8, 8)
        sar = torch.randn(1, 384, 8, 8)
        instances = torch.zeros(1, 8, 8, dtype=torch.long)
        instances[:, 1:7, 2:6] = 1
        with torch.no_grad():
            output, diagnostics = module(joint, rgb, sar, instances)
        torch.testing.assert_close(output, joint, rtol=0, atol=0)
        self.assertLessEqual(
            float(diagnostics["instance_offset_mean"]),
            module.max_offset,
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_q2_declares_deep_sparse_fusion_only(self):
        model = build_h5_biq_damageformer("q2", pretrained_rgb=False)
        self.assertEqual(set(model.instance_sparse_fusions.keys()), {"stage3", "stage4"})
        self.assertFalse(H5_VARIANTS["q2"].extra_supervised_losses)


if __name__ == "__main__":
    unittest.main()
