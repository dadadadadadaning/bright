import unittest

import torch

from model.DamageFormer_S8_MICB_DSMIL import BuildingBagSpec
from model.DamageFormer_S8_P5_C2_Recovery import (
    C2RecoveryHeads,
    error_flow_margin_loss,
    get_c2_recovery_spec,
    pool_building_embeddings,
    supervised_contrastive_loss,
)


class S8P5C2RecoveryTests(unittest.TestCase):
    def test_recovery_variants_are_strictly_nested(self):
        r0 = get_c2_recovery_spec("r0")
        r1 = get_c2_recovery_spec("r1")
        r2 = get_c2_recovery_spec("r2")
        r3 = get_c2_recovery_spec("r3")
        self.assertFalse(r0.p3_anchor)
        self.assertTrue(r1.p3_anchor)
        self.assertFalse(r1.error_flow_margin)
        self.assertTrue(r2.error_flow_margin)
        self.assertFalse(r2.instance_supcon)
        self.assertTrue(r3.instance_supcon)

    def test_zero_gate_is_exact_p3_and_has_gradient(self):
        heads = C2RecoveryHeads(feature_channels=4, projection_dim=2)
        c2_loc = torch.randn(1, 2, 3, 3)
        c2_damage = torch.randn(1, 4, 3, 3, requires_grad=True)
        p3_loc = torch.randn(1, 2, 3, 3)
        p3_damage = torch.randn(1, 4, 3, 3)
        out_loc, out_damage = heads.anchor_outputs(
            c2_loc, c2_damage, p3_loc, p3_damage, enabled=True
        )
        self.assertTrue(torch.equal(out_loc, p3_loc))
        self.assertTrue(torch.equal(out_damage, p3_damage))
        target = torch.randn_like(out_damage)
        torch.nn.functional.mse_loss(out_damage, target).backward()
        self.assertIsNotNone(heads.damage_residual_gate.grad)
        self.assertTrue(torch.isfinite(heads.damage_residual_gate.grad))
        self.assertNotEqual(float(heads.damage_residual_gate.grad.item()), 0.0)

    def test_error_flow_margin_targets_only_declared_flows(self):
        logits = torch.zeros(1, 4, 2, 2, requires_grad=True)
        labels = torch.tensor([[[1, 2], [3, 0]]])
        result = error_flow_margin_loss(logits, labels, margin=0.3)
        self.assertAlmostEqual(float(result["loss"].item()), 0.3, places=6)
        self.assertEqual(int(result["damaged_pixel_count"].item()), 1)
        self.assertEqual(int(result["destroyed_pixel_count"].item()), 1)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_building_pool_and_supcon_are_finite(self):
        feature = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        left = torch.zeros(4, 4)
        left[:, :2] = 1
        right = 1 - left
        bags = [
            BuildingBagSpec(batch_index=0, mask=left, label=1, area=8),
            BuildingBagSpec(batch_index=0, mask=left, label=1, area=8),
            BuildingBagSpec(batch_index=0, mask=right, label=2, area=8),
            BuildingBagSpec(batch_index=0, mask=right, label=2, area=8),
        ]
        embeddings, labels = pool_building_embeddings(feature, bags)
        self.assertEqual(tuple(embeddings.shape), (4, 2))
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])
        result = supervised_contrastive_loss(embeddings, labels, temperature=0.1)
        self.assertEqual(result["anchor_count"], 4)
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_supcon_missing_positive_returns_differentiable_zero(self):
        embeddings = torch.randn(3, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        result = supervised_contrastive_loss(embeddings, labels)
        self.assertEqual(result["anchor_count"], 0)
        result["loss"].backward()
        self.assertTrue(torch.equal(embeddings.grad, torch.zeros_like(embeddings)))


if __name__ == "__main__":
    unittest.main()
