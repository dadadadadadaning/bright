import unittest
import torch

from .association import masked_local_softmax, soft_peak_pool, soft_pool_mean_std
from .building_ops import adaptive_building_candidates, assert_same_building_candidates
from .losses import building_local_balance_loss, weak_building_bag_loss


class ContractTests(unittest.TestCase):
    def test_same_building_candidates_and_null_background(self):
        ids = torch.tensor([
            [
                [0, 1, 1, 0, 2, 2],
                [0, 1, 1, 0, 2, 2],
                [0, 1, 1, 0, 2, 2],
            ]
        ], dtype=torch.long)
        pack = adaptive_building_candidates(
            ids, target_region_area=2, min_region_area=1, max_regions_per_building=4
        )
        assert_same_building_candidates(
            ids,
            pack["candidate_ids"],
            pack["candidate_valid"],
            pack["token_building_ids"],
        )
        bg = ids.reshape(1, -1) == 0
        first = pack["candidate_ids"][:, 0]
        self.assertTrue(torch.equal(first[bg], torch.zeros_like(first[bg])))
        self.assertTrue(bool(pack["candidate_valid"][:, 0][bg].all().item()))

    def test_soft_pool_mass_conservation(self):
        ids = torch.ones((1, 2, 2), dtype=torch.long)
        pack = adaptive_building_candidates(
            ids, target_region_area=2, min_region_area=1, max_regions_per_building=4
        )
        logits = torch.zeros((1, 9, 2, 2))
        probs = masked_local_softmax(logits, pack["candidate_valid"])
        content = torch.arange(4.0).reshape(1, 1, 2, 2)
        pooled = soft_pool_mean_std(
            content, probs, pack["candidate_ids"], int(pack["token_count"].item())
        )
        self.assertAlmostEqual(float(pooled["mass"].sum()), 4.0, places=5)
        self.assertTrue(torch.isfinite(pooled["mean"]).all())
        self.assertTrue(torch.isfinite(pooled["std"]).all())

    def test_bf16_moment_pooling_dtype_safe(self):
        ids = torch.ones((1, 2, 2), dtype=torch.long)
        pack = adaptive_building_candidates(
            ids, target_region_area=2, min_region_area=1, max_regions_per_building=4
        )
        logits = torch.zeros((1, 9, 2, 2), dtype=torch.float32)
        probs = masked_local_softmax(logits, pack["candidate_valid"])
        content = torch.arange(4.0, dtype=torch.bfloat16).reshape(1, 1, 2, 2)
        pooled = soft_pool_mean_std(
            content, probs, pack["candidate_ids"], int(pack["token_count"].item())
        )
        self.assertEqual(pooled["mean"].dtype, torch.bfloat16)
        self.assertEqual(pooled["std"].dtype, torch.bfloat16)
        self.assertEqual(pooled["mass"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(pooled["mean"]).all())
        self.assertTrue(torch.isfinite(pooled["std"]).all())

    def test_soft_peak_preserves_high_evidence(self):
        building = torch.ones((1, 1, 4), dtype=torch.long)
        pack = adaptive_building_candidates(
            building, target_region_area=8, min_region_area=1, max_regions_per_building=1
        )
        logits = torch.zeros((1, 9, 1, 4))
        probs = masked_local_softmax(logits, pack["candidate_valid"])
        content = torch.tensor([[[[0.0, 0.0, 0.0, 10.0]]]])
        evidence = torch.tensor([[[[-5.0, -5.0, -5.0, 5.0]]]])
        mean = soft_pool_mean_std(
            content, probs, pack["candidate_ids"], int(pack["token_count"].item())
        )["mean"][0, 1, 0]
        peak = soft_peak_pool(
            content, evidence, probs, pack["candidate_ids"], int(pack["token_count"].item()), 0.2
        )[0, 1, 0]
        self.assertGreater(float(peak), float(mean))

    def test_building_local_balance_zero_at_target(self):
        aux = {
            "region_mass": torch.tensor([[0.0, 2.0, 2.0, 3.0, 3.0]]),
            "token_target_mass": torch.tensor([[0.0, 2.0, 2.0, 3.0, 3.0]]),
            "token_building_ids": torch.tensor([[0, 1, 1, 2, 2]]),
        }
        loss = building_local_balance_loss(aux)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_weak_bag_loss_uses_building_label_only(self):
        # Predicted building 1 covers a GT-Damaged block. Region 2 has strong Damaged logit.
        aux = {
            "region_logits": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 6.0, 0.0]]], requires_grad=True),
            "token_building_ids": torch.tensor([[0, 1, 1]]),
            "region_mass": torch.tensor([[0.0, 2.0, 2.0]]),
            "building_ids": torch.tensor([[[1, 1], [1, 1]]]),
        }
        damage = torch.tensor([[[2, 2], [2, 2]]])
        loss, stats = weak_building_bag_loss(aux, damage, aggregator="logmeanexp")
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["bag_count"], 1.0)
        loss.backward()
        self.assertIsNotNone(aux["region_logits"].grad)


if __name__ == "__main__":
    unittest.main()
