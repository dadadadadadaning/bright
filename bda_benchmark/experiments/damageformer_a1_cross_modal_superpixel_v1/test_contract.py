from __future__ import annotations

import unittest

import torch

from .association import candidate_map, masked_local_softmax, soft_pool_unpool
from .losses import region_losses
from .variants import VARIANTS, get_variant


class ContractTests(unittest.TestCase):
    def test_variants_do_not_use_memory(self):
        for spec in VARIANTS.values():
            self.assertFalse(hasattr(spec, "memory"))

    def test_damage_hint_is_an_explicit_risk_control(self):
        self.assertFalse(get_variant("c5_aligned_assoc").use_damage_hint)
        self.assertTrue(get_variant("c8_damage_hint").use_damage_hint)

    def test_local_probabilities_sum_to_one(self):
        logits = torch.randn(2, 9, 7, 9)
        _, valid, _, _ = candidate_map(7, 9, 4, logits.device)
        probs = masked_local_softmax(logits, valid)
        self.assertTrue(torch.allclose(probs.sum(1), torch.ones(2, 63), atol=1e-6))

    def test_constant_feature_survives_pool_and_unpool(self):
        content = torch.ones(2, 5, 8, 10)
        logits = torch.zeros(2, 9, 8, 10)
        ids, valid, gh, gw = candidate_map(8, 10, 4, logits.device)
        probs = masked_local_softmax(logits, valid)
        result = soft_pool_unpool(content, probs, ids, gh * gw)
        self.assertTrue(torch.allclose(result["reconstruction"], content, atol=1e-5))

    def test_region_building_loss_accepts_low_precision_features(self):
        content = torch.randn(1, 6, 8, 8, dtype=torch.bfloat16)
        logits = torch.randn(1, 9, 8, 8, dtype=torch.bfloat16)
        ids, valid, gh, gw = candidate_map(8, 8, 4, logits.device)
        probabilities = masked_local_softmax(logits, valid)
        pooled = soft_pool_unpool(content, probabilities, ids, gh * gw)
        aux = {
            "feature_roi": content,
            "region_content": content,
            "region_relation": content,
            "region_confidence": torch.ones(1, 1, 8, 8, dtype=torch.bfloat16),
            "association_probabilities": probabilities,
            "candidate_ids": ids,
            "token_count": torch.tensor(float(gh * gw)),
            "region_mass": pooled["mass"],
        }
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        losses = region_losses(aux, target, use_boundary_loss=False)
        self.assertTrue(torch.isfinite(losses["building"]))


if __name__ == "__main__":
    unittest.main()
