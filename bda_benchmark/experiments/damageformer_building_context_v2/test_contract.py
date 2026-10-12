from __future__ import annotations
import unittest
import torch

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map,
    masked_local_softmax,
    soft_pool_unpool,
)
from .building_ops import soft_unpool_tokens
from .variants import get_variant


class ContractTests(unittest.TestCase):
    def test_full_is_c5_plus_building(self):
        spec = get_variant("full_joint")
        self.assertEqual(spec.base_region_variant, "c5_aligned_assoc")
        self.assertFalse(spec.freeze_cmsp)

    def test_frozen_bir_requires_c5(self):
        spec = get_variant("bir_on_frozen_c5")
        self.assertTrue(spec.freeze_cmsp)
        self.assertTrue(spec.requires_c5_checkpoint)

    def test_wrong_building_control(self):
        self.assertTrue(
            get_variant("bir_shuffle_on_frozen_c5").shuffle_building_context
        )

    def test_bf16_confidence_pool_contract(self):
        b, h, w = 1, 8, 8
        confidence = torch.rand(b, 1, h, w, dtype=torch.float32)
        logits = torch.randn(b, 9, h, w, dtype=torch.bfloat16)
        ids, valid, gh, gw = candidate_map(h, w, 4, logits.device)
        probs = masked_local_softmax(logits, valid)

        confidence_for_pool = confidence.to(probs.dtype)
        result = soft_pool_unpool(
            confidence_for_pool, probs, ids, gh * gw
        )
        self.assertTrue(torch.isfinite(result["tokens"]).all())
        self.assertEqual(result["tokens"].dtype, torch.bfloat16)

    def test_refined_soft_unpool_shape(self):
        b, k, c, h, w = 1, 4, 3, 2, 2
        tokens = torch.randn(b, k, c)
        ids = torch.zeros(9, h * w, dtype=torch.long)
        probs = torch.zeros(b, 9, h * w)
        probs[:, 0] = 1.0
        out = soft_unpool_tokens(tokens, probs, ids, h, w)
        self.assertEqual(tuple(out.shape), (b, c, h, w))


if __name__ == "__main__":
    unittest.main()
