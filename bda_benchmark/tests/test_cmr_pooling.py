# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from model.DamageFormer_S8_MICB_DSMIL import MICBDSMILHead
from model.cmr.pooling import (
    CMR_KEY_VARIANTS,
    build_cmr_feature_maps,
    extract_cmr_pooling_keys,
)
from script.standard_ML.audit_CMR_pooling import (
    binary_average_precision,
    load_cache,
    rescue_harm_audit,
)


class CMRPoolingTests(unittest.TestCase):
    def _fixture(self):
        torch.manual_seed(3407)
        head = MICBDSMILHead(
            in_channels=4,
            token_dim=8,
            topk=3,
            max_tokens=16,
            attention_temperature=0.2,
        ).eval()
        rgb = torch.randn(1, 4, 4, 4)
        sar = torch.randn(1, 4, 4, 4)
        fused = torch.randn(1, 4, 4, 4)
        mask = torch.zeros(8, 8)
        mask[2:6, 2:6] = 1
        maps = build_cmr_feature_maps(head, rgb, sar, fused)
        return head, maps, mask

    def test_every_declared_pooling_key_is_finite_and_normalised(self):
        head, maps, mask = self._fixture()
        keys = extract_cmr_pooling_keys(head, maps, mask)
        self.assertEqual(tuple(keys), CMR_KEY_VARIANTS)
        for name, key in keys.items():
            self.assertTrue(torch.isfinite(key).all(), name)
            torch.testing.assert_close(key.norm(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_mean_variant_exactly_matches_masked_fused_roi_mean(self):
        head, maps, mask = self._fixture()
        keys = extract_cmr_pooling_keys(head, maps, mask)
        weights = torch.nn.functional.interpolate(
            mask[None, None], size=(4, 4), mode="area"
        )[0, 0]
        fused = maps["fused_feature"][0]
        expected = (fused * weights[None]).sum((1, 2)) / weights.sum()
        expected = torch.nn.functional.normalize(expected, dim=0)
        torch.testing.assert_close(keys["mean"], expected)

    def test_mean_key_ignores_values_outside_component(self):
        head, maps, mask = self._fixture()
        first = extract_cmr_pooling_keys(head, maps, mask)["mean"]
        changed = {name: value.clone() for name, value in maps.items()}
        weights = torch.nn.functional.interpolate(
            mask[None, None], size=(4, 4), mode="area"
        )[0, 0]
        outside = weights == 0
        changed["fused_feature"][0, :, outside] += 1000.0
        second = extract_cmr_pooling_keys(head, changed, mask)["mean"]
        torch.testing.assert_close(first, second)

    def test_average_precision_is_one_for_perfect_ranking(self):
        score = torch.tensor([0.9, 0.8, 0.2, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        self.assertEqual(binary_average_precision(score, target), 1.0)

    def test_rescue_harm_audit_uses_memory_margin(self):
        truth = torch.tensor([1, 0, 1, 0])
        base = torch.zeros(4, dtype=torch.long)
        retrieval = torch.ones(4, dtype=torch.long)
        scores = torch.tensor([
            [0.1, 0.9, 0.0],
            [0.4, 0.5, 0.1],
            [0.2, 0.8, 0.0],
            [0.3, 0.4, 0.2],
        ])
        audit = rescue_harm_audit(truth, base, retrieval, scores)
        self.assertEqual(audit["rescues"], 2)
        self.assertEqual(audit["harms"], 2)
        self.assertEqual(audit["retrieval_margin_auroc"], 1.0)

    def test_test_derived_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            torch.save({
                "key_variants": {name: torch.ones(1, 2) for name in CMR_KEY_VARIANTS},
                "labels": torch.zeros(1, dtype=torch.long),
                "base_predictions": torch.zeros(1, dtype=torch.long),
                "base_probabilities": torch.ones(1, 3) / 3,
                "events": ["e"],
                "image_ids": ["i"],
                "metadata": {
                    "protocol": "CMR_P0_frozen_pooling_audit_v1",
                    "role": "query",
                    "test_accessed": True,
                },
            }, path)
            with self.assertRaises(RuntimeError):
                load_cache(str(path), "query")


if __name__ == "__main__":
    unittest.main()
