# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from model.cmr.metric import CMRResidualMetric, event_aware_supcon_loss
from model.cmr.pooling import CMR_KEY_VARIANTS
from script.standard_ML.cmr_metric_utils import (
    cross_event_retrieval_eval,
    frozen_memory_view,
    load_cmr_cache,
    relation_statistics,
)
from script.standard_ML.train_CMR_M1_metric import retrieval_gate


class CMRM1MetricTests(unittest.TestCase):
    def test_step_zero_is_identity_and_adapter_receives_gradient(self):
        torch.manual_seed(3407)
        feature = torch.randn(12, 8)
        labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        events = torch.tensor([0, 0, 1, 1] * 3)
        model = CMRResidualMetric(8, hidden_dim=6)
        expected = torch.nn.functional.normalize(feature, dim=1)
        torch.testing.assert_close(model(feature), expected, atol=1e-7, rtol=0.0)
        loss = event_aware_supcon_loss(model(feature), labels, events)
        loss.backward()
        self.assertGreater(float(model.adapter[-1].weight.grad.abs().sum()), 0.0)

    def test_frozen_memory_view_uses_only_declared_indices(self):
        feature = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        events = torch.tensor([0, 1, 0, 1, 0, 1])
        memory_x, memory_y, memory_e, indices = frozen_memory_view(
            feature, labels, events, {"memory_indices": torch.tensor([1, 2, 5])}
        )
        self.assertEqual(indices.tolist(), [1, 2, 5])
        torch.testing.assert_close(memory_x, feature[indices])
        self.assertEqual(memory_y.tolist(), [0, 1, 2])
        self.assertEqual(memory_e.tolist(), [1, 0, 1])

    def test_missing_memory_indices_is_rejected(self):
        with self.assertRaises(KeyError):
            frozen_memory_view(
                torch.ones(3, 2),
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 1, 2]),
                {},
            )

    def test_vectorised_retrieval_excludes_same_event(self):
        reference_z = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        reference_y = torch.tensor([0, 1, 1, 2])
        reference_e = torch.tensor([0, 1, 0, 1])
        query_z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        query_y = torch.tensor([1, 1])
        query_e = torch.tensor([0, 1])
        result = cross_event_retrieval_eval(
            reference_z,
            reference_y,
            reference_e,
            query_z,
            query_y,
            query_e,
            k_values=(1, 2),
            ap_k=2,
            query_chunk=1,
            device="cpu",
            print_progress=False,
        )
        self.assertEqual(result["num_reference"], 4)
        self.assertEqual(result["num_queries"], 2)
        self.assertTrue(result["same_event_excluded"])
        self.assertEqual(result["cross_event_R@1"], 1.0)
        self.assertEqual(result["per_class"][1]["mAP@2"], 1.0)
        self.assertEqual(result["macro_mAP@2"], 1.0)

    def test_relation_statistics_detects_class_over_event_geometry(self):
        features = []
        labels = []
        events = []
        for cls, centre in enumerate(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))):
            for event in range(3):
                for _ in range(8):
                    features.append(centre)
                    labels.append(cls)
                    events.append(event)
        result = relation_statistics(
            torch.tensor(features),
            torch.tensor(labels),
            torch.tensor(events),
            num_pairs=3000,
            seed=7,
        )
        self.assertGreater(result["delta_cross_event"], 0.5)

    def test_gate_enforces_all_predeclared_conditions(self):
        raw = {
            "cross_event_mAP@10": 0.50,
            "macro_mAP@10": 0.48,
            "per_class": {
                0: {"R@1": 0.90},
                1: {"R@1": 0.40},
                2: {"R@1": 0.60},
            },
        }
        candidate = {
            "cross_event_mAP@10": 0.55,
            "macro_mAP@10": 0.53,
            "per_class": {
                0: {"R@1": 0.89},
                1: {"R@1": 0.45},
                2: {"R@1": 0.59},
            },
        }
        checks = retrieval_gate(
            candidate,
            raw,
            {"delta_cross_event": 0.08},
            {"delta_cross_event": -0.02},
            ap_k=10,
        )
        self.assertTrue(all(checks.values()))

    def test_cache_loader_refuses_test_and_reads_requested_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pth"
            payload = {
                "key_variants": {
                    name: torch.ones(3, 2) + index
                    for index, name in enumerate(CMR_KEY_VARIANTS)
                },
                "labels": torch.tensor([0, 1, 2]),
                "events": ["a", "b", "c"],
                "image_ids": ["i0", "i1", "i2"],
                "metadata": {
                    "protocol": "CMR_P0_frozen_pooling_audit_v1",
                    "role": "query",
                    "test_accessed": False,
                },
            }
            torch.save(payload, path)
            feature, labels, codes, event_map, loaded = load_cmr_cache(
                str(path), "mean_std", expected_role="query"
            )
            self.assertEqual(feature.shape, (3, 2))
            self.assertEqual(labels.tolist(), [0, 1, 2])
            self.assertEqual(codes.tolist(), [0, 1, 2])
            self.assertEqual(set(event_map), {"a", "b", "c"})
            self.assertEqual(loaded["metadata"], payload["metadata"])


if __name__ == "__main__":
    unittest.main()
