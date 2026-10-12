from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.DamageFormer_S7_SIRM_M2B import (
    InstanceQuery,
    RetrievalConfig,
    StaticBuildingInstanceMemory,
    balance_instance_records,
    connected_components_4,
    extract_instance_queries,
    records_to_cache_payload,
    refine_damage_logits,
)


class S7SIRMTests(unittest.TestCase):
    def test_connected_components_and_pooling(self):
        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask[0, 1:3, 1:3] = True
        mask[0, 5:7, 5:8] = True
        components = connected_components_4(mask[0])
        self.assertEqual(sorted(int(item.sum()) for item in components), [4, 6])

        feature = torch.zeros(1, 3, 4, 4)
        feature[:, 0] = 1.0
        queries = extract_instance_queries(feature, mask, min_area=2)
        self.assertEqual(len(queries), 2)
        self.assertEqual(sorted(item.area for item in queries), [4, 6])
        for item in queries:
            self.assertTrue(torch.allclose(item.feature.norm(), torch.tensor(1.0), atol=1e-6))

    def test_retrieval_and_region_only_residual(self):
        keys = torch.eye(3)
        labels = torch.tensor([0, 1, 2])
        memory = StaticBuildingInstanceMemory(keys, labels)
        config = RetrievalConfig(
            topk=1,
            alpha=1.0,
            beta=5.5,
            similarity_threshold=0.0,
            purity_threshold=0.0,
            margin_threshold=0.0,
            laplace=0.01,
        )
        flat_indices = torch.tensor([5, 6], dtype=torch.long)
        instance = InstanceQuery(
            batch_index=0,
            flat_indices=flat_indices,
            feature=torch.tensor([0.0, 1.0, 0.0]),
            area=2,
        )
        logits = torch.zeros(1, 4, 4, 4)
        logits[:, 1] = 0.01  # low-margin Intact base prediction
        refined, stats = refine_damage_logits(logits, [instance], memory, config)
        self.assertEqual(int(stats["cache_prediction"][0].item()), 1)
        self.assertGreater(float(stats["gate"][0].item()), 0.0)
        self.assertTrue(bool(stats["applied_instance"][0].item()))
        # Background is never touched.
        self.assertTrue(torch.equal(refined[:, 0], logits[:, 0]))
        # Pixels outside the component are unchanged.
        outside = torch.ones(16, dtype=torch.bool)
        outside[flat_indices] = False
        self.assertTrue(
            torch.equal(
                refined[0, 1:4].reshape(3, -1)[:, outside],
                logits[0, 1:4].reshape(3, -1)[:, outside],
            )
        )
        # Damaged cache evidence increases only within the component.
        self.assertTrue((refined[0, 2].reshape(-1)[flat_indices] > 0).all())

    def test_directional_gate_preserves_confident_and_non_intact_pixels(self):
        keys = torch.eye(3)
        labels = torch.tensor([0, 1, 2])
        memory = StaticBuildingInstanceMemory(keys, labels)
        config = RetrievalConfig(
            topk=1,
            alpha=1.0,
            beta=5.5,
            similarity_threshold=0.0,
            purity_threshold=0.0,
            margin_threshold=0.0,
            laplace=0.01,
            base_margin_threshold=0.20,
            directional_intact_only=True,
        )
        indices = torch.tensor([0, 1, 2], dtype=torch.long)
        damaged_query = InstanceQuery(
            batch_index=0,
            flat_indices=indices,
            feature=torch.tensor([0.0, 1.0, 0.0]),
            area=3,
        )
        logits = torch.zeros(1, 4, 1, 3)
        # Pixel 0: highly confident Intact -> uncertainty gate must block it.
        logits[0, 1, 0, 0] = 5.0
        # Pixel 1: uncertain Intact -> eligible for a Damaged boost.
        logits[0, 1, 0, 1] = 0.10
        # Pixel 2: already Damaged -> directional gate must preserve it.
        logits[0, 2, 0, 2] = 0.20

        refined, stats = refine_damage_logits(logits, [damaged_query], memory, config)
        self.assertTrue(torch.equal(refined[0, :, 0, 0], logits[0, :, 0, 0]))
        self.assertGreater(float(refined[0, 2, 0, 1]), float(logits[0, 2, 0, 1]))
        self.assertTrue(torch.equal(refined[0, :, 0, 2], logits[0, :, 0, 2]))
        self.assertEqual(int(stats["modified_pixel_count"][0].item()), 1)

    def test_intact_cache_vote_is_never_applied(self):
        keys = torch.eye(3)
        labels = torch.tensor([0, 1, 2])
        memory = StaticBuildingInstanceMemory(keys, labels)
        config = RetrievalConfig(
            topk=1,
            alpha=1.0,
            similarity_threshold=0.0,
            purity_threshold=0.0,
            margin_threshold=0.0,
            directional_intact_only=True,
        )
        instance = InstanceQuery(
            batch_index=0,
            flat_indices=torch.tensor([0], dtype=torch.long),
            feature=torch.tensor([1.0, 0.0, 0.0]),
            area=1,
        )
        logits = torch.zeros(1, 4, 1, 1)
        logits[0, 1] = 0.01
        refined, stats = refine_damage_logits(logits, [instance], memory, config)
        self.assertTrue(torch.equal(refined, logits))
        self.assertFalse(bool(stats["applied_instance"][0].item()))

    def test_balance_is_event_and_class_capped(self):
        records = []
        for label in range(3):
            for event in ("event_a", "event_b"):
                for index in range(10):
                    records.append(
                        {
                            "key": torch.tensor([1.0, float(label), float(index)]),
                            "label": label,
                            "event": event,
                            "area": 100,
                            "main_confidence": 1.0 - index / 100.0,
                            "hard_to_intact": index < 3 and label > 0,
                            "sample_id": f"{event}_{index}",
                            "component_index": index,
                            "purity": 1.0,
                        }
                    )
        selected = balance_instance_records(
            records,
            max_per_event_class=4,
            max_per_class=6,
            hard_fraction=0.25,
        )
        for label in range(3):
            class_records = [record for record in selected if record["label"] == label]
            self.assertEqual(len(class_records), 6)
            for event in ("event_a", "event_b"):
                self.assertLessEqual(
                    sum(record["event"] == event for record in class_records),
                    4,
                )
        payload = records_to_cache_payload(selected, metadata={"unit_test": True})
        self.assertEqual(payload["keys"].shape[0], 18)
        self.assertEqual(payload["keys"].dtype, torch.float16)
        self.assertEqual(payload["metadata"]["format"], "S7_SIRM_v1")


if __name__ == "__main__":
    unittest.main()
