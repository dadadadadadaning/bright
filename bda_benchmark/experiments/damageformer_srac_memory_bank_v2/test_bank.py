"""CPU contract tests; never read BRIGHT Train, Val or Test images."""
from __future__ import annotations

import unittest

import torch

from .bank import (bounded_push, bucket_counts, choose_bucket, quality_key,
                   stable_tie, validate_shard)


class BankTests(unittest.TestCase):
    def test_zero_quota_selects_nothing(self):
        item = ((1.0, 4.0, 1), 0, 0)
        self.assertEqual(choose_bucket([item], [item], 0), [])

    def test_quota_exact_with_per_image_relaxation(self):
        capped, all_rows = [], []
        for image, region, purity in ((0, 0, .99), (0, 1, .98),
                                      (0, 2, .97), (1, 0, .96)):
            item = ((purity, 4.0, image * 10 + region), image, region)
            bounded_push(all_rows, item, 8)
            if region == 0:
                bounded_push(capped, item, 8)
        chosen = choose_bucket(capped, all_rows, 3)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(sum(fallback for _item, fallback in chosen), 1)
        self.assertEqual(len({item[1:3] for item, _ in chosen}), 3)

    def test_stable_tie_and_bounded_heap(self):
        self.assertEqual(stable_tie(3407, "event_0", 3), stable_tie(3407, "event_0", 3))
        self.assertNotEqual(stable_tie(3407, "event_0", 3), stable_tie(3407, "event_0", 4))
        self.assertEqual(quality_key(.8, 100., 1, "x", 0)[1], 16.)
        heap = []
        for value in (4, 2, 8, 1, 3):
            bounded_push(heap, (value, value, 0), 2)
        self.assertEqual(sorted(row[0] for row in heap), [4, 8])

    def test_soft_labels_remain_soft_and_count_by_dominant_class(self):
        bank = dict(distribution=torch.tensor([[.15, .55, .30, 0.], [.1, .2, .1, .6],
                                               [.05, .85, .05, .05]]),
                    events=torch.tensor([0, 0, 1]))
        self.assertEqual(bucket_counts(bank, 2), [[0, 1, 0, 1], [0, 1, 0, 0]])
        self.assertAlmostEqual(float(bank["distribution"][0, 2]), .3, places=6)

    def test_shard_identity_and_quality_validated(self):
        shard = dict(name="event_0", image=0, event=1,
                     tokens=torch.ones(2, 3, dtype=torch.float16),
                     distribution=torch.tensor([[.1, .9, 0., 0.], [.4, .6, 0., 0.]]),
                     geometry=torch.zeros(2, 5, dtype=torch.float16),
                     weights=torch.ones(2), mass=torch.tensor([8., 4.]),
                     region_ids=torch.arange(2))
        eligible, dominant, purity = validate_shard(
            shard, image_id=0, image_name="event_0", event_id=1,
            expected_dim=3, min_purity=.8)
        self.assertEqual(eligible.tolist(), [True, False])
        self.assertEqual(dominant.tolist(), [1, 1])
        self.assertAlmostEqual(float(purity[0]), .9, places=6)
        shard["region_ids"] = torch.tensor([1, 0])
        with self.assertRaisesRegex(ValueError, "region IDs"):
            validate_shard(shard, image_id=0, image_name="event_0", event_id=1,
                           expected_dim=3, min_purity=.8)


if __name__ == "__main__":
    unittest.main()
