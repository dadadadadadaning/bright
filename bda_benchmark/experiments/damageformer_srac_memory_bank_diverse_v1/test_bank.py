"""CPU contract tests; never read BRIGHT Train, Val or Test images."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from . import build
from .bank import (bounded_push, bucket_counts, choose_bucket, choose_diverse_bucket,
                   key_redundancy, quality_key,
                   stable_tie, validate_shard)


class BankTests(unittest.TestCase):
    def test_candidate_coverage_uses_frozen_projector_not_raw_tokens(self):
        metric = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            metric.weight.copy_(torch.tensor([[2., 0.], [0., 1.]]))
        shards = [dict(tokens=torch.tensor([[1., 1.]])),
                  dict(tokens=torch.tensor([[0., 1.]]))]
        def read(_bundle, image_id):
            return None, shards[image_id], torch.tensor([True]), None, None
        row0 = ((.99, 8., 2), 0, 0)
        row1 = ((.98, 8., 1), 1, 0)
        bundle = dict(device=torch.device("cpu"), metric=metric)
        with patch.object(build, "_read_shard", side_effect=read):
            keys = build._candidate_keys(bundle, {(0, 1): [row0, row1]}, {})
        self.assertEqual(set(keys), {(0, 0), (1, 0)})
        self.assertTrue(torch.allclose(keys[(0, 0)], torch.tensor([2., 1.]) / 5 ** .5))

    def test_coverage_prefers_a_distinct_projected_key(self):
        candidates = [
            ((.99, 8., 4), 0, 0),
            ((.98, 8., 3), 1, 0),
            ((.97, 8., 2), 2, 0),
            ((.96, 8., 1), 3, 0),
        ]
        keys = {(0, 0): torch.tensor([1., 0.]),
                (1, 0): torch.tensor([.999, .04]),
                (2, 0): torch.tensor([-1., 0.]),
                (3, 0): torch.tensor([0., 1.])}
        chosen = choose_diverse_bucket(candidates, candidates, 2, keys, 1)
        self.assertEqual([(item[1], item[2]) for item, _ in chosen], [(0, 0), (2, 0)])
        self.assertEqual([fallback for _, fallback in chosen], [False, False])

    def test_exact_quota_and_only_needed_image_cap_relaxation(self):
        candidates = [((.99, 8., 3), 0, 0), ((.98, 8., 2), 0, 1),
                      ((.97, 8., 1), 1, 0)]
        keys = {(0, 0): torch.tensor([1., 0.]),
                (0, 1): torch.tensor([-1., 0.]),
                (1, 0): torch.tensor([0., 1.])}
        chosen = choose_diverse_bucket(candidates, candidates, 3, keys, 1)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(sum(fallback for _, fallback in chosen), 1)
        self.assertEqual(len({item[1:3] for item, _ in chosen}), 3)
        self.assertEqual(choose_diverse_bucket(candidates, candidates, 3, keys, 1), chosen)

    def test_diverse_bank_missing_projected_key_fails(self):
        row = ((.99, 8., 1), 0, 0)
        with self.assertRaisesRegex(ValueError, "Missing projected key"):
            choose_diverse_bucket([row], [row], 1, {}, 1)

    def test_redundancy_diagnostic_detects_duplicate_keys(self):
        duplicate = key_redundancy(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))
        spread = key_redundancy(torch.tensor([[1., 0.], [-1., 0.], [0., 1.]]))
        self.assertGreater(duplicate["mean_nearest_cosine"], spread["mean_nearest_cosine"])
        self.assertEqual(key_redundancy(torch.tensor([[1., 0.]]))["mean_nearest_cosine"], None)

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
