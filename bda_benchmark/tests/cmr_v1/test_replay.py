import unittest

import numpy as np
import torch

from model.DamageFormer_S8_MICB_DSMIL import connected_components_4
from model.cmr_v1.replay import (
    bbox_iou_xyxy,
    build_component_index,
    replay_cached_component,
)


class ReplayTests(unittest.TestCase):
    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertGreater(bbox_iou_xyxy((0, 0, 10, 10), (1, 0, 11, 10)), 0.8)

    def test_exact_match(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 3:6] = True
        key = torch.tensor([1.0, 0.0])
        match = replay_cached_component(mask, (3, 2, 6, 5), 9, key, lambda _: key)
        self.assertIsNotNone(match)
        self.assertEqual(match.mode, "exact_bbox_area")

    def test_verified_fallback_uses_key(self):
        mask = np.zeros((12, 20), dtype=bool)
        mask[2:8, 2:8] = True
        # Current component differs from historical bbox by one boundary pixel.
        expected_bbox = (1, 2, 8, 8)
        expected_area = 42
        expected_key = torch.tensor([1.0, 0.0])

        def key_fn(component):
            return torch.tensor([0.9999, 0.01])

        match = replay_cached_component(
            mask,
            expected_bbox,
            expected_area,
            expected_key,
            key_fn,
            min_bbox_iou=0.80,
            min_area_ratio=0.80,
            min_key_cosine=0.99,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.mode, "verified_geometry_plus_rawmean")

    def test_precomputed_index_preserves_exact_identity(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:4, 1:5] = True
        mask[7:10, 8:11] = True
        components = connected_components_4(mask)
        index = build_component_index(components)
        key = torch.tensor([1.0, 0.0])
        match = replay_cached_component(
            mask, (1, 1, 5, 4), 12, key, lambda _: key,
            precomputed_components=components,
            exact_index=index,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.mode, "exact_bbox_area")


if __name__ == "__main__":
    unittest.main()
