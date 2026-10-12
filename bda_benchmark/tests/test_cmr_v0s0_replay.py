# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from script.standard_ML.cmr_v0_replay_runtime import bbox_iou, guarded_identity, replay_decision


class CMRV0S0ReplayTests(unittest.TestCase):
    def test_bbox_iou_is_exact_for_identical_boxes(self):
        self.assertEqual(bbox_iou([1, 2, 5, 8], [1, 2, 5, 8]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_guarded_identity_requires_geometry_feature_and_probability(self):
        row = {
            "bbox_iou": 0.995,
            "relative_area_error": 0.005,
            "mean_key_cosine": 0.9995,
            "probability_max_abs_delta": 1e-6,
        }
        self.assertTrue(guarded_identity(row))
        row["mean_key_cosine"] = 0.98
        self.assertFalse(guarded_identity(row))

    def test_near_identity_never_admits_old_cache_for_training(self):
        checks, decision = replay_decision(exact_rate=0.50, guarded_identity_rate=1.0)
        self.assertFalse(checks["exact_component_rate_at_least_declared"])
        self.assertEqual(decision, "REBUILD_P0_M3_V0_IN_ONE_RUNTIME_DO_NOT_FUZZY_MATCH_OLD_CACHE")


if __name__ == "__main__":
    unittest.main()

