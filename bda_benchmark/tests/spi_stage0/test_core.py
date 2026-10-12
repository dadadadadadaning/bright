import unittest

import numpy as np
import torch

from bda_benchmark.script.standard_ML.audit_H9_building_oracle import BuildingInstance
from bda_benchmark.script.standard_ML.spi_stage0.core import (
    build_stage0_decision,
    building_histogram_from_map,
    building_metrics_from_histogram,
    damage_class_components,
    evaluate_geometry,
    pooled_instance_raster,
)


def _instance(mask, label, instance_id):
    return BuildingInstance(
        mask=np.asarray(mask, dtype=np.bool_),
        label=label,
        instance_id=instance_id,
    )


class Stage0ComponentTests(unittest.TestCase):
    def test_damage_class_components_split_different_classes(self):
        localization = np.ones((3, 4), dtype=np.bool_)
        severity = np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
            dtype=np.int64,
        )
        regions = damage_class_components(
            localization, severity, minimum_area=1
        )
        self.assertEqual(len(regions), 2)
        self.assertEqual(sorted(region.area for region in regions), [6, 6])

    def test_same_class_touching_buildings_remain_one_component(self):
        localization = np.ones((2, 4), dtype=np.bool_)
        severity = np.full((2, 4), 2, dtype=np.int64)
        regions = damage_class_components(
            localization, severity, minimum_area=1
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].area, 8)

    def test_pooling_excludes_background_after_geometry_is_known(self):
        mask = np.ones((2, 2), dtype=np.bool_)
        probability = torch.tensor(
            [
                [[0.90, 0.90], [0.90, 0.90]],
                [[0.02, 0.02], [0.02, 0.02]],
                [[0.07, 0.07], [0.07, 0.07]],
                [[0.01, 0.01], [0.01, 0.01]],
            ],
            dtype=torch.float32,
        )
        raster, labels, confidence, overlap = pooled_instance_raster(
            probability,
            [_instance(mask, None, 1)],
            height=2,
            width=2,
        )
        self.assertEqual(labels, [2])
        self.assertTrue(bool((raster == 2).all()))
        self.assertGreater(confidence[0], 0.5)
        self.assertEqual(overlap, 0)


class Stage0MetricTests(unittest.TestCase):
    def test_building_majority_background_is_counted_as_missed(self):
        mask = np.ones((2, 2), dtype=np.bool_)
        prediction = torch.tensor([[0, 0], [0, 2]])
        histogram = building_histogram_from_map(
            prediction, [_instance(mask, 2, 1)]
        )
        metrics = building_metrics_from_histogram(histogram)
        self.assertEqual(metrics["missed_as_background"], 1)
        self.assertEqual(metrics["per_class"]["damaged"]["recall"], 0.0)

    def test_geometry_reports_one_merge(self):
        first = np.zeros((3, 4), dtype=np.bool_)
        first[:, :2] = True
        second = np.zeros((3, 4), dtype=np.bool_)
        second[:, 2:] = True
        merged = np.ones((3, 4), dtype=np.bool_)
        result = evaluate_geometry(
            [_instance(merged, None, 1)],
            [_instance(first, 1, 10), _instance(second, 2, 11)],
            match_iou=0.5,
        )
        self.assertEqual(result["merge"], 1)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_negative"], 1)

    def test_optimized_geometry_matches_direct_mask_definition(self):
        pred_a = np.zeros((5, 6), dtype=np.bool_)
        pred_a[1:4, 1:3] = True
        pred_b = np.zeros((5, 6), dtype=np.bool_)
        pred_b[1:4, 3:5] = True
        truth_a = np.zeros((5, 6), dtype=np.bool_)
        truth_a[1:4, 1:3] = True
        truth_b = np.zeros((5, 6), dtype=np.bool_)
        truth_b[1:4, 3:5] = True
        predicted = [_instance(pred_a, None, 1), _instance(pred_b, None, 2)]
        target = [_instance(truth_a, 1, 10), _instance(truth_b, 2, 11)]
        result = evaluate_geometry(predicted, target, match_iou=0.5, device="cpu")
        direct = np.asarray(
            [
                [
                    np.logical_and(pred.mask, truth.mask).sum()
                    / max(np.logical_or(pred.mask, truth.mask).sum(), 1)
                    for truth in target
                ]
                for pred in predicted
            ]
        )
        self.assertEqual(result["true_positive"], 2)
        self.assertAlmostEqual(
            result["matched_iou_sum"],
            float(direct[0, 0] + direct[1, 1]),
        )

    def test_overlapping_predictions_use_exact_fallback(self):
        truth = np.zeros((6, 6), dtype=np.bool_)
        truth[1:5, 1:5] = True
        first = np.zeros((6, 6), dtype=np.bool_)
        first[1:5, 1:4] = True
        second = np.zeros((6, 6), dtype=np.bool_)
        second[1:5, 2:5] = True
        result = evaluate_geometry(
            [_instance(first, None, 1), _instance(second, None, 2)],
            [_instance(truth, 2, 10)],
            match_iou=0.5,
            device="cpu",
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["split"], 1)
        self.assertAlmostEqual(result["mean_matched_iou"], 0.75)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_geometry_matches_cpu(self):
        first = np.zeros((8, 8), dtype=np.bool_)
        first[1:5, 1:5] = True
        second = np.zeros((8, 8), dtype=np.bool_)
        second[3:7, 3:7] = True
        predicted = [_instance(first, None, 1), _instance(second, None, 2)]
        target = [_instance(first, 1, 10), _instance(second, 2, 11)]
        cpu = evaluate_geometry(predicted, target, match_iou=0.5, device="cpu")
        cuda = evaluate_geometry(predicted, target, match_iou=0.5, device="cuda:0")
        self.assertEqual(cpu, cuda)


class Stage0DecisionTests(unittest.TestCase):
    @staticmethod
    def _report(valid, damaged_f1, intact_iou, building_macro):
        return {
            "semantic": {
                "valid_miou": valid,
                "damaged_f1": damaged_f1,
                "intact_iou": intact_iou,
            },
            "building": {"macro_f1_present_gt": building_macro},
        }

    def test_gate_passes_only_when_all_preregistered_checks_pass(self):
        reports = {
            "e0_raw_p3": self._report(0.60, 0.50, 0.70, 0.60),
            "e4_gt_instance_pool": self._report(0.61, 0.52, 0.699, 0.62),
        }
        events = [
            {"e4_vs_e0_building_macro_f1_gain_pp": 0.1}
            for _ in range(10)
        ]
        decision = build_stage0_decision(reports, events)
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["recommendation"],
            "PROCEED_TO_T1_COMPONENT_BALANCED_JOINT_TRAINING",
        )

    def test_gate_stops_when_damaged_f1_does_not_improve(self):
        reports = {
            "e0_raw_p3": self._report(0.60, 0.50, 0.70, 0.60),
            "e4_gt_instance_pool": self._report(0.61, 0.505, 0.70, 0.62),
        }
        events = [
            {"e4_vs_e0_building_macro_f1_gain_pp": 0.1}
            for _ in range(10)
        ]
        decision = build_stage0_decision(reports, events)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["gt_pool_damaged_f1_gain"])


if __name__ == "__main__":
    unittest.main()
