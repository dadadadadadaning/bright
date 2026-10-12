import math
import unittest

from bda_benchmark.script.standard_ML.summarize_P3_eventwise_eval import (
    build_rows,
    derive_event_metrics,
)


class P3EventwiseEvaluationTests(unittest.TestCase):
    def test_occurrence_and_damaged_flows_come_from_gt_by_prediction_confusion(self):
        confusion = [
            [80, 10, 5, 5],
            [4, 86, 8, 2],
            [3, 27, 60, 10],
            [2, 8, 10, 80],
        ]
        metrics = derive_event_metrics(confusion)
        self.assertAlmostEqual(metrics["damaged_gate_miss"], 0.30)
        self.assertAlmostEqual(metrics["damaged_to_intact"], 0.27)
        self.assertAlmostEqual(metrics["damaged_to_destroyed"], 0.10)
        self.assertAlmostEqual(metrics["affected_recall"], 0.80)
        self.assertTrue(math.isfinite(metrics["loc_f1_derived"]))

    def test_optional_baseline_produces_percentage_point_deltas(self):
        official = [{
            "event": "demo-event",
            "num_images": "1",
            "num_valid_pixels": "400",
            "num_building_pixels": "300",
            "num_intact_pixels": "100",
            "num_damaged_pixels": "100",
            "num_destroyed_pixels": "100",
            "target_3class_miou": "0.60",
            "official_style_4class_miou": "0.70",
            "background_iou": "0.90",
            "intact_iou": "0.65",
            "damaged_iou": "0.50",
            "destroyed_iou": "0.65",
            "loc_head_precision": "0.91",
            "loc_head_recall": "0.89",
            "loc_head_f1": "0.90",
        }]
        baseline = {"demo-event": {
            "target_3class_miou": "0.55",
            "official_style_4class_miou": "0.68",
            "damaged_iou": "0.45",
        }}
        rows = build_rows(official, {"demo-event": [[100, 0, 0, 0]] * 4}, baseline)
        self.assertAlmostEqual(rows[0]["delta_valid_miou_pp"], 5.0)
        self.assertAlmostEqual(rows[0]["delta_miou4_pp"], 2.0)
        self.assertAlmostEqual(rows[0]["delta_damaged_iou_pp"], 5.0)
        self.assertAlmostEqual(rows[0]["loc_precision"], 91.0)
        self.assertAlmostEqual(rows[0]["loc_recall"], 89.0)
        self.assertAlmostEqual(rows[0]["loc_f1"], 90.0)


if __name__ == "__main__":
    unittest.main()
