import unittest

from bda_benchmark.script.standard_ML.summarize_H6_A2_DG_ablation import (
    candidate_rank,
    metric_deltas_pp,
    passes_promotion_gate,
)


class H6A2DGProtocolTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "valid_miou": 0.60,
            "damaged_iou": 0.42,
            "destroyed_iou": 0.63,
            "event_mean_official_style_4class_miou": 0.51,
            "event_worst_official_style_4class_miou": 0.40,
        }

    def test_promotion_requires_meaningful_damaged_gain(self):
        metrics = dict(self.baseline)
        metrics["damaged_iou"] += 0.0029
        delta = metric_deltas_pp(metrics, self.baseline)
        self.assertFalse(passes_promotion_gate(delta))
        metrics["damaged_iou"] += 0.0002
        delta = metric_deltas_pp(metrics, self.baseline)
        self.assertTrue(passes_promotion_gate(delta))

    def test_promotion_rejects_destroyed_regression(self):
        metrics = dict(self.baseline)
        metrics["damaged_iou"] += 0.005
        metrics["destroyed_iou"] -= 0.0021
        self.assertFalse(
            passes_promotion_gate(metric_deltas_pp(metrics, self.baseline))
        )

    def test_rank_prioritizes_damaged_then_event_worst(self):
        first = {
            "delta_pp": {
                "damaged_iou": 0.5,
                "event_worst_official_style_4class_miou": 0.0,
                "event_mean_official_style_4class_miou": 0.0,
                "valid_miou": 0.0,
                "destroyed_iou": 0.0,
            }
        }
        second = {
            "delta_pp": {
                "damaged_iou": 0.4,
                "event_worst_official_style_4class_miou": 1.0,
                "event_mean_official_style_4class_miou": 1.0,
                "valid_miou": 1.0,
                "destroyed_iou": 1.0,
            }
        }
        self.assertGreater(candidate_rank(first), candidate_rank(second))


if __name__ == "__main__":
    unittest.main()
