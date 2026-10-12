import unittest

import torch
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.spi_t1.core import (
    SOURCE_VARIANT,
    VARIANTS,
    build_t1_decision,
    class_component_id_map,
    component_balanced_nll,
)


def report(valid_miou, damaged_f1, intact_iou, building_macro):
    return {
        "semantic": {
            "valid_miou": valid_miou,
            "damaged_f1": damaged_f1,
            "intact_iou": intact_iou,
        },
        "building": {"macro_f1_present_gt": building_macro},
    }


class ComponentBalancedLossTests(unittest.TestCase):
    def test_different_touching_classes_are_separate_components(self):
        labels = torch.tensor([[[1, 2], [1, 2]]])
        ids, count = class_component_id_map(labels, minimum_area=1)
        self.assertEqual(count, 2)
        self.assertNotEqual(int(ids[0, 0, 0]), int(ids[0, 0, 1]))

    def test_same_class_touching_pixels_remain_one_component(self):
        labels = torch.tensor([[[1, 1], [1, 1]]])
        _, count = class_component_id_map(labels, minimum_area=1)
        self.assertEqual(count, 1)

    def test_each_component_gets_equal_weight(self):
        labels = torch.tensor([[[1, 1, 1, 0, 2]]])
        logits = torch.zeros(1, 4, 1, 5)
        logits[:, 1, :, :3] = 5.0
        logits[:, 0, :, 3] = 5.0
        logits[:, 1, :, 4] = 5.0  # deliberately wrong for the small class-2 region
        value, stats = component_balanced_nll(logits, labels, minimum_area=1)
        pixel_nll = F.cross_entropy(logits, labels, reduction="none")
        expected = 0.5 * (pixel_nll[0, 0, :3].mean() + pixel_nll[0, 0, 4])
        self.assertTrue(torch.allclose(value, expected))
        self.assertEqual(stats["component_count"], 2)

    def test_no_foreground_component_returns_differentiable_zero(self):
        logits = torch.randn(1, 4, 2, 2, requires_grad=True)
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        value, stats = component_balanced_nll(logits, labels, minimum_area=1)
        value.backward()
        self.assertEqual(float(value.item()), 0.0)
        self.assertEqual(stats["component_count"], 0)
        self.assertIsNotNone(logits.grad)


class T1GateTests(unittest.TestCase):
    def test_gate_passes_only_when_all_checks_pass(self):
        reports = {
            SOURCE_VARIANT: report(0.50, 0.40, 0.70, 0.60),
            VARIANTS[0]: report(0.50, 0.40, 0.70, 0.60),
            VARIANTS[1]: report(0.51, 0.42, 0.698, 0.62),
        }
        events = [
            {"t1_vs_t0_building_macro_f1_gain_pp": 0.0}
            for _ in range(10)
        ]
        decision = build_t1_decision(reports, events)
        self.assertTrue(decision["passed"])

    def test_damaged_f1_failure_stops_progression(self):
        reports = {
            SOURCE_VARIANT: report(0.50, 0.40, 0.70, 0.60),
            VARIANTS[0]: report(0.50, 0.40, 0.70, 0.60),
            VARIANTS[1]: report(0.51, 0.405, 0.70, 0.62),
        }
        events = [
            {"t1_vs_t0_building_macro_f1_gain_pp": 0.0}
            for _ in range(10)
        ]
        decision = build_t1_decision(reports, events)
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["decision"], "STOP_AT_T1_COMPONENT_BALANCING_INSUFFICIENT"
        )


if __name__ == "__main__":
    unittest.main()
