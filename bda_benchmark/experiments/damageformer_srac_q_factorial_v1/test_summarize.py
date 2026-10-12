"""CPU-only diagnostic summary contract checks."""

from __future__ import annotations

import unittest

from .summarize import summarize


def fake_report():
    row = dict(gt_damaged_pixels=10, damaged_correct=6,
               intact_to_damaged=2)
    return {
        "format": "SRAC_Q_FROZEN_DIAGNOSTIC_V1",
        "complete": True, "full_validation_evidence": True,
        "image_count": 450, "source_reproduction": {"passed": True},
        "arms": {"old": {
            "region_mixing": dict(labelled_mass=100,
                                  high_foreground_mass=50,
                                  background_foreground_mixed_mass=20,
                                  foreground_class_mixed_mass=10,
                                  mean_foreground_class_purity=0.8),
            "building_probability": {"Damaged": {
                "low_probability_mass_fraction": {"0.5": 0.25}}},
            "prediction_confusion_by_building_probability": {
                "a1_pixel": [dict(row) for _ in range(5)]},
        }},
    }


class SummaryTests(unittest.TestCase):
    def test_rates_use_fixed_denominators(self):
        result = summarize(fake_report())["arms"]["old"]
        self.assertEqual(result["background_foreground_mixed_labelled_mass_pct"], 20)
        self.assertEqual(result["foreground_class_mixed_high_foreground_mass_pct"], 20)
        self.assertEqual(result["gt_damaged_mass_with_predicted_B_below_0.5_pct"], 25)
        a1 = result["frozen_prediction_error_by_B"]["a1_pixel"]
        self.assertEqual(a1["damaged_misses_at_B_below_0.5"], 12)
        self.assertEqual(a1["fraction_of_all_damaged_misses_at_B_below_0.5_pct"], 60)
        self.assertEqual(a1["fraction_of_all_intact_to_damaged_at_B_below_0.5_pct"], 60)

    def test_rejects_short_or_failed_report(self):
        for mutation in ({"image_count": 449},
                         {"source_reproduction": {"passed": False}},
                         {"full_validation_evidence": False}):
            report = fake_report()
            report.update(mutation)
            with self.assertRaises(ValueError):
                summarize(report)


if __name__ == "__main__":
    unittest.main()
