"""Print a small decision-oriented summary of a completed frozen Q audit.

These are descriptive Val diagnostics, never a deployable GT-free score or an
automatic choice of Q, A1, or Memory architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _percent(numerator, denominator):
    return None if not denominator else round(100.0 * numerator / denominator, 4)


def summarize(report: dict) -> dict:
    if (report.get("format") != "SRAC_Q_FROZEN_DIAGNOSTIC_V1"
            or report.get("complete") is not True
            or report.get("full_validation_evidence") is not True
            or report.get("image_count") != 450
            or report.get("source_reproduction", {}).get("passed") is not True):
        raise ValueError("Only a full, reproduced Val450 frozen Q audit can be summarized")
    output = {}
    for arm, data in report.get("arms", {}).items():
        mix = data["region_mixing"]
        probability = data["building_probability"]
        labelled = mix.get("labelled_mass", 0)
        foreground = mix.get("high_foreground_mass", 0)
        damaged = probability.get("Damaged") if isinstance(probability, dict) else None
        damaged_low_b = (damaged.get("low_probability_mass_fraction", {}).get("0.5")
                         if isinstance(damaged, dict) else None)
        by_prediction = {}
        for name, bins in data.get("prediction_confusion_by_building_probability", {}).items():
            low = bins[:3]  # fixed predicted B < 0.5
            all_bins = bins
            low_damaged = sum(row["gt_damaged_pixels"] for row in low)
            low_damaged_miss = sum(row["gt_damaged_pixels"] - row["damaged_correct"] for row in low)
            all_damaged_miss = sum(row["gt_damaged_pixels"] - row["damaged_correct"]
                                   for row in all_bins)
            low_intact_fp = sum(row["intact_to_damaged"] for row in low)
            all_intact_fp = sum(row["intact_to_damaged"] for row in all_bins)
            by_prediction[name] = {
                "damaged_misses_at_B_below_0.5": low_damaged_miss,
                "fraction_of_all_damaged_misses_at_B_below_0.5_pct":
                    _percent(low_damaged_miss, all_damaged_miss),
                "damaged_miss_rate_within_low_B_pct":
                    _percent(low_damaged_miss, low_damaged),
                "intact_to_damaged_at_B_below_0.5": low_intact_fp,
                "fraction_of_all_intact_to_damaged_at_B_below_0.5_pct":
                    _percent(low_intact_fp, all_intact_fp),
            }
        output[arm] = {
            "background_foreground_mixed_labelled_mass_pct":
                _percent(mix.get("background_foreground_mixed_mass", 0), labelled),
            "foreground_class_mixed_high_foreground_mass_pct":
                _percent(mix.get("foreground_class_mixed_mass", 0), foreground),
            "mean_foreground_class_purity": mix.get("mean_foreground_class_purity"),
            "gt_damaged_mass_with_predicted_B_below_0.5_pct":
                None if damaged_low_b is None else round(100.0 * damaged_low_b, 4),
            "frozen_prediction_error_by_B": by_prediction,
        }
    return {
        "format": "SRAC_Q_DIAGNOSTIC_SUMMARY_V1",
        "image_count": 450,
        "arms": output,
        "interpretation": (
            "B and Q strata are descriptive, not causal. A low-B error concentration "
            "alone does not prove retraining A1 will help. Region mixing does not "
            "identify building instances. Use paired equal-budget Val results to "
            "test any proposed Q intervention."
        ),
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", required=True, help="Completed audit output or report.json")
    return p


def main():
    args = parser().parse_args()
    path = Path(args.audit)
    if path.is_dir():
        path = path / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(summarize(report), ensure_ascii=False, indent=2,
                     allow_nan=False))


if __name__ == "__main__":
    main()
