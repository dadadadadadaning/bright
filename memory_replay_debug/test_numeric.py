"""CPU tests for fixed gates and evidence about current FP16 rounding."""
import importlib.util
import json
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("memory_replay_numeric", Path(__file__).with_name("numeric.py"))
numeric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric)


class TokenComparisonTests(unittest.TestCase):
    def test_identical_exact_values_keep_fixed_original_gates(self):
        raw = torch.tensor([[.25, -.5, 0.0]], dtype=torch.float32)
        old = raw.to(torch.float16)
        report = numeric.compare_tokens(raw, old)
        self.assertTrue(report["raw_to_stored"]["passed"])
        self.assertTrue(report["cast_to_stored"]["passed"])
        self.assertEqual(report["compared_values"], 3)
        self.assertEqual(report["outliers"], [])
        self.assertEqual(report["raw_to_stored"]["atol"], 2e-5)
        self.assertEqual(report["raw_to_stored"]["rtol"], 2e-5)
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["full_validation_evidence"])

    def test_near_identical_fp32_values_can_cast_to_different_half_neighbors(self):
        lower = torch.tensor([.25], dtype=torch.float16)
        upper = torch.nextafter(lower, torch.full_like(lower, float("inf")))
        midpoint = (lower.float() + upper.float()) / 2.0
        previous_raw, current_raw = midpoint - 1e-6, midpoint + 1e-6
        old = previous_raw.half()
        self.assertTrue(numeric.close_report(current_raw, previous_raw)["passed"])
        self.assertTrue(torch.equal(old, lower))
        self.assertTrue(torch.equal(current_raw.half(), upper))
        report = numeric.compare_tokens(current_raw, old)
        self.assertFalse(report["cast_to_stored"]["passed"])
        item = report["outliers"][0]
        self.assertEqual(item["fp16"]["current_rounding_relation"], "upper_neighbor")
        self.assertFalse(item["fp16"]["interval_contains_current_raw"])
        self.assertEqual(item["fp16"]["cast_distance_in_directional_local_steps"], 1.0)
        self.assertFalse(report["historical_raw_fp32_available"])
        self.assertFalse(item["fp16"]["historical_raw_fp32_reconstructed"])

    def test_rounding_same_half_does_not_override_raw_fixed_gate(self):
        old = torch.tensor([.25], dtype=torch.float16)
        raw = torch.tensor([.25010], dtype=torch.float32)
        report = numeric.compare_tokens(raw, old)
        self.assertFalse(report["raw_to_stored"]["passed"])
        self.assertTrue(report["cast_to_stored"]["passed"])
        self.assertTrue(report["outliers"][0]["fp16"]["interval_contains_current_raw"])
        self.assertEqual(report["outliers"][0]["fp16"]["current_rounding_relation"], "stored_value")
        self.assertNotIn("passed", report)

    def test_exactly_one_half_step_reports_one_step(self):
        old = torch.tensor([.25], dtype=torch.float16)
        next_value = torch.nextafter(old, torch.full_like(old, float("inf")))
        report = numeric.compare_tokens(next_value.float(), old)
        cell = report["outliers"][0]["fp16"]
        self.assertEqual(cell["raw_distance_in_directional_local_steps"], 1)
        self.assertEqual(cell["cast_distance_in_directional_local_steps"], 1)
        self.assertEqual(report["cast_to_stored"]["exact_different_values"], 1)

    def test_negative_one_half_step_uses_lower_direction(self):
        old = torch.tensor([-.25], dtype=torch.float16)
        next_value = torch.nextafter(old, torch.full_like(old, -float("inf")))
        report = numeric.compare_tokens(next_value.float(), old)
        cell = report["outliers"][0]["fp16"]
        self.assertEqual(cell["current_rounding_relation"], "lower_neighbor")
        self.assertEqual(cell["cast_distance_in_directional_local_steps"], 1)

    def test_one_half_step_at_small_magnitude_can_still_pass_unchanged_gate(self):
        old = torch.tensor([.01], dtype=torch.float16)
        upper = torch.nextafter(old, torch.full_like(old, float("inf")))
        report = numeric.compare_tokens(upper.float(), old)
        self.assertTrue(report["cast_to_stored"]["passed"])
        self.assertEqual(report["cast_to_stored"]["exact_different_values"], 1)
        self.assertEqual(report["cast_to_stored"]["outside_tolerance"], 0)

    def test_midpoint_tie_is_reported_without_claiming_original_fp32(self):
        old = torch.tensor([.25], dtype=torch.float16)
        upper = torch.nextafter(old, torch.full_like(old, float("inf")))
        raw = (old.float() + upper.float()) / 2
        cell = numeric.compare_tokens(raw, old)["outliers"][0]["fp16"]
        self.assertEqual(cell["current_raw_position"], "at_upper_midpoint")
        self.assertTrue(cell["midpoint_endpoints_may_require_ties_to_even"])
        self.assertEqual(cell["current_rounding_relation"], "stored_value")

    def test_large_drift_is_not_explained_away_as_quantization(self):
        old, raw = torch.tensor([.25], dtype=torch.float16), torch.tensor([2.0])
        report = numeric.compare_tokens(raw, old)
        self.assertFalse(report["raw_to_stored"]["passed"])
        self.assertFalse(report["cast_to_stored"]["passed"])
        self.assertEqual(report["outliers"][0]["fp16"]["current_rounding_relation"], "beyond_upper_neighbor")
        self.assertGreater(report["outliers"][0]["fp16"]["cast_distance_in_directional_local_steps"], 1)

    def test_dtype_mismatch_rejected_instead_of_silent_cast(self):
        for raw_dtype, old_dtype in ((torch.float64, torch.float16),
                                     (torch.float16, torch.float16),
                                     (torch.float32, torch.float32)):
            with self.assertRaisesRegex(ValueError, "FP32.*FP16"):
                numeric.compare_tokens(torch.ones(2, dtype=raw_dtype), torch.ones(2, dtype=old_dtype))

    def test_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            numeric.compare_tokens(torch.ones(1, 2), torch.ones(2, dtype=torch.float16))

    def test_nonfinite_values_fail_and_report_is_strict_json_safe(self):
        raw = torch.tensor([float("nan"), float("inf"), 1.0])
        old = torch.tensor([0.0, float("inf"), 1.0], dtype=torch.float16)
        report = numeric.compare_tokens(raw, old)
        self.assertFalse(report["raw_to_stored"]["passed"])
        self.assertFalse(report["cast_to_stored"]["passed"])
        self.assertEqual(report["raw_to_stored"]["nonfinite_values"], 2)
        self.assertFalse(report["current_raw"]["finite"])
        self.assertEqual(report["outliers_saved"], 2)
        json.dumps(report, allow_nan=False)

    def test_first_twenty_outliers_are_row_major_and_counts_cover_all(self):
        raw, old = torch.ones(5, 10), torch.zeros(5, 10, dtype=torch.float16)
        report = numeric.compare_tokens(raw, old)
        self.assertEqual(report["compared_values"], 50)
        self.assertEqual(report["outliers_available"], 50)
        self.assertEqual(report["outliers_saved"], 20)
        self.assertEqual(report["outliers"][0]["coordinate"], [0, 0])
        self.assertEqual(report["outliers"][-1]["coordinate"], [1, 9])
        self.assertEqual(report["cast_to_stored"]["exact_different_values"], 50)

    def test_cast_failure_is_prioritized_over_many_earlier_raw_only_rounding_differences(self):
        old = torch.full((30,), .25, dtype=torch.float16)
        raw = torch.full((30,), .25010, dtype=torch.float32)
        raw[-1] = float(torch.nextafter(old[-1], torch.full_like(old[-1], float("inf"))))
        report = numeric.compare_tokens(raw, old)
        self.assertEqual(report["raw_to_stored"]["outside_tolerance"], 30)
        self.assertEqual(report["cast_to_stored"]["outside_tolerance"], 1)
        self.assertEqual(report["outliers_saved"], 20)
        self.assertEqual(report["outliers"][0]["coordinate"], [29])
        self.assertTrue(report["outliers"][0]["outside_cast_tolerance"])
        self.assertEqual([row["flat_index"] for row in report["outliers"][1:]], list(range(19)))
        self.assertTrue(all(not row["outside_cast_tolerance"] for row in report["outliers"][1:]))

    def test_half_maximum_infinite_neighbor_does_not_prove_midpoint_membership(self):
        old = torch.tensor([65504], dtype=torch.float16)
        raw = torch.tensor([1e8], dtype=torch.float32)
        report = numeric.compare_tokens(raw, old)
        cell = report["outliers"][0]["fp16"]
        self.assertFalse(cell["finite_midpoint_interval"])
        self.assertIsNone(cell["interval_contains_current_raw"])
        self.assertEqual(cell["current_rounding_relation"], "current_cast_nonfinite")
        self.assertFalse(report["cast_to_stored"]["passed"])
        json.dumps(report, allow_nan=False)

    def test_read_only_hashes_cover_values_and_dtype(self):
        raw, old = torch.tensor([.25, 1.001], requires_grad=True), torch.tensor([.25, 1.0], dtype=torch.float16)
        before_raw, before_old = raw.detach().clone(), old.clone()
        report = numeric.compare_tokens(raw, old)
        self.assertTrue(torch.equal(before_raw, raw))
        self.assertTrue(torch.equal(before_old, old))
        self.assertIsNone(raw.grad)
        self.assertEqual(report["current_raw"]["sha256"], numeric.tensor_digest(raw))
        self.assertNotEqual(report["current_raw"]["sha256"], report["current_cast"]["sha256"])
        self.assertEqual(report["current_raw"]["minimum"], .25)
        self.assertEqual(report["current_cast"]["maximum"], float(raw.detach().half().max()))

    def test_empty_and_invalid_outlier_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            numeric.compare_tokens(torch.empty(0), torch.empty(0, dtype=torch.float16))
        for maximum in (0, 21, True):
            with self.assertRaisesRegex(ValueError, "1 to 20"):
                numeric.compare_tokens(torch.ones(1), torch.ones(1, dtype=torch.float16), maximum_outliers=maximum)

    def test_non_cpu_input_is_rejected_without_device_computation(self):
        raw = torch.empty(1, device="meta", dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "already be on CPU"):
            numeric.compare_tokens(raw, torch.ones(1, dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()
