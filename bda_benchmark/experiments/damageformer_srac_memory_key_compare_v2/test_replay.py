"""Archive rounding uncertainty is bounded; true replay drift still fails."""

import unittest

import torch

from .replay import ATOL, RTOL, _half_rounding_cell, stored_token_replay


class StoredTokenReplayTests(unittest.TestCase):
    def test_adjacent_half_boundary_retains_fixed_tolerance(self):
        stored = torch.tensor([.04], dtype=torch.float16)
        next_value = torch.nextafter(stored, torch.full_like(stored, torch.inf))
        midpoint = (stored.double() + next_value.double()) / 2
        actual = (midpoint + 1e-6).float()
        self.assertEqual(actual.half().item(), next_value.item())
        report = stored_token_replay(actual, stored)
        self.assertTrue(report["passed"])
        self.assertTrue(report["quantization_aware"])
        self.assertFalse(report["legacy_report"]["passed"])
        self.assertAlmostEqual(report["max_abs_stored_precision"], 3.0517578125e-05)
        self.assertAlmostEqual(report["max_quantization_interval_distance"], 1e-6, places=8)
        self.assertEqual(report["atol"], 2e-5)
        self.assertEqual(report["rtol"], 2e-5)
        self.assertEqual(report["rounded_equal_fraction"], 0)

    def test_inside_storage_cell_is_compatible_but_not_raw_identity(self):
        stored = torch.tensor([4.], dtype=torch.float16)
        lower, upper = _half_rounding_cell(stored)
        actual = ((upper + stored.double()) / 2).float()
        report = stored_token_replay(actual, stored)
        self.assertTrue(report["passed"])
        self.assertFalse(report["raw_report"]["passed"])
        self.assertEqual(report["max_quantization_interval_distance"], 0)
        self.assertEqual(report["rounded_equal_fraction"], 1)
        self.assertGreater(report["max_abs_raw"], ATOL + RTOL * 4)
        self.assertEqual(lower.item(), 3.9990234375)
        self.assertEqual(upper.item(), 4.001953125)

    def test_real_drift_beyond_cell_and_original_tolerance_rejected(self):
        stored = torch.tensor([4., .04], dtype=torch.float16)
        _, upper = _half_rounding_cell(stored)
        tolerance = ATOL + RTOL * stored.double().abs()
        actual = (upper + tolerance * 2).float()
        report = stored_token_replay(actual, stored)
        self.assertFalse(report["passed"])
        self.assertEqual(report["outside_tolerance"], 2)
        self.assertGreater(report["max_quantization_interval_distance"], tolerance.max())

    def test_asymmetric_power_of_two_and_negative_cells(self):
        stored = torch.tensor([1., -1., 2., -2.], dtype=torch.float16)
        lower, upper = _half_rounding_cell(stored)
        expected_lower = torch.tensor([.999755859375, -1.00048828125,
                                       1.99951171875, -2.0009765625], dtype=torch.float64)
        expected_upper = torch.tensor([1.00048828125, -.999755859375,
                                       2.0009765625, -1.99951171875], dtype=torch.float64)
        torch.testing.assert_close(lower, expected_lower, rtol=0, atol=0)
        torch.testing.assert_close(upper, expected_upper, rtol=0, atol=0)
        self.assertTrue(stored_token_replay(((lower + upper) / 2).float(), stored)["passed"])

    def test_zero_and_subnormal_intervals_are_finite_and_exact(self):
        tiny = 2 ** -24
        stored = torch.tensor([0., -0., tiny, -tiny], dtype=torch.float16)
        lower, upper = _half_rounding_cell(stored)
        torch.testing.assert_close(lower, torch.tensor([0., -tiny / 2,
                                                        tiny / 2, -3 * tiny / 2], dtype=torch.float64),
                                   rtol=0, atol=0)
        torch.testing.assert_close(upper, torch.tensor([tiny / 2, 0.,
                                                        3 * tiny / 2, -tiny / 2], dtype=torch.float64),
                                   rtol=0, atol=0)
        report = stored_token_replay(stored.float(), stored)
        self.assertTrue(report["passed"])
        self.assertTrue(report["finite"])
        self.assertEqual(report["max_quantization_interval_distance"], 0)
        outside_zero = stored_token_replay(torch.tensor([tiny], dtype=torch.float32),
                                           torch.tensor([0.], dtype=torch.float16))
        self.assertEqual(outside_zero["max_quantization_interval_distance"], tiny / 2)

    def test_max_half_cells_do_not_have_infinite_bounds(self):
        stored = torch.tensor([65504., -65504.], dtype=torch.float16)
        lower, upper = _half_rounding_cell(stored)
        torch.testing.assert_close(lower, torch.tensor([65488., -65520.], dtype=torch.float64),
                                   rtol=0, atol=0)
        torch.testing.assert_close(upper, torch.tensor([65520., -65488.], dtype=torch.float64),
                                   rtol=0, atol=0)
        self.assertTrue(torch.isfinite(lower).all())
        self.assertTrue(torch.isfinite(upper).all())
        report = stored_token_replay(torch.tensor([65524., -65524.]), stored)
        self.assertFalse(report["passed"])
        self.assertEqual(report["max_quantization_interval_distance"], 4)
        self.assertFalse(report["rounded_finite"])
        self.assertIsNone(report["max_abs_stored_precision"])
        huge = stored_token_replay(torch.tensor([1e20, -1e20]), stored)
        self.assertFalse(huge["passed"])
        self.assertEqual(huge["outside_tolerance"], 2)

    def test_fp32_archives_keep_original_direct_tolerance(self):
        stored = torch.tensor([1., -2., 0.], dtype=torch.float32)
        inside = stored + torch.tensor([3e-5, -5e-5, 1.9e-5])
        report = stored_token_replay(inside, stored)
        self.assertTrue(report["passed"])
        self.assertFalse(report["quantization_aware"])
        self.assertEqual(report["max_abs_raw"], report["max_quantization_interval_distance"])
        self.assertEqual(report["raw_report"]["passed"], report["passed"])
        outside = stored + torch.tensor([5e-5, -7e-5, 2.1e-5])
        rejected = stored_token_replay(outside, stored)
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["outside_tolerance"], 3)

    def test_nonfinite_raw_or_archive_rejected(self):
        for actual, stored in (
                (torch.tensor([float("nan")]), torch.tensor([1.], dtype=torch.float16)),
                (torch.tensor([float("inf")]), torch.tensor([1.], dtype=torch.float16)),
                (torch.tensor([1.]), torch.tensor([float("nan")], dtype=torch.float16)),
                (torch.tensor([1.]), torch.tensor([float("inf")], dtype=torch.float32))):
            report = stored_token_replay(actual, stored)
            self.assertFalse(report["passed"])
            self.assertFalse(report["finite"])
            self.assertIsNone(report["max_abs_raw"])
            self.assertIsNone(report["max_quantization_interval_distance"])

    def test_shape_and_dtype_cannot_silently_coerce_or_broadcast(self):
        mismatch = stored_token_replay(torch.ones(1, 2), torch.ones(2, dtype=torch.float16))
        self.assertFalse(mismatch["passed"])
        self.assertEqual(mismatch["error"], "shape mismatch")
        for stored_dtype in (torch.float64, torch.bfloat16, torch.int32):
            with self.assertRaisesRegex(ValueError, "archive precision"):
                stored_token_replay(torch.ones(2), torch.ones(2, dtype=stored_dtype))
        with self.assertRaisesRegex(ValueError, "raw FP32"):
            stored_token_replay(torch.ones(2, dtype=torch.float16), torch.ones(2, dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "Empty"):
            stored_token_replay(torch.empty(0), torch.empty(0, dtype=torch.float16))

    def test_check_does_not_modify_replay_or_archived_values(self):
        stored = torch.tensor([.04, 1., -3., 0.], dtype=torch.float16)
        actual = stored.float() + 1e-6
        archived_before, actual_before = stored.clone(), actual.clone()
        stored_token_replay(actual, stored)
        self.assertTrue(torch.equal(stored, archived_before))
        self.assertTrue(torch.equal(actual, actual_before))

    def test_three_statuses_distinguish_possible_from_proven_replay(self):
        # A small FP16 cell is fully inside tolerance, whereas the cell near
        # 4 is too wide to certify the lost raw value even for exact replay.
        # A far-away value is incompatible with every possible original.
        stored = torch.tensor([.04, 4., 4.], dtype=torch.float16)
        actual = torch.tensor([stored[0].item(), 4., 4.01], dtype=torch.float32)
        report = stored_token_replay(actual, stored)
        self.assertFalse(report["passed"])
        self.assertEqual(report["definite_within_tol_count"], 1)
        self.assertEqual(report["storage_uncertain_count"], 1)
        self.assertEqual(report["definite_drift_count"], 1)
        self.assertEqual(sum(report[key] for key in (
            "definite_within_tol_count", "storage_uncertain_count", "definite_drift_count")), 3)
        self.assertGreater(report["d_max"], report["d_min"])
        compatible = stored_token_replay(actual[:2], stored[:2])
        self.assertTrue(compatible["passed"])
        self.assertEqual(compatible["storage_uncertain_count"], 1)
        self.assertEqual(compatible["definite_drift_count"], 0)

    def test_fp32_has_no_storage_uncertainty_category(self):
        report = stored_token_replay(torch.tensor([1., 1.001]), torch.tensor([1., 1.]))
        self.assertEqual(report["storage_uncertain_count"], 0)
        self.assertEqual(report["definite_within_tol_count"], 1)
        self.assertEqual(report["definite_drift_count"], 1)
        self.assertEqual(report["d_min"], report["d_max"])


if __name__ == "__main__":
    unittest.main()
