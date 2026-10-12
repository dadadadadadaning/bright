"""CPU contracts for strict numerical setup and observation-only Q evidence.

CUDA initialization is mocked; these tests do not prove CUDA reproducibility.
Every changed process flag/environment value is restored after each test.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from .numeric_runtime import assert_determinism, configure_determinism, q_difference


_WORKSPACE = "CUBLAS_WORKSPACE_CONFIG"


class NumericRuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.saved_flags = self.flags()
        self.had_workspace = _WORKSPACE in os.environ
        self.saved_workspace = os.environ.get(_WORKSPACE)

    def tearDown(self):
        torch.use_deterministic_algorithms(
            self.saved_flags["enabled"], warn_only=self.saved_flags["warn_only"],
        )
        torch.backends.cudnn.deterministic = self.saved_flags["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = self.saved_flags["cudnn_benchmark"]
        if self.had_workspace:
            os.environ[_WORKSPACE] = self.saved_workspace
        else:
            os.environ.pop(_WORKSPACE, None)

    @staticmethod
    def flags():
        return dict(
            enabled=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
        )

    @staticmethod
    def set_valid_state(workspace=":4096:8"):
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ[_WORKSPACE] = workspace

    def test_configuration_enables_strict_flags_before_cuda_initialization(self):
        os.environ.pop(_WORKSPACE, None)
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        with mock.patch("torch.cuda.is_initialized", return_value=False):
            report = configure_determinism()
        self.assertIsInstance(report, dict)
        self.assertEqual(os.environ[_WORKSPACE], ":4096:8")
        self.assertEqual(self.flags(), dict(enabled=True, warn_only=False,
                                           cudnn_deterministic=True, cudnn_benchmark=False))
        self.assertIsInstance(assert_determinism(), dict)

    def test_missing_workspace_after_cuda_initialization_is_rejected(self):
        os.environ.pop(_WORKSPACE, None)
        with mock.patch("torch.cuda.is_initialized", return_value=True):
            with self.assertRaises((RuntimeError, ValueError)):
                configure_determinism()
        self.assertNotIn(_WORKSPACE, os.environ)

    def test_supported_existing_workspace_is_preserved_even_after_initialization(self):
        for workspace in (":16:8", ":4096:8"):
            with self.subTest(workspace=workspace):
                os.environ[_WORKSPACE] = workspace
                with mock.patch("torch.cuda.is_initialized", return_value=True):
                    report = configure_determinism()
                self.assertIsInstance(report, dict)
                self.assertEqual(os.environ[_WORKSPACE], workspace)
                self.assertIsInstance(assert_determinism(), dict)

    def test_unsupported_workspace_is_rejected_not_silently_replaced(self):
        for workspace in ("", "4096:8", ":4096:2", "random"):
            with self.subTest(workspace=workspace):
                os.environ[_WORKSPACE] = workspace
                with mock.patch("torch.cuda.is_initialized", return_value=False):
                    with self.assertRaises((RuntimeError, ValueError)):
                        configure_determinism()
                self.assertEqual(os.environ[_WORKSPACE], workspace)

    def test_assertion_is_read_only_on_success_and_rejects_warn_only(self):
        self.set_valid_state()
        before = self.flags()
        workspace = os.environ[_WORKSPACE]
        self.assertIsInstance(assert_determinism(), dict)
        self.assertEqual(self.flags(), before)
        self.assertEqual(os.environ[_WORKSPACE], workspace)
        torch.use_deterministic_algorithms(True, warn_only=True)
        before = self.flags()
        with self.assertRaises((RuntimeError, ValueError)):
            assert_determinism()
        self.assertEqual(self.flags(), before)
        self.assertEqual(os.environ[_WORKSPACE], workspace)

    def test_assertion_rejects_each_weakened_flag_without_repair(self):
        changes = (
            lambda: torch.use_deterministic_algorithms(False),
            lambda: setattr(torch.backends.cudnn, "deterministic", False),
            lambda: setattr(torch.backends.cudnn, "benchmark", True),
            lambda: os.environ.pop(_WORKSPACE),
            lambda: os.environ.__setitem__(_WORKSPACE, ":4096:2"),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.set_valid_state()
                change()
                before = self.flags()
                workspace = os.environ.get(_WORKSPACE)
                with self.assertRaises((RuntimeError, ValueError)):
                    assert_determinism()
                self.assertEqual(self.flags(), before)
                self.assertEqual(os.environ.get(_WORKSPACE), workspace)

    def test_q_difference_equal_tensors_and_inputs_are_unchanged(self):
        left = torch.tensor([[[0.7, 0.2], [0.3, 0.8]]])
        right = left.clone()
        ids = torch.tensor([[0, 0], [1, 1]])
        snapshots = [value.clone() for value in (left, right, ids)]
        report = q_difference(left, right, ids)
        self.assertTrue(report["passed"])
        self.assertTrue(report["finite"])
        for key in ("max_abs", "mean_abs", "elements_above_tolerance",
                    "candidate_argmax_changed_positions", "region_id_changed_positions"):
            self.assertEqual(report[key], 0, msg=key)
        for value, saved in zip((left, right, ids), snapshots):
            self.assertTrue(torch.equal(value, saved))

    def test_q_difference_uses_original_absolute_tolerance_without_relative_slack(self):
        left = torch.full((1, 2, 2), 0.5, dtype=torch.float64)
        ids = torch.tensor([[0, 0], [1, 1]])
        below = left.clone()
        below[0, 0, 0] += 1.5e-5
        below[0, 1, 0] -= 1.5e-5
        self.assertTrue(q_difference(left, below, ids)["passed"])
        above = left.clone()
        above[0, 0, 0] += 2.5e-5
        above[0, 1, 0] -= 2.5e-5
        report = q_difference(left, above, ids)
        self.assertFalse(report["passed"])
        self.assertEqual(report["elements_above_tolerance"], 2)
        self.assertAlmostEqual(report["max_abs"], 2.5e-5, places=12)
        self.assertAlmostEqual(report["mean_abs"], 1.25e-5, places=12)
        # An allclose(atol=2e-5, rtol=2e-5) comparison would mistakenly pass
        # this case at values near .5; the replay contract is absolute.

    def test_q_difference_distinguishes_candidate_slot_and_true_region_changes(self):
        left = torch.tensor([[[0.8, 0.8], [0.2, 0.2]]])
        right = torch.tensor([[[0.2, 0.8], [0.8, 0.2]]])
        ids = torch.tensor([[2, 2], [2, 5]])
        report = q_difference(left, right, ids)
        self.assertEqual(report["candidate_argmax_changed_positions"], 1)
        self.assertEqual(report["region_id_changed_positions"], 0)
        right[0, :, 1] = torch.tensor([0.2, 0.8])
        report = q_difference(left, right, ids)
        self.assertEqual(report["candidate_argmax_changed_positions"], 2)
        self.assertEqual(report["region_id_changed_positions"], 1)

    def test_q_difference_observes_values_without_renormalizing(self):
        # Both are proportional vectors, so normalizing would erase the drift.
        left = torch.tensor([[[0.8], [0.2]]])
        right = 2 * left
        ids = torch.tensor([[0], [1]])
        saved = right.clone()
        report = q_difference(left, right, ids)
        self.assertFalse(report["passed"])
        self.assertGreater(report["max_abs"], 0.79)
        self.assertTrue(torch.equal(right, saved))

    def test_q_difference_rejects_nan_and_infinity(self):
        left = torch.tensor([[[0.8, 0.3], [0.2, 0.7]]])
        ids = torch.tensor([[0, 0], [1, 1]])
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                right = left.clone()
                right[0, 0, 0] = value
                report = q_difference(left, right, ids)
                self.assertFalse(report["passed"])
                self.assertFalse(report["finite"])

    def test_q_difference_rejects_shape_mismatches(self):
        left = torch.ones(1, 2, 3) / 2
        ids = torch.tensor([[0, 0, 0], [1, 1, 1]])
        for right, candidate_ids in ((left[:, :, :2], ids), (left, ids[:, :2]),
                                     (left.squeeze(0), ids), (left, ids[None])):
            with self.subTest(right_shape=tuple(right.shape), ids_shape=tuple(candidate_ids.shape)):
                with self.assertRaises(ValueError):
                    q_difference(left, right, candidate_ids)


if __name__ == "__main__":
    unittest.main()
