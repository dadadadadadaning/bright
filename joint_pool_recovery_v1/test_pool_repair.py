"""CPU tests for scope, independent references and unchanged routing semantics.

These tests cannot establish that the candidate resolves the server CUDA fault.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import pool_repair
from pool_ops import contiguous_index_algorithm, native_original_algorithm


PROJECT = Path("/workspace/bright")
if PROJECT.is_dir() and str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
native_pool = importlib.import_module(
    "bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool")
association = importlib.import_module(
    "bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association")


def inputs(channels=3):
    content = torch.arange(channels * 6, dtype=torch.float32).reshape(1, channels, 2, 3) / 7
    weights = torch.arange(54, dtype=torch.float32).reshape(1, 9, 6) / 200
    ids = torch.arange(54, dtype=torch.long).reshape(9, 6).remainder(4)
    return content, weights, ids, 4


class PoolRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original_class = native_pool.PoolGuard
        self.original_pool = native_pool._ORIGINAL_POOL
        self.original_association = association.soft_pool_unpool

    def tearDown(self):
        self.assertIs(native_pool.PoolGuard, self.original_class)
        self.assertIs(native_pool._ORIGINAL_POOL, self.original_pool)
        self.assertIs(association.soft_pool_unpool, self.original_association)
        self.assertIsNone(native_pool._ACTIVE)
        self.temporary.cleanup()

    def test_dispatch_uses_cuda_candidate_and_cpu_original(self):
        original = mock.Mock(return_value="original")
        candidate = mock.Mock(return_value="candidate")
        cpu = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
        cuda = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        self.assertEqual(pool_repair.dispatch_pool(cpu, 2, 3, 4, original_pool=original,
                                                  candidate_pool=candidate), "original")
        self.assertEqual(pool_repair.dispatch_pool(cuda, 2, 3, 4, original_pool=original,
                                                  candidate_pool=candidate), "candidate")
        original.assert_called_once_with(cpu, 2, 3, 4)
        candidate.assert_called_once_with(cuda, 2, 3, 4)

    def test_dispatch_rejects_other_device(self):
        meta = types.SimpleNamespace(device=types.SimpleNamespace(type="meta"))
        with self.assertRaisesRegex(ValueError, "only CPU or CUDA"):
            pool_repair.dispatch_pool(meta, None, None, 4, original_pool=self.original_pool)

    def test_imported_callers_are_intercepted_and_restored(self):
        importer = types.ModuleType("bda_benchmark.recovery_test_importer")
        importer.pool = self.original_pool
        sys.modules[importer.__name__] = importer
        try:
            with pool_repair.patched_pool_guard(native_pool) as guard_class:
                self.assertIs(native_pool.PoolGuard, guard_class)
                self.assertIs(native_pool._ORIGINAL_POOL, self.original_pool)
                with guard_class(self.root / "audit") as guard:
                    self.assertIsNot(importer.pool, self.original_pool)
                    self.assertIs(association.soft_pool_unpool, importer.pool)
                    actual = importer.pool(*inputs())
                    expected = self.original_pool(*inputs())
                    for key in actual:
                        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                    self.assertIs(native_pool._ORIGINAL_POOL, self.original_pool)
                    self.assertEqual(guard.calls, 1)
                    self.assertEqual(guard.reference_checks, 1)
                self.assertIs(importer.pool, self.original_pool)
            report = json.loads((self.root / "audit" / "guard.json").read_text())
            self.assertTrue(report["restored"])
            self.assertEqual(report["backend"], pool_repair.BACKEND)
            self.assertEqual(report["original_backend"], "native")
        finally:
            sys.modules.pop(importer.__name__, None)

    def test_cpu_reference_never_uses_candidate(self):
        observed = []
        dispatcher = pool_repair.dispatch_pool

        def inspect_dispatch(content, *args, **kwargs):
            observed.append((content.device.type, kwargs["original_pool"]))
            return dispatcher(content, *args, **kwargs)

        with mock.patch.object(pool_repair, "dispatch_pool", side_effect=inspect_dispatch), \
                mock.patch.object(pool_repair, "contiguous_index_algorithm", side_effect=AssertionError("CPU candidate used")):
            with pool_repair.patched_pool_guard(native_pool) as guard_class:
                with guard_class(self.root / "audit"):
                    association.soft_pool_unpool(*inputs())
        self.assertEqual(observed, [("cpu", self.original_pool), ("cpu", self.original_pool)])

    def test_first_reference_for_each_unique_layout_and_count(self):
        with pool_repair.patched_pool_guard(native_pool) as guard_class:
            with guard_class(self.root / "audit") as guard:
                association.soft_pool_unpool(*inputs())
                changed_values = list(inputs())
                changed_values[0] += 1
                association.soft_pool_unpool(*changed_values)
                association.soft_pool_unpool(*inputs(channels=5))
                changed_count = list(inputs())
                changed_count[-1] = 5
                association.soft_pool_unpool(*changed_count)
                changed_stride = list(inputs())
                changed_stride[1] = torch.ones((1, 9, 1)).expand(1, 9, 6)
                association.soft_pool_unpool(*changed_stride)
                self.assertEqual(guard.calls, 5)
                self.assertEqual(guard.reference_checks, 4)
                self.assertEqual(len(guard.tested_layouts), 4)
                self.assertTrue(all(row["cpu_reference_passed"] for row in guard.tested_layouts))

    def test_requested_reference_checks_every_call(self):
        with pool_repair.patched_pool_guard(native_pool) as guard_class:
            with guard_class(self.root / "audit", check_reference=True) as guard:
                for _ in range(3):
                    association.soft_pool_unpool(*inputs())
                self.assertEqual(guard.reference_checks, 3)
                self.assertEqual(len(guard.tested_layouts), 1)

    def test_reference_failure_propagates_preserves_snapshot_and_restores(self):
        with mock.patch.object(native_pool, "compare_results", return_value={"passed": False}):
            with self.assertRaisesRegex(ValueError, "reference gate failed"):
                with pool_repair.patched_pool_guard(native_pool) as guard_class:
                    with guard_class(self.root / "audit") as guard:
                        guard.set_image("failed-example")
                        association.soft_pool_unpool(*inputs())
            self.assertEqual(guard.tested_layouts, [])
        failure = json.loads((self.root / "audit" / "failure.json").read_text())
        self.assertEqual(failure["phase"], "cpu_reference_gate")
        self.assertEqual(failure["backend"], pool_repair.BACKEND)
        snapshot = torch.load(failure["input"], weights_only=True)
        self.assertEqual(snapshot["backend"], pool_repair.BACKEND)
        self.assertEqual(snapshot["image"], "failed-example")
        self.assertEqual(snapshot["comparison"], {"passed": False})

    def test_kernel_exception_is_not_retried_and_records_cpu_input(self):
        with mock.patch.object(pool_repair, "dispatch_pool", side_effect=RuntimeError("CUDA poisoned sentinel")) as dispatch:
            with self.assertRaisesRegex(RuntimeError, "CUDA poisoned sentinel"):
                with pool_repair.patched_pool_guard(native_pool) as guard_class:
                    with guard_class(self.root / "audit"):
                        association.soft_pool_unpool(*inputs())
        self.assertEqual(dispatch.call_count, 1)
        failure = json.loads((self.root / "audit" / "failure.json").read_text())
        self.assertEqual(failure["phase"], "pool_kernel")
        self.assertTrue(failure["restart_process_required"])
        snapshot = torch.load(failure["input"], weights_only=True)
        self.assertEqual(snapshot["content"].device.type, "cpu")
        self.assertEqual(snapshot["ids"].device.type, "cpu")

    def test_invalid_id_is_rejected_without_kernel_or_clamping(self):
        bad = list(inputs())
        bad[2][0, 0] = 400
        with mock.patch.object(pool_repair, "dispatch_pool") as dispatch:
            with self.assertRaisesRegex(ValueError, "Out-of-range"):
                with pool_repair.patched_pool_guard(native_pool) as guard_class:
                    with guard_class(self.root / "audit"):
                        association.soft_pool_unpool(*bad)
        dispatch.assert_not_called()
        failure = json.loads((self.root / "audit" / "failure.json").read_text())
        self.assertEqual(failure["phase"], "input_validation")
        snapshot = torch.load(failure["input"], weights_only=True)
        self.assertEqual(int(snapshot["ids"][0, 0]), 400)

    def test_nested_patch_rejected_and_outer_scope_survives(self):
        with pool_repair.patched_pool_guard(native_pool) as guard_class:
            with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                with pool_repair.patched_pool_guard(native_pool):
                    pass
            self.assertIs(native_pool.PoolGuard, guard_class)
            self.assertIs(native_pool._ORIGINAL_POOL, self.original_pool)

    def test_patch_over_active_original_guard_rejected(self):
        with self.original_class(self.root / "original"):
            with self.assertRaisesRegex(RuntimeError, "active guard"):
                with pool_repair.patched_pool_guard(native_pool):
                    pass

    def test_grouped_mode_rejected(self):
        with pool_repair.patched_pool_guard(native_pool) as guard_class:
            with self.assertRaisesRegex(ValueError, "grouped"):
                guard_class(self.root / "audit", mode="grouped")

    def test_scope_exception_restores_original_class_and_callable(self):
        with self.assertRaisesRegex(RuntimeError, "scope sentinel"):
            with pool_repair.patched_pool_guard(native_pool):
                raise RuntimeError("scope sentinel")

    def test_capture_all_records_actual_backend_and_layout(self):
        with pool_repair.patched_pool_guard(native_pool) as guard_class:
            with guard_class(self.root / "audit", capture="all") as guard:
                association.soft_pool_unpool(*inputs())
        snapshot = torch.load(guard.saved_inputs[0], weights_only=True)
        self.assertEqual(snapshot["backend"], pool_repair.BACKEND)
        self.assertEqual(snapshot["layout"]["content"]["shape"], [1, 3, 2, 3])
        self.assertEqual(snapshot["caller"]["function"],
                         "test_capture_all_records_actual_backend_and_layout")
        for name in ("manifest.json", "progress.json", "guard.json"):
            record = json.loads((self.root / "audit" / name).read_text())
            self.assertEqual(record["backend"], pool_repair.BACKEND)
            self.assertEqual(record["original_backend"], "native")

    def test_candidate_preserves_duplicate_routes_empty_tokens_and_unnormalized_weights(self):
        content, weights, ids, _ = inputs(channels=2)
        ids.fill_(0)
        # Deliberately unnormalized Q*attention and a mass below clamp epsilon.
        weights.fill_(1e-9)
        original = native_original_algorithm(content, weights, ids, 5)
        candidate = contiguous_index_algorithm(content, weights, ids, 5)
        for key in original:
            torch.testing.assert_close(original[key], candidate[key], atol=0, rtol=0)
        self.assertEqual(float(candidate["mass"][0, 1]), 0)
        self.assertLess(float(candidate["mass"][0, 0]), 1e-6)
        expected_numerator = (content.flatten(2).transpose(1, 2) * 1e-9).sum(1) * 9
        torch.testing.assert_close(candidate["tokens"][:, 0], expected_numerator / 1e-6)


if __name__ == "__main__":
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    unittest.main()
