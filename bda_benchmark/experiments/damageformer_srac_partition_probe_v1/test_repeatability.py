"""Synthetic, CPU-only contracts for single-image repeatability diagnostics.

Global numerical flags and the cuBLAS environment are restored after every
test.  Passing does not establish BRIGHT/CUDA repeatability or full-Val quality.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.test_entry import TinyFront
from . import repeatability


class RepeatabilityEntryContracts(unittest.TestCase):
    def setUp(self):
        self.saved_flags = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.get_num_threads(),
        )
        self.had_workspace = "CUBLAS_WORKSPACE_CONFIG" in os.environ
        self.saved_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(2)

    def tearDown(self):
        enabled, warn_only, cudnn_deterministic, cudnn_benchmark, threads = self.saved_flags
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_num_threads(threads)
        if self.had_workspace:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.saved_workspace
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    def fixture(self, output):
        image = "ukraine-conflict_00000089"
        args = repeatability.parser().parse_args([
            "--checkpoint", "not-opened.pth", "--a1_checkpoint", "not-opened-a1.pth",
            "--dataset", "not-opened", "--train_list", "train_set.txt", "--val_list", "val_set.txt",
            "--output", str(output), "--device", "cpu", "--workers", "0",
            "--image", image, "--repeats", "3",
        ])
        torch.manual_seed(3407)
        front = TinyFront().requires_grad_(False).eval()
        front.partition.stride = 8
        front.partition.iterations = 3
        pre, post = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
        target = torch.arange(32 * 32).reshape(1, 32, 32) % 4
        batches = [(pre, post, (target > 0).long(), target, [image])]
        return args, front, batches

    def run_mock(self, args, front, batches):
        with mock.patch.object(repeatability, "load_fixed", return_value=(front, {}, {})), \
             mock.patch.object(repeatability, "lists", return_value=([], ["unused-image", args.image])), \
             mock.patch.object(repeatability, "loader", return_value=batches), \
             mock.patch.object(repeatability, "parse_batch", side_effect=lambda batch, device: batch), \
             contextlib.redirect_stdout(io.StringIO()):
            repeatability.run(args)

    @staticmethod
    def saved(output):
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        journal = output / "repeats.jsonl"
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.is_file() else []
        return summary, rows

    def test_success_preserves_parameters_and_never_claims_full_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "repeat"
            args, front, batches = self.fixture(output)
            before = module_fingerprint(front)
            self.run_mock(args, front, batches)
            summary, rows = self.saved(output)
            self.assertEqual(module_fingerprint(front), before)
            self.assertTrue(all(p.grad is None for p in front.parameters()))
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["gates_passed"])
            self.assertTrue(summary["weights_buffers_unchanged"])
            self.assertFalse(summary["weights_updated"])
            self.assertFalse(summary["test_accessed"])
            self.assertFalse(summary["memory_accessed"])
            self.assertFalse(summary["full_validation_evidence"])
            self.assertTrue(summary["diagnostic_subset"])
            self.assertEqual(summary["val_index_one_based"], 2)
            self.assertEqual(summary["repeats_completed"], 3)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["passed"] for row in rows))
            self.assertTrue(all(row["self_repeat_exact"] for row in rows))

    def test_changed_candidate_ids_fail_and_do_not_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "ids_changed"
            args, front, batches = self.fixture(output)
            original = repeatability.run_partition
            def changed(*values, **keywords):
                result = original(*values, **keywords)
                result["ids"] = result["ids"].clone()
                result["ids"][4, 0] = (result["ids"][4, 0] + 1) % result["count"]
                return result
            with mock.patch.object(repeatability, "run_partition", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, "candidate IDs changed"):
                    self.run_mock(args, front, batches)
            summary, rows = self.saved(output)
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["gates_passed"])
            self.assertEqual(summary["repeats_completed"], 1)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["ids_match"])

    def test_changed_fixed_input_fails_with_saved_immutability_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "input_changed"
            args, front, batches = self.fixture(output)
            original = repeatability.run_partition
            def changed(module, embedding, building, **keywords):
                embedding[0, 0, 0, 0] += 1e-4
                return original(module, embedding, building, **keywords)
            with mock.patch.object(repeatability, "run_partition", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, "Fixed evidence"):
                    self.run_mock(args, front, batches)
            summary, rows = self.saved(output)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["repeats_completed"], 1)
            self.assertFalse(rows[0]["fixed_unchanged"]["structural"]["passed"])
            self.assertEqual(rows[0]["fixed_unchanged"]["structural"]["changed_elements"], 1)

    def test_subthreshold_native_drift_fails_exact_gate_but_retains_all_repetitions(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "small_drift"
            args, front, batches = self.fixture(output)
            original = front.partition.forward
            calls = [0]
            def drift(*values, **keywords):
                result = original(*values, **keywords)
                calls[0] += 1
                # Call 1 is native capture; call 3 is repetition 2.  Only one
                # native repetition drifts, by less than the cross-code gate.
                if calls[0] == 3:
                    result["q"] = result["q"].clone()
                    result["q"][0, 4, 0] += 1e-6
                    result["q"][0, 5, 0] -= 1e-6
                return result
            with mock.patch.object(front.partition, "forward", side_effect=drift):
                with self.assertRaisesRegex(RuntimeError, "Repeatability gate FAILED"):
                    self.run_mock(args, front, batches)
            summary, rows = self.saved(output)
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["self_repeat_exact"])
            self.assertEqual(summary["repeats_completed"], 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["passed"] for row in rows], [True, False, True])
            self.assertFalse(rows[1]["self_repeat_exact"])
            for row in rows:
                for check in row["checks"].values():
                    self.assertTrue(check["finite"])
                    self.assertTrue(check["passed"])
                    self.assertLessEqual(check["max_abs"], 2e-5)
            self.assertGreater(summary["checks_max_abs"]["native_first_vs_current"], 0)

    def test_unsupported_deterministic_operation_stops_and_saves_failed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "unsupported"
            args, front, batches = self.fixture(output)
            failure = RuntimeError("synthetic operator has no deterministic implementation")
            with mock.patch.object(repeatability, "run_partition", side_effect=failure) as replay:
                with self.assertRaisesRegex(RuntimeError, "no deterministic implementation"):
                    self.run_mock(args, front, batches)
            summary, rows = self.saved(output)
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["gates_passed"])
            self.assertEqual(summary["repeats_completed"], 0)
            self.assertEqual(rows, [])
            self.assertEqual(replay.call_count, 1)
            self.assertIn("no deterministic implementation", summary["failure"]["message"])

    def test_existing_output_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            args, front, batches = self.fixture(output)
            with self.assertRaises(FileExistsError):
                self.run_mock(args, front, batches)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
