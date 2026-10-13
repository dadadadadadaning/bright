"""CPU tests for exact live-packet analysis and narrow continuation gates."""
from __future__ import annotations

import copy
import io
import json
import unittest
from unittest import mock

import torch

import analysis
from numeric_core import pool_literal
from pool_ops import tensor_layout


def packet(mode="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(44)
    raw = torch.randn(1, 256, 2, 2, generator=generator)
    x = torch.cat((raw, raw.square()), 1)
    q = torch.rand(1, 9, 4, generator=generator)
    q /= q.sum(1, keepdim=True)
    ids = torch.arange(36).reshape(9, 4).remainder(2)
    backend = analysis.BACKENDS[mode]
    snapshot = {"format": "native_pool_input_v1", "content": x, "weights": q,
                "ids": ids, "token_count": 2, "image": "event_0000001", "call": 3,
                "backend": backend, "caller": {"file": "/repo/model.py", "function": "forward", "line": 288},
                "layout": {name: tensor_layout(value) for name, value in
                           (("content", x), ("weights", q), ("ids", ids))}}
    with analysis.cpu_threads(2):
        expected = pool_literal(x, q, ids, 2)
    return {"format": "LIVE_POOL_CAPTURE_V2", "snapshot": snapshot,
            "actual": {key: value.clone() for key, value in expected.items()},
            "expected": {key: value.clone() for key, value in expected.items()},
            "meta": {"backend": backend, "image": snapshot["image"], "call": 3,
                     "content_channels": 512, "original_num_threads": 2,
                     "cpu_execution_num_threads": 2,
                     "captured_before_original_comparison": True,
                     "original_source": "/repo/association.py", "original_sha256": "a"*64,
                     "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                     "binding": {"experiment_kind": "synthetic"}}}


class LiveAnalysisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.patches = []
        for name in ("_lazy_init", "is_available", "device_count"):
            patched = mock.patch.object(torch.cuda, name,
                        side_effect=AssertionError("CPU analysis queried CUDA: " + name))
            patched.start()
            self.patches.append(patched)

    def tearDown(self):
        for patched in reversed(self.patches):
            patched.stop()

    def analyze(self, value, mode="cpu", **kwargs):
        return analysis.analyze_packet(value, mode, emit=lambda *args, **kwargs: None, **kwargs)

    def test_declared_cpu_packet_passes_source_and_independent_controls(self):
        result = self.analyze(packet())
        for key in ("execution_complete", "diagnostic_complete", "adequate_capture",
                    "cpu64_reference_agreement", "cpu32_reference_reproduced",
                    "CPU_backend_vs_FP64_passed", "cpu_continuation_checks_passed"):
            self.assertTrue(result[key], key)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["root_cause_confirmed"])
        self.assertIn("cpu32_source_threads", result["timings_seconds"])
        self.assertIn("cpu64_independent_grouped", result["timings_seconds"])
        json.dumps(result, allow_nan=False)

    def test_live_gpu_numeric_difference_is_observation_not_cpu_approval(self):
        value = packet("live")
        value["actual"]["tokens"][0, 0, 0] += .1
        result = self.analyze(value, "live")
        self.assertTrue(result["adequate_capture"])
        self.assertTrue(result["diagnostic_complete"])
        self.assertFalse(result["comparisons"]["actual_vs_exact_live_expected"]["passed"])
        self.assertFalse(result["CPU_backend_vs_FP64_passed"])
        self.assertFalse(result["cpu_continuation_checks_passed"])

    def test_changed_cpu_outputs_block_cpu_continuation(self):
        value = packet()
        value["actual"]["tokens"][0, 0, 0] += .1
        result = self.analyze(value)
        self.assertFalse(result["CPU_backend_vs_FP64_passed"])
        self.assertFalse(result["cpu_continuation_checks_passed"])

    def test_tampered_expected_even_matching_actual_does_not_self_approve(self):
        value = packet()
        for key in ("actual", "expected"):
            value[key]["tokens"][0, 0, 0] += .1
        result = self.analyze(value)
        self.assertTrue(result["comparisons"]["actual_vs_exact_live_expected"]["passed"])
        self.assertFalse(result["cpu32_reference_reproduced"])
        self.assertFalse(result["cpu_continuation_checks_passed"])

    def test_four_thread_contrast_is_diagnostic_not_cpu_production_gate(self):
        value = packet()
        original = analysis.pool_literal
        def contrasted(*args, **kwargs):
            result = original(*args, **kwargs)
            if torch.get_num_threads() == 4 and kwargs.get("precision", "fp32") == "fp32":
                result["tokens"] = result["tokens"] + .1
            return result
        with mock.patch.object(analysis, "pool_literal", side_effect=contrasted):
            result = self.analyze(value)
        self.assertTrue(result["cpu_thread_difference_present"])
        self.assertTrue(result["cpu32_reference_reproduced"])
        self.assertTrue(result["CPU_backend_vs_FP64_passed"])
        self.assertTrue(result["cpu_continuation_checks_passed"])

    def test_disagreeing_independent_fp64_controls_block(self):
        original = analysis.cpu_reference64
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            result["tokens"][0, 0, 0] += .1
            return result
        with mock.patch.object(analysis, "cpu_reference64", side_effect=changed):
            result = self.analyze(packet())
        self.assertFalse(result["cpu64_reference_agreement"])
        self.assertFalse(result["cpu_continuation_checks_passed"])

    def test_snapshot_metadata_and_parent_identity_mismatch_rejected(self):
        for key, changed in (("image", "other"), ("call", 4), ("backend", "other")):
            with self.subTest(key=key):
                value = packet()
                value["meta"][key] = changed
                with self.assertRaisesRegex(ValueError, "identity|backend"):
                    self.analyze(value)
        with self.assertRaisesRegex(ValueError, "parent binding"):
            self.analyze(packet(), expected_call=9)

    def test_live_packet_cannot_substitute_cpu_backend_probe(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            self.analyze(packet("live"), "cpu")

    def test_missing_precomparison_proof_and_layout_rejected(self):
        value = packet()
        value["meta"]["captured_before_original_comparison"] = False
        with self.assertRaisesRegex(ValueError, "before"):
            self.analyze(value)
        value = packet()
        value["snapshot"].pop("layout")
        with self.assertRaisesRegex(ValueError, "layout"):
            self.analyze(value)

    def test_nonfinite_or_wrong_shape_live_outputs_rejected(self):
        value = packet()
        value["actual"]["tokens"][0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "Invalid/nonfinite"):
            self.analyze(value)
        value = packet()
        value["expected"]["mass"] = value["expected"]["mass"][:, :1]
        with self.assertRaisesRegex(ValueError, "Invalid/nonfinite"):
            self.analyze(value)

    def test_reference_thread_policy_mismatch_rejected(self):
        value = packet()
        value["meta"]["cpu_execution_num_threads"] = 4
        with self.assertRaisesRegex(ValueError, "thread count"):
            self.analyze(value)

    def test_parent_original_source_binding_checked(self):
        value = packet()
        evidence = {"preserved_sha256": {"/repo/association.py": "a"*64}}
        self.assertTrue(self.analyze(value, evidence=evidence)["adequate_capture"])
        evidence["preserved_sha256"]["/repo/association.py"] = "b"*64
        with self.assertRaisesRegex(ValueError, "original source"):
            self.analyze(value, evidence=evidence)

    def test_thread_count_restored_after_success_and_failure(self):
        before = torch.get_num_threads()
        self.analyze(packet())
        self.assertEqual(torch.get_num_threads(), before)
        with mock.patch.object(analysis, "pool_literal", side_effect=ValueError("synthetic CPU failure")):
            with self.assertRaisesRegex(ValueError, "synthetic CPU"):
                self.analyze(packet())
        self.assertEqual(torch.get_num_threads(), before)


if __name__ == "__main__":
    unittest.main()
