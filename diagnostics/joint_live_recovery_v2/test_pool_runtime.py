"""CPU verification of explicit backend scope, exact capture and guarded stop.

CUDA dispatch is simulated where needed; this is not a real GPU validation.
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

import pool_runtime


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


def stats_inputs():
    raw = torch.arange(256 * 4, dtype=torch.float32).reshape(1, 256, 2, 2) / 700
    content = torch.cat((raw, raw.square()), dim=1)
    weights = torch.arange(36, dtype=torch.float32).reshape(1, 9, 4) / 200
    ids = torch.arange(36, dtype=torch.long).reshape(9, 4).remainder(4)
    return content, weights, ids, 4


class PoolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original = {name: getattr(native_pool, name) for name in
                         ("PoolGuard", "_ORIGINAL_POOL", "snapshot_inputs", "compare_results")}
        self.association_original = association.soft_pool_unpool

    def tearDown(self):
        for name, value in self.original.items():
            self.assertIs(getattr(native_pool, name), value)
        self.assertIs(association.soft_pool_unpool, self.association_original)
        self.assertIsNone(native_pool._ACTIVE)
        self.temporary.cleanup()

    def test_cpu_backend_exact_outputs_and_no_cuda_scatter(self):
        with mock.patch.object(pool_runtime, "contiguous_index_algorithm", side_effect=AssertionError("CUDA scatter called")):
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard") as guard:
                    actual = association.soft_pool_unpool(*inputs())
                    expected = self.original["_ORIGINAL_POOL"](*inputs())
                    for key in actual:
                        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                        self.assertEqual(actual[key].dtype, torch.float32)
                    self.assertEqual(guard.cuda_scatter_calls, 0)
                    self.assertEqual(guard.cpu_arithmetic_calls, 2)

    def test_cpu_snapshot_execution_and_transfer_are_explicit(self):
        source = native_pool.snapshot_inputs(*inputs())
        observed = []

        def original(c, p, ids, count):
            observed.append((c.device.type, p.device.type, ids.device.type))
            return self.original["_ORIGINAL_POOL"](c, p, ids, count)

        destination = torch.device("cuda:0")
        with mock.patch.object(pool_runtime, "transfer_outputs", side_effect=lambda outputs, device: outputs) as transfer:
            result = pool_runtime.cpu_pool_from_snapshot(source, destination, original_pool=original)
        self.assertEqual(observed, [("cpu", "cpu", "cpu")])
        self.assertEqual(transfer.call_args.args[1], destination)
        self.assertEqual(result["tokens"].device.type, "cpu")

    def test_guard_uses_cached_validated_snapshot_for_simulated_cuda_dispatch(self):
        source_pool = self.original["_ORIGINAL_POOL"]
        transfers = []
        gate = self.original["compare_results"]

        def compare_with_dispatch(actual, expected):
            fake_content = types.SimpleNamespace(dtype=torch.float32,
                device=types.SimpleNamespace(type="cuda", index=0))
            fake_weights = types.SimpleNamespace(dtype=torch.float32)
            replay = native_pool._ORIGINAL_POOL(fake_content, fake_weights, object(), 4)
            for key in replay:
                torch.testing.assert_close(replay[key], source_pool(*inputs())[key], atol=0, rtol=0)
            return gate(actual, expected)

        def transfer(outputs, device):
            transfers.append(device.type)
            return outputs

        with mock.patch.object(native_pool, "compare_results", side_effect=compare_with_dispatch), \
                mock.patch.object(pool_runtime, "transfer_outputs", side_effect=transfer), \
                mock.patch.object(pool_runtime, "contiguous_index_algorithm", side_effect=AssertionError("CUDA scatter called")):
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard"):
                    association.soft_pool_unpool(*inputs())
        self.assertEqual(transfers, ["cuda"])

    def test_probe_stops_only_after_successful_stats_call(self):
        with self.assertRaises(pool_runtime.ProbeComplete) as stop:
            with pool_runtime.patched_guard(native_pool, backend=pool_runtime.CPU_BACKEND,
                                           stop_after_stats=True) as guard_class:
                with guard_class(self.root / "guard") as guard:
                    guard.set_image("synthetic")
                    association.soft_pool_unpool(*inputs())
                    self.assertEqual(guard.calls, 1)
                    association.soft_pool_unpool(*stats_inputs())
        self.assertEqual(guard.calls, 2)
        self.assertEqual(guard.failures, [])
        self.assertEqual(stop.exception.report["content_channels"], 512)
        self.assertTrue(stop.exception.report["strict_cpu_gate_passed"])
        self.assertFalse(stop.exception.report["training_started"])
        self.assertTrue(guard.restored)

    def test_live_probe_on_cpu_captures_and_stops_without_switching_backend(self):
        with self.assertRaises(pool_runtime.ProbeComplete) as stop:
            with pool_runtime.patched_guard(native_pool, backend=pool_runtime.LIVE_BACKEND,
                                           stop_after_stats=True) as guard_class:
                with guard_class(self.root / "guard") as guard:
                    association.soft_pool_unpool(*stats_inputs())
        self.assertEqual(stop.exception.report["backend"], pool_runtime.LIVE_BACKEND)
        self.assertFalse(guard.report()["explicit_cpu_execution"])
        self.assertFalse(guard.report()["automatic_cpu_fallback"])

    def test_exact_packet_is_saved_before_original_comparison(self):
        original_compare = self.original["compare_results"]
        observations = []

        def compare(actual, expected):
            packet_path = self.root / "guard/packet_000001.pth"
            self.assertTrue(packet_path.is_file())
            packet = torch.load(packet_path, weights_only=True)
            for key in ("tokens", "mass", "reconstruction"):
                torch.testing.assert_close(packet["actual"][key], actual[key], atol=0, rtol=0)
                torch.testing.assert_close(packet["expected"][key], expected[key], atol=0, rtol=0)
            observations.append(True)
            return original_compare(actual, expected)

        with mock.patch.object(native_pool, "compare_results", side_effect=compare):
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard"):
                    association.soft_pool_unpool(*inputs())
        self.assertEqual(observations, [True])

    def test_packet_schema_has_exact_cpu_values_metadata_and_sha(self):
        with pool_runtime.patched_guard(native_pool, binding={"checkpoint": 1500}) as guard_class:
            with guard_class(self.root / "guard") as guard:
                guard.set_image("example")
                association.soft_pool_unpool(*inputs())
        record = guard.captured_packets[0]
        packet = torch.load(record["path"], weights_only=True)
        self.assertEqual(record["sha256"], pool_runtime._sha256(record["path"]))
        self.assertEqual(packet["format"], "LIVE_POOL_CAPTURE_V2")
        for where in (packet["snapshot"], packet["meta"]):
            self.assertEqual(where["backend"], pool_runtime.CPU_BACKEND)
            self.assertEqual(where["image"], "example")
            self.assertEqual(where["call"], 1)
        for part in ("actual", "expected"):
            self.assertTrue(all(value.device.type == "cpu" for value in packet[part].values()))
            self.assertTrue(all(value.dtype == torch.float32 for value in packet[part].values()))
        self.assertEqual(type(packet["meta"]["torch_version"]), str)
        self.assertEqual(packet["meta"]["binding"], {"checkpoint": 1500})
        self.assertGreater(packet["meta"]["original_num_threads"], 0)
        self.assertTrue(record["strict_cpu_gate_passed"])
        self.assertEqual(packet["snapshot"]["caller"]["function"],
                         "test_packet_schema_has_exact_cpu_values_metadata_and_sha")

    def test_strict_reference_failure_is_not_replaced_by_probe_success(self):
        with mock.patch.object(native_pool, "compare_results", return_value={"passed": False}):
            with self.assertRaisesRegex(ValueError, "reference gate failed"):
                with pool_runtime.patched_guard(native_pool, stop_after_stats=True) as guard_class:
                    with guard_class(self.root / "guard") as guard:
                        association.soft_pool_unpool(*stats_inputs())
        self.assertFalse(guard.captured_packets[0]["strict_cpu_gate_passed"])
        self.assertEqual(guard.tested_layouts, [])
        failure = json.loads((self.root / "guard/failure.json").read_text())
        self.assertEqual(failure["phase"], "cpu_reference_gate")
        self.assertEqual(failure["comparison"], {"passed": False})

    def test_invalid_ids_fail_before_cpu_kernel_without_clamping(self):
        invalid = list(inputs())
        invalid[2][0, 0] = 1234
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard") as guard:
                    association.soft_pool_unpool(*invalid)
        self.assertEqual(guard.cpu_arithmetic_calls, 0)
        self.assertEqual(guard.captured_packets, [])
        failure = json.loads((self.root / "guard/failure.json").read_text())
        saved = torch.load(failure["input"], weights_only=True)
        self.assertEqual(int(saved["ids"][0, 0]), 1234)

    def test_nonfinite_weights_rejected_before_execution(self):
        invalid = list(inputs())
        invalid[1][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard") as guard:
                    association.soft_pool_unpool(*invalid)
        self.assertEqual(guard.cpu_arithmetic_calls, 0)

    def test_fp64_execution_is_rejected_not_silently_downcast(self):
        invalid = list(inputs())
        invalid[0], invalid[1] = invalid[0].double(), invalid[1].double()
        with self.assertRaisesRegex(TypeError, "FP32 source"):
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard"):
                    association.soft_pool_unpool(*invalid)

    def test_snapshot_helper_rejects_fp64(self):
        source = native_pool.snapshot_inputs(*inputs())
        source["content"], source["weights"] = source["content"].double(), source["weights"].double()
        with self.assertRaisesRegex(TypeError, "FP32"):
            pool_runtime.cpu_pool_from_snapshot(source, "cpu", original_pool=self.original["_ORIGINAL_POOL"])

    def test_metadata_backend_and_flags_are_explicit(self):
        with pool_runtime.patched_guard(native_pool) as guard_class:
            with guard_class(self.root / "guard") as guard:
                association.soft_pool_unpool(*inputs())
        for name in ("manifest.json", "progress.json", "guard.json"):
            record = json.loads((self.root / "guard" / name).read_text())
            self.assertEqual(record["backend"], pool_runtime.CPU_BACKEND)
            self.assertFalse(record["automatic_cpu_fallback"])
            self.assertFalse(record["precision_intervention"])
        report = guard.report()
        self.assertTrue(report["execution_device_intervention"])
        self.assertTrue(report["slowdown_expected"])

    def test_first_reference_per_layout_and_optional_all_calls(self):
        with pool_runtime.patched_guard(native_pool) as guard_class:
            with guard_class(self.root / "first") as guard:
                association.soft_pool_unpool(*inputs())
                association.soft_pool_unpool(*inputs())
                association.soft_pool_unpool(*inputs(5))
                self.assertEqual(guard.reference_checks, 2)
                self.assertEqual(len(guard.captured_packets), 2)
                self.assertEqual(len(guard.tested_layouts), 2)
            with guard_class(self.root / "all", check_reference=True) as guard:
                association.soft_pool_unpool(*inputs())
                association.soft_pool_unpool(*inputs())
                self.assertEqual(guard.reference_checks, 2)
                self.assertEqual(len(guard.captured_packets), 2)

    def test_grouped_and_unknown_modes_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown explicit"):
            with pool_runtime.patched_guard(native_pool, backend="automatic"):
                pass
        with pool_runtime.patched_guard(native_pool) as guard_class:
            with self.assertRaisesRegex(ValueError, "not grouped"):
                guard_class(self.root / "guard", mode="grouped")

    def test_nested_scopes_and_active_guard_rejected(self):
        with pool_runtime.patched_guard(native_pool):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with pool_runtime.patched_guard(native_pool):
                    pass
        with self.original["PoolGuard"](self.root / "original"):
            with self.assertRaisesRegex(RuntimeError, "active guard"):
                with pool_runtime.patched_guard(native_pool):
                    pass

    def test_imported_aliases_are_restored(self):
        module = types.ModuleType("bda_benchmark.live_runtime_alias_test")
        module.pool = self.original["_ORIGINAL_POOL"]
        sys.modules[module.__name__] = module
        try:
            with pool_runtime.patched_guard(native_pool) as guard_class:
                with guard_class(self.root / "guard"):
                    self.assertIsNot(module.pool, self.original["_ORIGINAL_POOL"])
                    module.pool(*inputs())
            self.assertIs(module.pool, self.original["_ORIGINAL_POOL"])
        finally:
            sys.modules.pop(module.__name__, None)

    def test_scope_exception_restores_all_functions(self):
        with self.assertRaisesRegex(RuntimeError, "scope sentinel"):
            with pool_runtime.patched_guard(native_pool):
                raise RuntimeError("scope sentinel")

    def test_pool_failure_never_retries_or_queries_cuda_after_error(self):
        with mock.patch.object(pool_runtime, "_execution_state", return_value={"original_num_threads": 1}), \
                mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA accessed")):
            # A baseline comparison exception occurs after CPU output capture.
            with mock.patch.object(native_pool, "compare_results", side_effect=RuntimeError("CUDA poisoned sentinel")) as comparison:
                with self.assertRaisesRegex(RuntimeError, "poisoned sentinel"):
                    with pool_runtime.patched_guard(native_pool) as guard_class:
                        with guard_class(self.root / "guard") as guard:
                            association.soft_pool_unpool(*inputs())
                self.assertEqual(comparison.call_count, 1)
        self.assertEqual(guard.reference_checks, 0)
        self.assertEqual(len(guard.failures), 1)
        self.assertEqual(len(guard.captured_packets), 1)

    def test_kernel_exception_preserves_only_cpu_input_and_never_retries(self):
        observed = []

        def failed_original(content, probabilities, ids, count):
            observed.append(True)
            raise RuntimeError("CUDA device-side assert simulated at kernel")

        with mock.patch.object(native_pool, "_ORIGINAL_POOL", failed_original), \
                mock.patch.object(pool_runtime, "_execution_state", return_value={"original_num_threads": 1}), \
                mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA accessed")):
            with self.assertRaisesRegex(RuntimeError, "device-side assert simulated"):
                with pool_runtime.patched_guard(native_pool) as guard_class:
                    with guard_class(self.root / "guard") as guard:
                        guard.wrapper(*inputs())
        self.assertEqual(observed, [True])
        self.assertEqual(guard.captured_packets, [])
        self.assertEqual(len(guard.saved_inputs), 1)
        failure = json.loads((self.root / "guard/failure.json").read_text())
        self.assertEqual(failure["phase"], "pool_kernel")
        self.assertTrue(failure["restart_process_required"])
        snapshot = torch.load(failure["input"], weights_only=True)
        self.assertEqual(snapshot["content"].device.type, "cpu")

    def test_metadata_and_runtime_flags_are_not_mutated(self):
        before = pool_runtime._execution_state()
        with pool_runtime.patched_guard(native_pool) as guard_class:
            with guard_class(self.root / "guard"):
                association.soft_pool_unpool(*inputs())
        self.assertEqual(pool_runtime._execution_state(), before)


if __name__ == "__main__":
    unittest.main()
