"""CPU contracts for route observation; original validator owns acceptance."""
from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import torch

from route_trace import RouteTrace


def bindings(shared=False):
    calls, failure = [], ValueError("original route rejection")
    slots = {"torch": torch, "calls": calls, "failure": failure}
    exec(compile("""
def validate_routes(q, ids, count, batch, pixels):
    calls.append((q, ids))
    if (not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=2e-5, rtol=2e-5)):
        raise failure
    return ids

def sparse_unpool(values, q, ids):
    return validate_routes(q, ids, values.shape[1], q.shape[0], q.shape[-1])
""", "synthetic_slots.py", "exec"), slots)
    model = slots if shared else {
        "torch": torch, "validate_routes": slots["validate_routes"],
        "sparse_unpool": slots["sparse_unpool"],
    }
    exec(compile("""
class Classifier:
    def forward(self, q, ids):
        parent = validate_routes(q, ids, 5, q.shape[0], q.shape[-1])
        child = sparse_unpool(torch.ones(1, 5, 1), q, ids)
        return parent, child
""", "synthetic_model.py", "exec"), model)
    return model["Classifier"], model, slots, calls, failure


class RouteTraceTests(unittest.TestCase):
    def setUp(self):
        self.q = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]], dtype=torch.float32)
        self.ids = torch.tensor([[[0, 1], [2, 3]]])
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def test_distinct_global_aliases_preserve_exact_return_and_restore(self):
        cls, model, slots, calls, _ = bindings()
        original = model["validate_routes"]
        with RouteTrace(torch, cls, self.emit) as trace:
            self.assertIsNot(model["validate_routes"], original)
            self.assertIsNot(slots["validate_routes"], original)
            self.assertIsNot(model["validate_routes"], slots["validate_routes"])
            result = cls().forward(self.q, self.ids)
            self.assertIs(result[0], self.ids)
            self.assertIs(result[1], self.ids)
            self.assertEqual(len(calls), 2)
        self.assertIs(model["validate_routes"], original)
        self.assertIs(slots["validate_routes"], original)
        summary = trace.summary()
        self.assertEqual(summary["validation_call_count"], 2)
        self.assertEqual(summary["diagnostic_error_count"], 0)
        self.assertEqual(summary["validators"]["classifier"]["latest"]["callsite"]["path"],
                         "synthetic_model.py")
        self.assertEqual(summary["validators"]["sparse_unpool"]["latest"]["callsite"]["path"],
                         "synthetic_slots.py")

    def test_shared_globals_are_patched_once(self):
        cls, model, slots, calls, _ = bindings(shared=True)
        with RouteTrace(torch, cls, self.emit) as trace:
            cls().forward(self.q, self.ids)
        self.assertEqual(len(trace.summary()["bindings"]), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(trace.summary()["validation_call_count"], 2)

    def test_original_exception_identity_and_cached_illegal_weight_diagnosis(self):
        cls, model, slots, calls, failure = bindings()
        original = model["validate_routes"]
        bad = torch.tensor([[[-0.1, 0.5], [1.1, 0.8]]])
        with self.assertRaises(ValueError) as caught:
            with RouteTrace(torch, cls, self.emit) as trace:
                trace.set_context("after_optimizer", arm="d")
                cls().forward(bad, self.ids)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(calls), 1)
        self.assertIs(model["validate_routes"], original)
        self.assertIs(slots["validate_routes"], original)
        report = trace.summary()["validators"]["classifier"]["first_failure"]
        self.assertEqual(report["weights"]["negative_count"], 1)
        self.assertEqual(report["weights"]["nonfinite_count"], 0)
        self.assertEqual(report["cpu_float32_sums"]["bad_count"], 1)
        self.assertEqual(report["cpu_float64_sums"]["bad_count"], 1)
        self.assertEqual(report["bad_pixel_count"], 2)
        self.assertEqual(report["first_bad_pixels"][0]["flat_pixel"], 0)
        self.assertEqual(report["context_stage"], "after_optimizer")
        self.assertEqual(report["arm"], "d")
        self.assertEqual(trace.summary()["validator_failure_count"], 1)
        self.assertEqual(trace.summary()["first_validation_failure"], report)

    def test_snapshots_leave_inputs_and_rng_unchanged(self):
        cls, _, _, _, _ = bindings()
        original_q, original_ids = self.q.clone(), self.ids.clone()
        versions = (self.q._version, self.ids._version)
        torch.manual_seed(719)
        random.seed(719)
        torch_rng, python_rng = torch.get_rng_state(), random.getstate()
        with RouteTrace(torch, cls, self.emit) as trace:
            trace.watch("parent", self.q, self.ids)
            cls().forward(self.q, self.ids)
            trace.snapshot("after_forward", arm="e")
        self.assertTrue(torch.equal(self.q, original_q))
        self.assertTrue(torch.equal(self.ids, original_ids))
        self.assertEqual((self.q._version, self.ids._version), versions)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(trace.summary()["watched_change_count"], 0)

    def test_raw_data_mutation_is_found_without_version_change(self):
        cls, _, _, _, _ = bindings()
        with RouteTrace(torch, cls, self.emit) as trace:
            trace.watch("soft", self.q)
            initial_version = self.q._version
            self.q.data[0, 0, 0] = 0.25
            self.assertEqual(self.q._version, initial_version)
            trace.snapshot("after_first_mutation", arm="b")
            trace.snapshot("unchanged_again", arm="b")
            self.q.data[0, 0, 0] = 0.28
            trace.snapshot("after_second_mutation", arm="d")
        report = trace.summary()
        self.assertEqual(report["watched_change_count"], 2)
        self.assertEqual(report["first_watch_change"]["stage"], "after_first_mutation")
        watched = report["watched"]["soft"]
        self.assertEqual(watched["initial"]["weights"]["version"], watched["latest"]["weights"]["version"])
        self.assertNotEqual(watched["initial"]["weights"]["cpu_sha256"],
                            watched["latest"]["weights"]["cpu_sha256"])
        self.assertEqual(watched["first_change"]["stage"], "after_first_mutation")

    def test_diagnostic_errors_do_not_mask_original_result_or_exception(self):
        for bad in (False, True):
            with self.subTest(bad=bad):
                cls, _, _, calls, failure = bindings()
                q = -self.q if bad else self.q
                with RouteTrace(torch, cls, lambda *a, **k: (_ for _ in ()).throw(
                    RuntimeError("synthetic emit failure"))) as trace:
                    with mock.patch.object(trace, "_capture_impl",
                                           side_effect=RuntimeError("synthetic capture failure")):
                        if bad:
                            with self.assertRaises(ValueError) as caught:
                                cls().forward(q, self.ids)
                            self.assertIs(caught.exception, failure)
                        else:
                            output = cls().forward(q, self.ids)
                            self.assertIs(output[0], self.ids)
                            self.assertIs(output[1], self.ids)
                self.assertEqual(len(calls), 1 if bad else 2)
                self.assertGreater(trace.summary()["diagnostic_error_count"], 0)

    def test_save_after_exit_uses_only_owned_cpu_snapshots(self):
        cls, _, _, _, _ = bindings()
        with RouteTrace(torch, cls, self.emit) as trace:
            trace.watch("parent", self.q, self.ids)
            cls().forward(self.q, self.ids)
            self.q.data[0, 0, 0] = 0.25
            trace.snapshot("changed")
        self.q.data.fill_(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.pt"
            with mock.patch.object(trace, "_cpu_copy", side_effect=AssertionError("device read")):
                trace.save_cpu_case(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            entry = saved["watched"]["parent"]
            self.assertAlmostEqual(float(entry["initial_case"]["q"][0, 0, 0]), 0.2)
            self.assertAlmostEqual(float(entry["latest_case"]["q"][0, 0, 0]), 0.25)
            self.assertEqual(entry["latest_case"]["q"].device.type, "cpu")
            self.assertEqual(saved["validators"]["classifier"]["latest_case"]["ids"].device.type, "cpu")
            json.dumps(saved["summary"], allow_nan=False)

    def test_nonfinite_route_statistics_remain_json_safe(self):
        cls, _, _, _, failure = bindings()
        q = self.q.clone()
        q[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError) as caught:
            with RouteTrace(torch, cls, self.emit) as trace:
                cls().forward(q, self.ids)
        self.assertIs(caught.exception, failure)
        report = trace.summary()["validators"]["classifier"]["first_failure"]
        self.assertEqual(report["weights"]["nonfinite_count"], 1)
        self.assertEqual(report["cpu_float64_sums"]["bad_count"], 1)
        self.assertEqual(trace.summary()["diagnostic_error_count"], 0)
        json.dumps(trace.summary(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
