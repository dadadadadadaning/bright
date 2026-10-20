"""CPU contracts for scoped NLL observation; not CUDA failure reproduction."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from nll_trace import NLLTrace, label_snapshot


def original_nll(probability, target, ignore_index=255):
    labels = target.long().reshape(-1).clone()
    valid = labels != ignore_index
    table = probability.permute(0, 2, 3, 1).reshape(-1, probability.shape[1]).contiguous()
    if not bool(valid.any()):
        return table.sum() * 0
    return F.nll_loss(table[valid].clamp_min(1e-8).log().contiguous(), labels[valid].contiguous())


class NLLTraceTests(unittest.TestCase):
    def setUp(self):
        self.deps = SimpleNamespace(region_nll=original_nll)
        self.events = []

    def emit(self, event, **fields):
        self.events.append(dict(event=event, **fields))

    def example(self):
        logits = torch.tensor(
            [[[[0.2, 0.8], [-0.4, 0.1]], [[-0.3, 0.2], [0.5, -0.2]],
              [[0.8, -0.1], [0.2, 0.7]], [[0.3, 0.2], [0.6, -0.8]]]],
            requires_grad=True,
        )
        return logits, torch.tensor([[[0, 255], [2, 3]]])

    def test_exact_original_objective_and_gradient_are_preserved(self):
        reference, target = self.example()
        expected = original_nll(reference.softmax(1), target)
        expected.backward()
        logits = reference.detach().clone().requires_grad_()
        with NLLTrace(torch, self.deps, self.emit) as trace:
            loss = self.deps.region_nll(logits.softmax(1), target)
            trace.snapshot("after_original_loss")
            with torch.no_grad():
                logged = self.deps.region_nll(logits.softmax(1), target)
            trace.snapshot("after_logging_nll")
            trace.snapshot("before_backward")
            self.assertEqual(len(trace.entries), 1)
            loss.backward()
        self.assertTrue(torch.equal(loss, expected))
        self.assertTrue(torch.equal(logged, expected))
        self.assertTrue(torch.equal(logits.grad, reference.grad))
        self.assertIs(self.deps.region_nll, original_nll)

    def test_captures_private_filtered_target_not_raw_alias(self):
        logits, target = self.example()
        with NLLTrace(torch, self.deps, self.emit) as trace:
            self.deps.region_nll(logits.softmax(1), target)
            entry = trace.entries[0]
            self.assertNotEqual(entry["target"].data_ptr(), target.data_ptr())
            self.assertEqual(entry["target"].tolist(), [0, 2, 3])
            self.assertEqual(entry["classes"], 4)
            self.assertEqual(entry["ignore_index"], -100)
            report = self.events[-1]
            self.assertEqual(report["target"]["shape"], [3])
            self.assertEqual(report["target"]["histogram"], [
                {"value": 0, "count": 1}, {"value": 2, "count": 1}, {"value": 3, "count": 1},
            ])
            self.assertEqual(report["invalid_count"], 0)

    def test_wrapper_is_restored_on_failure(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with NLLTrace(torch, self.deps, self.emit):
                raise RuntimeError("synthetic")
        self.assertIs(self.deps.region_nll, original_nll)

    def test_saved_target_mutation_is_detected_before_backward(self):
        for replacement in (1, 99):
            with self.subTest(replacement=replacement):
                logits, target = self.example()
                with NLLTrace(torch, self.deps, self.emit) as trace:
                    self.deps.region_nll(logits.softmax(1), target)
                    trace.entries[0]["target"][0] = replacement
                    with self.assertRaisesRegex(ValueError, "NLL saved target"):
                        trace.snapshot("after_logging_nll")
                self.assertTrue(self.events[-1]["target_changed"])
                self.assertTrue(self.events[-1]["version_changed"])
                self.assertIs(self.deps.region_nll, original_nll)

    def test_all_ignore_loss_has_no_nll_nodes(self):
        logits, target = self.example()
        target.fill_(255)
        with NLLTrace(torch, self.deps, self.emit) as trace:
            loss = self.deps.region_nll(logits.softmax(1), target)
            trace.snapshot("before_backward")
            loss.backward()
            self.assertEqual(trace.entries, [])
        self.assertEqual(self.events[-1]["count"], 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_saved_failure_uses_only_retained_cpu_copies(self):
        logits, target = self.example()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.pt"
            with NLLTrace(torch, self.deps, self.emit) as trace:
                self.deps.region_nll(logits.softmax(1), target)
                trace.snapshot("before_backward")
                # Saving must not snapshot again after the CUDA context fails.
                with mock.patch("nll_trace._cpu_copy", side_effect=AssertionError("device access")):
                    self.assertEqual(trace.save_failure(path), str(path))
                with self.assertRaises(FileExistsError):
                    trace.save_failure(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            case = saved["cases"][0]
            self.assertEqual(case["input"].device.type, "cpu")
            self.assertEqual(case["target"].device.type, "cpu")
            self.assertEqual(case["stage"], "before_backward")
            self.assertFalse(saved["exact_training_replay"])
            replay = case["input"].detach().requires_grad_()
            value = F.nll_loss(replay, case["target"], ignore_index=case["ignore_index"])
            value.backward()
            self.assertTrue(bool(torch.isfinite(replay.grad).all()))

    def test_label_snapshot_leaves_raw_labels_unchanged(self):
        target = torch.tensor([0, 1, 2, 255])
        copy = target.clone()
        before_version = target._version
        report = label_snapshot(torch, target)
        self.assertTrue(torch.equal(target, copy))
        self.assertEqual(target._version, before_version)
        self.assertEqual(report["minimum"], 0)
        self.assertEqual(report["maximum"], 255)
        self.assertEqual(len(report["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
