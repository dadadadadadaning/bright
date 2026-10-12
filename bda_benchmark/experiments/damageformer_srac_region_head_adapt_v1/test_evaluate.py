"""CPU contracts for head-evaluation gates and captured native replay."""
from __future__ import annotations

import unittest
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from .evaluate import (BASE_ARMS, INITIAL_PAIRS, _checks_passed,
                       formal_gate, initial_closeness)
from .native import NativeA1LogitsCapture, native_replay
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report


class EvaluationContracts(unittest.TestCase):
    def test_full_formal_requires_reproduction_immutability_native_and_initial(self):
        values = dict(image_count=450, reproduction={"passed": True},
                      immutable={"files": True, "runtime": True}, pool_ok=True,
                      native_a1_replay_ok=True, initial_ok=True, smoke=False)
        self.assertTrue(formal_gate(**values))
        for name, value in (("image_count", 449), ("reproduction", {"passed": False}),
                            ("immutable", {}), ("immutable", {"files": False}),
                            ("native_a1_replay_ok", False), ("pool_ok", False),
                            ("initial_ok", False), ("smoke", True)):
            with self.subTest(name=name, value=value):
                self.assertFalse(formal_gate(**{**values, name: value}))

    def test_initial_probability_tolerance_does_not_require_identical_argmax(self):
        base = torch.tensor([[[[0.30]], [[0.30]], [[0.20]], [[0.20]]]])
        changed = base.clone()
        changed[:, 0] -= 1e-6
        changed[:, 1] += 1e-6
        probabilities = {}
        for arm, baseline in INITIAL_PAIRS.items():
            probabilities[baseline] = base
            probabilities[arm] = changed
        checks = initial_closeness(probabilities)
        self.assertTrue(_checks_passed(checks))
        self.assertEqual(checks["p"]["prediction_differences"], 1)
        probabilities["r"] = base.roll(1, dims=1)
        self.assertFalse(_checks_passed(initial_closeness(probabilities)))

    def test_initial_parameter_map_cannot_be_empty_or_contain_failed_check(self):
        self.assertFalse(_checks_passed({}))
        self.assertTrue(_checks_passed({"p": True, "r": True}))
        self.assertFalse(_checks_passed({"p": True, "r": False}))
        self.assertFalse(_checks_passed({"p": {"passed": None}}))

    def test_native_capture_requires_exactly_one_call_and_removes_hook(self):
        layer = nn.Conv2d(3, 4, 1)
        front = SimpleNamespace(base=SimpleNamespace(clf_2=layer))
        capture = NativeA1LogitsCapture(front)
        value = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        try:
            with self.assertRaisesRegex(RuntimeError, "got 0"):
                capture.take()
            expected = layer(value)
            self.assertTrue(torch.equal(capture.take(), expected))
            # take resets its window; a second take cannot reuse stale logits.
            with self.assertRaisesRegex(RuntimeError, "got 0"):
                capture.take()
            layer(value)
            layer(value + 1)
            with self.assertRaisesRegex(RuntimeError, "got 2"):
                capture.take()
            capture.clear()
            expected = layer(value + 2)
            self.assertTrue(torch.equal(capture.take(), expected))
        finally:
            capture.close()
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_native_identity_uses_captured_logits_even_if_extra_classifier_changes(self):
        logits = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 7
        original = F.interpolate(logits, (5, 5), mode="bilinear", align_corners=False)
        self.assertTrue(native_replay(logits, original, (5, 5))["passed"])
        extra_logits = logits.clone()
        extra_logits[:, 2] += 0.1
        self.assertFalse(native_replay(extra_logits, original, (5, 5))["passed"])

    def test_historical_gate_keeps_all_five_paths_and_class_metrics_at_1e5(self):
        def metrics():
            return {arm: {"valid_miou": 0.5, "iou": [0.5] * 4,
                          "precision": [0.5] * 4, "recall": [0.5] * 4,
                          "f1": [0.5] * 4} for arm in BASE_ARMS}
        actual, expected = metrics(), metrics()
        report = reproduction_report(actual, expected, image_count=450)
        self.assertTrue(report["passed"])
        self.assertEqual(report["tolerance"], 1e-5)
        actual["j3"]["recall"][2] += 1.1e-5
        self.assertFalse(reproduction_report(actual, expected, image_count=450)["passed"])
        actual = metrics()
        del actual["j2"]
        with self.assertRaisesRegex(ValueError, "j2"):
            reproduction_report(actual, expected, image_count=450)

    def test_two_synthetic_images_write_checked_artifacts_and_restore_head_modes(self):
        """Exercise the evaluation loop without a checkpoint or a BRIGHT loader."""
        from .core import make_heads
        from .evaluate import evaluate_heads
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
            resize_probability, unpool)
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit
        from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import io_utils

        names = [f"synthetic_{i:08d}" for i in range(450)]
        feature = torch.linspace(-0.2, 0.2, 1024).reshape(1, 256, 2, 2)
        q = torch.ones(1, 9, 4) / 9
        ids = torch.tensor([[0, 0, 1, 1]] * 9, dtype=torch.long)
        tokens = torch.linspace(-0.2, 0.2, 256).reshape(1, 2, 128)
        target = torch.tensor([[[0, 1, 2], [1, 2, 3], [0, 3, 1]]])
        batches = [(torch.zeros(1, 3, 3, 3), torch.zeros(1, 1, 3, 3), None,
                    target, [name]) for name in names[:2]]

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(256, 4, 1)
                self.region_classifier = nn.Linear(128, 4)

            def forward(self, pre, post):
                logits = self.base.clf_2(feature)
                old = F.interpolate(logits, (3, 3), mode="bilinear", align_corners=False)
                prob = resize_probability(unpool(self.region_classifier(tokens).softmax(-1),
                                                 q, ids, (2, 2)), (3, 3))
                return dict(old_pixel=old, prob=prob, q=q, ids=ids,
                            count=2, shape=(2, 2), tokens=tokens)

        front = Front().eval()
        heads = make_heads(front)
        heads.train()

        bundle = dict(front=front, args=SimpleNamespace(dataset="unused", workers=0,
                      seed=3407, pool_mode="native"), val_names=names,
                      protocol=dict(selected_images=names, metrics={}, profile="legacy", pool_mode="native"),
                      topk=8, bank={"event_names": ["synthetic"]}, metric=None,
                      decoders={arm: None for arm in ("j1", "j2", "j3")},
                      reference={}, device=torch.device("cpu"), protocol_path="synthetic_protocol.json",
                      protocol_sha256="synthetic")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            replacements = dict(
                full_image_loader=lambda _dataset, selected, *a, **k: [
                    batch for batch in batches if batch[-1][0] in selected],
                parse_batch=lambda batch, device: batch,
                select_ids=lambda *a, **k: torch.arange(8).repeat(2, 1),
                validate_selected=lambda *a, **k: None,
                gather_support=lambda *a, **k: None,
                decoder_forward=lambda *a, **k: {"logits": feature[:, :4]},
            )
            for name, value in replacements.items():
                stack.enter_context(patch.object(memory_audit, name, value))
            stack.enter_context(patch.object(io_utils, "verify_unchanged", lambda bundle: {"synthetic": True}))
            output = Path(directory) / "evaluation"
            report = evaluate_heads(bundle, heads, output=output, limit=2, initial=True)
            self.assertTrue(report["complete"])
            self.assertFalse(report["full_validation_evidence"])
            self.assertIsNone(report["reproduction"]["passed"])
            self.assertTrue(report["initial_checks_passed"])
            self.assertTrue(heads.training)
            self.assertEqual(len(front.base.clf_2._forward_hooks), 0)
            self.assertEqual(len(front.base.clf_2._forward_pre_hooks), 0)
            journal = [json.loads(row) for row in (output / "images.jsonl").read_text().splitlines()]
            self.assertEqual([row["image"] for row in journal], names[:2])
            self.assertEqual(len(list((output / "predictions").glob("*.npz"))), 2)
            self.assertIn("region_mean__direct_region", report["transitions"])
            # A single selected image uses the real loop but stays nonformal.
            one = evaluate_heads(bundle, heads, output=Path(directory) / "single",
                                 image=names[1], initial=True)
            self.assertEqual(one["diagnostic_image"], names[1])
            self.assertEqual(one["image_count"], 1)
            self.assertFalse(one["full_validation_evidence"])
            from . import core
            from .probability_diagnostics import DiagnosticProbabilityError
            original_probability = core.head_probability

            def bad_r(arm, head, inputs, size):
                value = original_probability(arm, head, inputs, size)
                return value * 1.001 if arm == "r" else value

            failed = Path(directory) / "failed"
            with patch.object(core, "head_probability", side_effect=bad_r):
                with self.assertRaisesRegex(DiagnosticProbabilityError, "arm=r"):
                    evaluate_heads(bundle, heads, output=failed, image=names[1], initial=True)
            summary = json.loads((failed / "summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["image"], names[1])
            self.assertEqual(summary["probability_failure"]["arm"], "r")
            self.assertEqual(summary["image_count"], 0)
            self.assertTrue((failed / "probability_failure.pt").is_file())
            self.assertTrue(heads.training)
            self.assertEqual(len(front.base.clf_2._forward_hooks), 0)
            self.assertEqual(len(front.base.clf_2._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
