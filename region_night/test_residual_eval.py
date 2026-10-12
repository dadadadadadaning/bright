"""CPU gates and real two-image prediction/artifact loop contracts."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import residual_eval
from residual_core import make_heads
from residual_eval import ARMS, BASE_ARMS, evaluate, formal_gate, initial_closeness, checks_passed


class GateContracts(unittest.TestCase):
    def test_formal_requires_both_reproductions_native_and_immutability(self):
        values = dict(image_count=450, reproduction={"passed": True},
                      e0_comparison={"passed": True}, immutable={"files": True, "runtime": True},
                      pool_ok=True, native_ok=True, initial_ok=True, q_ok=True, smoke=False)
        self.assertTrue(formal_gate(**values))
        for key, value in (("image_count", 449), ("reproduction", {"passed": False}),
                           ("e0_comparison", {"passed": False}), ("immutable", {}),
                           ("immutable", {"files": False}), ("pool_ok", False),
                           ("native_ok", False), ("initial_ok", False),
                           ("q_ok", False), ("smoke", True)):
            with self.subTest(key=key):
                self.assertFalse(formal_gate(**{**values, key: value}))

    def test_initial_tolerance_is_not_argmax_identity(self):
        base = torch.tensor([[[[.30]], [[.30]], [[.20]], [[.20]]]])
        changed = base.clone()
        changed[:, 0] -= 1e-6
        changed[:, 1] += 1e-6
        probabilities = dict(grid_probq=base, local_delta=changed, region_delta=changed)
        checks = initial_closeness(probabilities)
        self.assertTrue(checks_passed(checks))
        self.assertEqual(checks["region_delta"]["prediction_differences"], 1)
        probabilities["region_delta"] = base.roll(1, dims=1)
        self.assertFalse(checks_passed(initial_closeness(probabilities)))

    def test_historical_precision_is_still_absolute_1e5(self):
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report

        def metrics():
            return {arm: dict(valid_miou=.5, iou=[.5] * 4, precision=[.5] * 4,
                              recall=[.5] * 4, f1=[.5] * 4) for arm in BASE_ARMS}
        actual, expected = metrics(), metrics()
        self.assertTrue(reproduction_report(actual, expected, image_count=450)["passed"])
        actual["j3"]["recall"][2] += 1.1e-5
        report = reproduction_report(actual, expected, image_count=450)
        self.assertFalse(report["passed"])
        self.assertEqual(report["tolerance"], 1e-5)

    def test_source_hash_binds_all_external_executables_and_inherited_code(self):
        from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1 import core

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("residual_core.py", "residual_eval.py", "region_residual.py"):
                (root / name).write_text("# executable\n", encoding="utf-8")
            with patch.object(residual_eval, "__file__", str(root / "residual_eval.py")), \
                    patch.object(core, "source_hash", return_value="a" * 64):
                before = residual_eval.source_hash()
                (root / "residual_core.py").write_text("# changed\n", encoding="utf-8")
                self.assertNotEqual(before, residual_eval.source_hash())
                (root / "region_residual.py").unlink()
                with self.assertRaises(FileNotFoundError):
                    residual_eval.source_hash()


class SyntheticEvaluationContracts(unittest.TestCase):
    @staticmethod
    def fixture():
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
            resize_probability, unpool)

        names = [f"synthetic_{i:08d}" for i in range(450)]
        feature = torch.linspace(-.2, .2, 1024).reshape(1, 256, 2, 2)
        q = torch.ones(1, 9, 4) / 9
        ids = torch.tensor([[0, 0, 1, 1]] * 9, dtype=torch.long)
        tokens = torch.linspace(-.2, .2, 256).reshape(1, 2, 128)
        # Ignore255 belongs only to scoring; every prediction still has a class.
        target = torch.tensor([[[255, 1, 2], [1, 2, 3], [0, 3, 1]]])
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
        return bundle, heads, batches, feature

    @staticmethod
    def patched(stack, batches, feature):
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit
        from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import io_utils

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
        stack.enter_context(patch.object(residual_eval, "source_hash", return_value="synthetic_source"))

    def test_two_images_preserve_zero_delta_base_and_write_exact_artifacts(self):
        from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import assert_formal_report

        bundle, heads, batches, feature = self.fixture()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.patched(stack, batches, feature)
            output = Path(directory) / "evaluation"
            report = evaluate(bundle, heads, output, initial=True, smoke=True,
                              source_sha256="synthetic_source")
            self.assertTrue(report["complete"])
            self.assertFalse(report["full_validation_evidence"])
            self.assertIsNone(report["reproduction"]["passed"])
            self.assertTrue(report["initial_checks_passed"])
            self.assertTrue(report["q_checks_passed"])
            self.assertEqual(report["arms"], list(ARMS))
            self.assertEqual(report["image_count"], 2)
            self.assertTrue(heads.training)
            assert_formal_report(report, smoke=True)
            with self.assertRaises(ValueError):
                assert_formal_report(report, smoke=False)
            self.assertEqual(len(bundle["front"].base.clf_2._forward_hooks), 0)
            self.assertEqual(len(bundle["front"].base.clf_2._forward_pre_hooks), 0)
            journal = [json.loads(row) for row in (output / "images.jsonl").read_text().splitlines()]
            self.assertEqual([row["image"] for row in journal], bundle["val_names"][:2])
            self.assertEqual([row["index"] for row in journal], [0, 1])
            with np.load(output / journal[0]["prediction_file"], allow_pickle=False) as saved:
                self.assertEqual(set(saved), {"target", "support_j2", "support_j3"}
                                 | {"prediction_" + arm for arm in ARMS})
                self.assertEqual(int(saved["target"][0, 0, 0]), 255)
            self.assertIn("grid_probq__region_delta", report["transitions"])
            self.assertIn("local_delta__region_delta", report["transitions"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["image_count"], 2)

    def test_invalid_probability_stops_and_records_failure_without_repair(self):
        import residual_core
        from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.probability_diagnostics import DiagnosticProbabilityError

        bundle, heads, batches, feature = self.fixture()
        original_probability = residual_core.probability

        def bad_probability(arm, head, inputs, size):
            p = original_probability(arm, head, inputs, size)
            return p * 1.001 if arm == "region_delta" else p

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.patched(stack, batches, feature)
            stack.enter_context(patch.object(residual_core, "probability", side_effect=bad_probability))
            output = Path(directory) / "failed"
            with self.assertRaisesRegex(DiagnosticProbabilityError, "arm=region_delta"):
                evaluate(bundle, heads, output, initial=True, smoke=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["full_validation_evidence"])
            self.assertEqual(summary["image_count"], 0)
            self.assertEqual(summary["probability_failure"]["arm"], "region_delta")
            self.assertTrue((output / "probability_failure.pt").is_file())
            self.assertTrue(heads.training)
            self.assertEqual(len(bundle["front"].base.clf_2._forward_hooks), 0)

    def test_wrong_source_or_val_order_is_rejected_before_output(self):
        bundle, heads, batches, feature = self.fixture()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.patched(stack, batches, feature)
            output = Path(directory) / "rejected"
            with self.assertRaisesRegex(ValueError, "source hash"):
                evaluate(bundle, heads, output, initial=True, smoke=True, source_sha256="wrong")
            self.assertFalse(output.exists())
            bundle["val_names"] = list(reversed(bundle["val_names"]))
            with self.assertRaisesRegex(ValueError, "Val450"):
                evaluate(bundle, heads, output, initial=True, smoke=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
