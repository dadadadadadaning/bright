"""Synthetic checks for equal data/budgets and head-only gradient ownership."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from .core import load_fit, make_heads, prepare_inputs
from .train import (SharedTrainStream, learning_rate_factor, matched_update,
                    run, validate_trainable_heads, validate_training_config)


def tiny_front():
    front = nn.Module()
    front.base = nn.Module()
    front.base.clf_2 = nn.Conv2d(256, 4, 1)
    front.region_classifier = nn.Linear(128, 4)
    return front.requires_grad_(False).eval()


def tiny_inputs():
    feature = torch.randn(1, 256, 2, 3, requires_grad=True)
    q = torch.ones(1, 9, 6, requires_grad=True) / 9
    ids = torch.tensor([[0, 0, 0, 1, 1, 1]] * 9)
    result = dict(q=q, ids=ids, count=2, shape=(2, 3),
                  tokens=torch.randn(1, 2, 128, requires_grad=True))
    return prepare_inputs(feature, result), feature, result


def optimizers(heads):
    return {arm: torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=5e-3)
            for arm, head in heads.items()}


class TrainingContractTests(unittest.TestCase):
    def test_formal_and_smoke_budgets_are_preregistered(self):
        self.assertEqual(validate_training_config(5000, 1e-4, False), 5000)
        self.assertEqual(validate_training_config(5000, 1e-4, True), 20)
        self.assertEqual(validate_training_config(20, 1e-4, True), 20)
        for steps, lr, smoke in ((4999, 1e-4, False), (30, 1e-4, True),
                                  (5000, 0, False), (5000, math.nan, False),
                                  (5000, math.inf, False), (True, 1e-4, False)):
            with self.subTest(steps=steps, lr=lr, smoke=smoke), self.assertRaises(ValueError):
                validate_training_config(steps, lr, smoke)

    def test_shared_stream_is_one_complete_shuffle_per_cycle(self):
        loader = [(None, None, None, None, [name]) for name in ("b", "a")]
        stream = SharedTrainStream(loader, ["a", "b"])
        self.assertEqual([(stream.next()[1:]) for _ in range(3)],
                         [("b", 1), ("a", 1), ("b", 2)])
        duplicate = SharedTrainStream([loader[0], loader[0]], ["a", "b"])
        duplicate.next()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            duplicate.next()
        missing = SharedTrainStream(loader[:1], ["a", "b"])
        missing.next()
        with self.assertRaisesRegex(ValueError, "skipped"):
            missing.next()

    def test_optimizer_cannot_touch_another_head_or_frozen_source(self):
        front = tiny_front()
        heads = make_heads(front)
        opts = optimizers(heads)
        validate_trainable_heads(heads, opts, [front])
        opts["p"] = torch.optim.AdamW(front.base.clf_2.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "private head"):
            validate_trainable_heads(heads, opts, [front])
        heads["r"] = heads["p"]
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_trainable_heads(heads, optimizers(heads), [front])

    def test_one_update_changes_all_heads_but_not_F_Q_or_source(self):
        torch.manual_seed(52)
        front = tiny_front()
        heads = make_heads(front)
        opts = optimizers(heads)
        inputs, original_feature, result = tiny_inputs()
        target = torch.tensor([[[0, 1, 2], [3, 2, 1]]])
        old_source = {k: v.clone() for k, v in front.state_dict().items()}
        old_head = {arm: {k: v.clone() for k, v in head.state_dict().items()}
                    for arm, head in heads.items()}
        logs = matched_update(heads, opts, inputs, target)
        self.assertEqual(set(logs), {"p", "r", "t"})
        self.assertIsNone(original_feature.grad)
        self.assertIsNone(result["tokens"].grad)
        self.assertTrue(all(not value.requires_grad for value in inputs.values()
                            if torch.is_tensor(value)))
        self.assertTrue(all(torch.equal(v, old_source[k]) for k, v in front.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in front.parameters()))
        for arm, head in heads.items():
            self.assertTrue(any(not torch.equal(v, old_head[arm][k])
                                for k, v in head.state_dict().items()))
            self.assertTrue(all(float(opts[arm].state[p]["step"]) == 1 for p in head.parameters()))

    def test_nonfinite_later_arm_prevents_every_optimizer_update(self):
        torch.manual_seed(71)
        heads = make_heads(tiny_front())
        opts = optimizers(heads)
        initial = {k: v.clone() for k, v in heads.state_dict().items()}
        inputs, _, _ = tiny_inputs()
        target = torch.tensor([[[0, 1, 2], [3, 2, 1]]])

        def probability(arm, head, data, size):
            p = head(data["feature"]).softmax(1)
            return p * float("nan") if arm == "r" else p

        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            matched_update(heads, opts, inputs, target, probability_fn=probability,
                           loss_fn=lambda p, y: F.nll_loss(p.log(), y))
        self.assertTrue(all(torch.equal(v, initial[k]) for k, v in heads.state_dict().items()))
        self.assertTrue(all(not optimizer.state for optimizer in opts.values()))

    def test_grad_enabled_inputs_are_rejected_before_update(self):
        heads = make_heads(tiny_front())
        opts = optimizers(heads)
        inputs, _, _ = tiny_inputs()
        inputs["q"] = inputs["q"].requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached"):
            matched_update(heads, opts, inputs, torch.zeros(1, 2, 3, dtype=torch.long))
        self.assertTrue(all(not opt.state for opt in opts.values()))

    def test_identical_schedule_for_every_arm(self):
        self.assertAlmostEqual(learning_rate_factor(1, 5000), 1 / 500)
        self.assertEqual(learning_rate_factor(500, 5000), 1)
        self.assertEqual(learning_rate_factor(5000, 5000), 0)
        with self.assertRaises(ValueError):
            learning_rate_factor(0, 5000)

    def test_complete_mock_smoke_fit_round_trips_and_remains_nonformal(self):
        """Exercise real losses/updates/artifacts with fake image I/O and audit."""
        from . import core, evaluate
        from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import io_utils
        from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1 import native_pool
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit

        torch.manual_seed(133)
        front = tiny_front()
        source_state = {name: value.clone() for name, value in front.state_dict().items()}
        names = ["event-a_00000001", "event-b_00000002"]
        target = torch.tensor([[[0, 1, 2], [3, 2, 1]]])
        loader = [(torch.randn(1, 3, 2, 3), torch.randn(1, 3, 2, 3),
                   torch.ones(1, 2, 3, dtype=torch.long), target, [name]) for name in names]
        feature = torch.randn(1, 256, 2, 3)
        result = dict(q=torch.ones(1, 9, 6) / 9,
                      ids=torch.tensor([[0, 0, 0, 1, 1, 1]] * 9),
                      count=2, shape=(2, 3), tokens=torch.randn(1, 2, 128))
        captures = []

        class FakeCapture:
            def __init__(self, module):
                self.module, self.closed, self.calls = module, False, 0
                captures.append(self)

            def __call__(self, pre, post):
                self.calls += 1
                return result, feature

            def close(self):
                self.closed = True

        class FakePoolGuard:
            def __init__(self, output, **kwargs):
                self.output, self.failures, self.restored = Path(output), [], False

            def __enter__(self):
                self.output.mkdir()
                return self

            def __exit__(self, *args):
                self.restored = True

            def set_image(self, image):
                self.image = image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_list, val_list = root / "train.txt", root / "val.txt"
            train_list.write_text("\n".join(names), encoding="utf-8")
            val_list.write_text("validation_1\nvalidation_2", encoding="utf-8")
            args = argparse.Namespace(dataset="mock-only", workers=0, seed=3407,
                                      pool_mode="native", train_list=str(train_list),
                                      val_list=str(val_list))
            bundle = dict(args=args, device=torch.device("cpu"), front=front,
                          metric=nn.Linear(2, 2).requires_grad_(False).eval(),
                          decoders={"j1": nn.Linear(2, 2).requires_grad_(False).eval()},
                          train_names=names, val_names=["validation_1", "validation_2"],
                          protocol_sha256="mock-passed-E0")
            fit = root / "fit"
            cli = argparse.Namespace(protocol="mock-protocol", output=str(fit),
                                     steps=5000, lr=1e-4, smoke=True)
            eval_calls = []

            def mock_evaluate(actual_bundle, heads, output, limit, initial):
                self.assertIs(actual_bundle, bundle)
                self.assertEqual(limit, 2)
                self.assertTrue((fit / "manifest.json").is_file())
                output = Path(output)
                output.mkdir()
                state = {k: v.clone() for k, v in heads.state_dict().items()}
                eval_calls.append(dict(initial=initial, path=output.name, heads=state))
                if not initial:
                    unfinished = torch.load(fit / "heads_last.pth", weights_only=False)
                    self.assertIs(unfinished["complete"], False)
                    self.assertEqual(unfinished["completed_steps"], 20)
                report = dict(complete=True, full_validation_evidence=False,
                              image_count=2, reproduction={"passed": None},
                              immutable={"frozen_models": True}, native_a1_replay_ok=True)
                io_utils.write_json(output / "result.json", report)
                return report

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(core, "load_source", return_value=bundle))
                stack.enter_context(mock.patch.object(evaluate, "evaluate_heads", side_effect=mock_evaluate))
                immutable_check = stack.enter_context(mock.patch.object(
                    io_utils, "verify_unchanged", return_value={"frozen_models": True}))
                loader_call = stack.enter_context(mock.patch.object(
                    memory_audit, "full_image_loader", return_value=loader))
                stack.enter_context(mock.patch.object(memory_audit, "FrozenFrontCapture", FakeCapture))
                stack.enter_context(mock.patch.object(native_pool, "PoolGuard", FakePoolGuard))
                stack.enter_context(redirect_stdout(io.StringIO()))
                summary = run(cli)
            self.assertEqual([x["initial"] for x in eval_calls], [True, False])
            self.assertEqual([x["path"] for x in eval_calls], ["initial_val", "final_val"])
            loader_call.assert_called_once_with("mock-only", names, True, 0, 3424)
            self.assertEqual(immutable_check.call_count, 2)
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["full_validation_evidence"])
            self.assertEqual(summary["updates_per_arm"], {"p": 20, "r": 20, "t": 20})
            self.assertEqual(captures[0].calls, 20)
            self.assertTrue(captures[0].closed)
            self.assertTrue(all(torch.equal(value, source_state[name])
                                for name, value in front.state_dict().items()))
            lines = [json.loads(line) for line in (fit / "training.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["image"] for row in lines], names * 10)
            self.assertEqual(len(lines), 20)
            manifest = json.loads((fit / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_list"]["sha256"], io_utils.digest(train_list))
            self.assertEqual(manifest["steps_per_arm"], summary["updates_per_arm"])
            self.assertEqual(summary["manifest_sha256"], io_utils.digest(fit / "manifest.json"))
            restored, metadata = load_fit(fit, bundle, allow_smoke=True)
            self.assertTrue(metadata["smoke"])
            self.assertEqual(metadata["steps"], 20)
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, eval_calls[-1]["heads"][name]))
            checkpoint = torch.load(fit / "heads_last.pth", weights_only=False)
            self.assertIs(checkpoint["complete"], True)
            self.assertTrue(all(float(state["step"]) == 20
                                for optimizer in checkpoint["optimizer_states"].values()
                                for state in optimizer["state"].values()))
            with self.assertRaisesRegex(ValueError, "smoke"):
                load_fit(fit, bundle)


if __name__ == "__main__":
    unittest.main()
