"""CPU checks using the actual addon update_arms, without repository data."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "slot_memory_ablation_v1"))
sys.path.append(str(ROOT / "slot_memory_failure_probe_v1"))

import torch
from torch.nn import functional as F

from ablation_runner import update_arms
from observer import UpdateObserver


def nll(probability, target):
    labels = target.reshape(-1).clone()
    valid = labels != 255
    table = probability.permute(0, 2, 3, 1).reshape(-1, 4)
    return F.nll_loss(table[valid].clamp_min(1e-8).log(), labels[valid])


class Runtime:
    def __init__(self, fail=False):
        self.deps = SimpleNamespace(region_nll=nll)
        self.fail = fail

    def loss(self, probability, target):
        loss = self.deps.region_nll(probability, target) + 0.01 * probability.square().mean()
        return FailBackward.apply(loss) if self.fail else loss


class FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("synthetic backward failure")


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, features):
        values = self.dropout(self.query_projection(features))
        return values.T.reshape(1, 4, 2, 2).softmax(1)


class Factory:
    def prepare(self, example, arms):
        return {arm: None for arm in arms}

    def forward(self, head, example, arm, support):
        return {"probabilities": head(example.features)}


class State:
    @staticmethod
    def capture_rng():
        return {"torch": torch.get_rng_state(), "python": random.getstate()}

    @staticmethod
    def restore_rng(state):
        torch.set_rng_state(state["torch"])
        random.setstate(state["python"])

    @staticmethod
    def state_fingerprint(state):
        return hashlib.sha256(
            state["torch"].numpy().tobytes() + repr(state["python"]).encode()
        ).hexdigest()

    @staticmethod
    def lr_at_step(step, **kwargs):
        return 1e-3


def fixture(fail=False):
    torch.manual_seed(1729)
    random.seed(1729)
    prototype = Head()
    heads = {arm: copy.deepcopy(prototype) for arm in ("e", "f")}
    optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=1e-3)
                  for arm, head in heads.items()}
    example = SimpleNamespace(
        features=torch.randn(4, 3), target=torch.tensor([[[0, 1], [2, 255]]]),
        image="synthetic_only",
    )
    return heads, optimizers, example, Runtime(fail=fail), Factory()


class UpdateObserverTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def run_updates(self, dense=None):
        heads, optimizers, example, runtime, factory = fixture()
        rows, events = [], []
        observer = None
        if dense is not None:
            observer = UpdateObserver(torch, runtime, factory, heads, optimizers,
                                      lambda event, **fields: events.append((event, fields)), dense=dense)
            observer.__enter__()
        try:
            for step in (1, 2, 3):
                if observer is not None:
                    observer.set_step(step)
                rows.append(update_arms(
                    heads, optimizers, example, runtime, factory, State,
                    step=step, multiplier=0.2,
                ))
                if observer is not None:
                    self.assertEqual(observer.completed_arms, ["e", "f"])
        finally:
            if observer is not None:
                observer.__exit__(None, None, None)
        result = dict(
            rows=rows, heads={arm: head.state_dict() for arm, head in heads.items()},
            gradients={arm: {name: p.grad for name, p in head.named_parameters()}
                       for arm, head in heads.items()},
            optimizers={arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
            rng=State.capture_rng(),
        )
        self.assertNotIn("forward", factory.__dict__)
        self.assertNotIn("loss", runtime.__dict__)
        for optimizer in optimizers.values():
            self.assertFalse(optimizer._optimizer_step_pre_hooks)
            self.assertFalse(optimizer._optimizer_step_post_hooks)
        return result, observer, events

    def test_prefix_and_dense_preserve_original_updates_gradients_and_rng(self):
        expected, _, _ = self.run_updates()
        for dense in (False, True):
            with self.subTest(dense=dense):
                result, observer, events = self.run_updates(dense=dense)
                self.assert_nested_equal(result, expected)
                self.assertIsNone(observer.failed_trace)
                self.assertEqual(observer.last_marker["phase"], "optimizer_step_end")
                self.assertEqual(observer.last_marker["step"], 3)
                self.assertEqual(observer.last_marker["arm"], "f")
                self.assertTrue(any(e[1].get("phase") == "backward_begin" for e in events))
                self.assertEqual(any(e[0] == "nll_saved_target" for e in events), dense)

    def test_prefix_does_not_instantiate_dense_tracer(self):
        with mock.patch("observer.NLLTrace", side_effect=AssertionError("dense access")):
            self.run_updates(dense=False)

    def test_backward_failure_restores_hooks_and_retains_cpu_case_after_exit(self):
        heads, optimizers, example, runtime, factory = fixture(fail=True)
        events = []
        observer = UpdateObserver(torch, runtime, factory, heads, optimizers,
                                  lambda event, **fields: events.append((event, fields)), dense=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic backward failure"):
            with observer:
                observer.set_step(1)
                update_arms(heads, optimizers, example, runtime, factory, State,
                            step=1, multiplier=0.2)
        self.assertEqual(observer.last_marker["phase"], "backward_begin")
        self.assertEqual(observer.last_marker["arm"], "e")
        self.assertEqual(observer.completed_arms, [])
        self.assertIsNotNone(observer.failed_trace)
        self.assertIs(runtime.deps.region_nll, nll)
        self.assertNotIn("forward", factory.__dict__)
        self.assertNotIn("loss", runtime.__dict__)
        for optimizer in optimizers.values():
            self.assertFalse(optimizer._optimizer_step_pre_hooks)
            self.assertFalse(optimizer._optimizer_step_post_hooks)
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "case.pt"
            with mock.patch("nll_trace._cpu_copy", side_effect=AssertionError("device access")):
                observer.failed_trace.save_failure(case)
            saved = torch.load(case, map_location="cpu", weights_only=False)
            self.assertEqual(saved["cases"][0]["stage"], "before_backward")
            self.assertEqual(saved["cases"][0]["target"].device.type, "cpu")

    def test_second_arm_failure_tracks_only_completed_optimizer_and_active_trace(self):
        heads, optimizers, example, runtime, factory = fixture()
        original_loss = runtime.loss
        calls = []

        def loss(probability, target):
            result = original_loss(probability, target)
            calls.append(1)
            return FailBackward.apply(result) if len(calls) == 2 else result

        runtime.loss = loss
        observer = UpdateObserver(torch, runtime, factory, heads, optimizers,
                                  lambda *a, **k: None, dense=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic backward failure"):
            with observer:
                observer.set_step(1)
                update_arms(heads, optimizers, example, runtime, factory, State,
                            step=1, multiplier=0.2)
        self.assertEqual(observer.completed_arms, ["e"])
        self.assertEqual(observer.last_marker["arm"], "f")
        self.assertEqual(observer.last_marker["phase"], "backward_begin")
        self.assertEqual(len(observer.failed_trace.entries), 1)
        self.assertTrue(optimizers["e"].state)
        self.assertFalse(optimizers["f"].state)
        self.assertIs(runtime.loss, loss)
        self.assertIs(runtime.deps.region_nll, nll)

    def test_enter_failure_removes_partially_installed_hooks_and_wrappers(self):
        heads, optimizers, _, runtime, factory = fixture()
        observer = UpdateObserver(torch, runtime, factory, heads, optimizers, lambda *a, **k: None)
        with mock.patch.object(optimizers["f"], "register_step_post_hook",
                               side_effect=RuntimeError("synthetic install failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic install failure"):
                observer.__enter__()
        self.assertNotIn("forward", factory.__dict__)
        self.assertNotIn("loss", runtime.__dict__)
        for optimizer in optimizers.values():
            self.assertFalse(optimizer._optimizer_step_pre_hooks)
            self.assertFalse(optimizer._optimizer_step_post_hooks)


if __name__ == "__main__":
    unittest.main()
