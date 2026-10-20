"""CPU contracts for exact AdamW restoration; no BRIGHT data or CUDA needed."""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import random
import unittest
from unittest import mock

import torch

from restore_state import restore_training_state, validate_restored_rng


class CPUState:
    """Small independent adapter with the production state helper's interface."""
    def __init__(self):
        self.restore_calls = 0

    def state_fingerprint(self, value):
        def canonical(obj):
            if isinstance(obj, torch.Tensor):
                data = obj.detach().cpu().contiguous()
                return ["tensor", str(data.dtype), list(data.shape),
                        data.reshape(-1).view(torch.uint8).numpy().tobytes().hex()]
            if isinstance(obj, dict):
                return ["dict", [[canonical(k), canonical(obj[k])] for k in
                                 sorted(obj, key=lambda k: (type(k).__name__, repr(k)))]]
            if isinstance(obj, (tuple, list)):
                return [type(obj).__name__, [canonical(v) for v in obj]]
            if obj is None or isinstance(obj, (bool, str, int, float)):
                return [type(obj).__name__, repr(obj)]
            raise TypeError(type(obj).__name__)
        return hashlib.sha256(json.dumps(canonical(value)).encode()).hexdigest()

    def capture_rng(self):
        return {
            "python": copy.deepcopy(random.getstate()),
            "torch_cpu": torch.get_rng_state().clone(),
        }

    def restore_rng(self, value, *, require_cuda=False):
        if require_cuda:
            raise ValueError("This test adapter is CPU only")
        self.restore_calls += 1
        random.setstate(value["python"])
        torch.set_rng_state(value["torch_cpu"])


def optimizers_for(heads):
    result = OrderedDict()
    for arm, head in heads.items():
        parameters = list(head.parameters())
        result[arm] = torch.optim.AdamW([
            {"params": parameters[::2], "lr": 0.003, "weight_decay": 0.005},
            {"params": parameters[1::2], "lr": 0.001, "weight_decay": 0.02},
        ], betas=(0.8, 0.95), eps=1e-7, amsgrad=True, foreach=False)
    return result


def fresh(seed=11):
    torch.manual_seed(seed)
    prototype = torch.nn.Sequential(
        torch.nn.Linear(3, 7), torch.nn.Tanh(), torch.nn.Dropout(0.25),
        torch.nn.Linear(7, 2),
    )
    heads = OrderedDict((arm, copy.deepcopy(prototype)) for arm in ("e", "d"))
    return heads, optimizers_for(heads)


def train(heads, optimizers, batches):
    for inputs, targets in batches:
        for arm, head in heads.items():
            head.train()
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(head(inputs), targets)
            loss.backward()
            optimizer.step()


def checkpoint(heads, optimizers, state, step):
    return copy.deepcopy({
        "format": "BRIGHT_SLOT_MEMORY_ABLATION_V1", "step": step,
        "heads": OrderedDict((a, h.state_dict()) for a, h in heads.items()),
        "optimizers": OrderedDict((a, o.state_dict()) for a, o in optimizers.items()),
        "rng": state.capture_rng(),
    })


class RestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.state = CPUState()
        generator = torch.Generator().manual_seed(73)
        self.batches = [(torch.randn(5, 3, generator=generator),
                         torch.randn(5, 2, generator=generator)) for _ in range(7)]
        self.heads, self.optimizers = fresh()
        train(self.heads, self.optimizers, self.batches[:3])
        self.saved = checkpoint(self.heads, self.optimizers, self.state, step=3)

    def test_resumed_adamw_matches_uninterrupted_with_dropout_and_rng(self):
        saved_hash = self.state.state_fingerprint(self.saved)
        train(self.heads, self.optimizers, self.batches[3:])
        expected = {
            a: (self.state.state_fingerprint(h.state_dict()),
                self.state.state_fingerprint(self.optimizers[a].state_dict()))
            for a, h in self.heads.items()
        }
        resumed, optimizers = fresh(seed=81)
        # The checkpoint must restore recorded flags and parameter groups,
        # including nondefault AdamW options, not current optimizer defaults.
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                group.update(lr=0.1, weight_decay=0.9, amsgrad=False)
        current_rng = self.state.state_fingerprint(self.state.capture_rng())
        evidence = restore_training_state(torch, resumed, optimizers, self.saved, self.state)
        self.assertTrue(evidence["rng_unchanged"])
        self.assertEqual(self.state.restore_calls, 0)
        self.assertEqual(current_rng, self.state.state_fingerprint(self.state.capture_rng()))
        # Simulate loader setup consuming random values before the explicit
        # restoration call. The checkpoint stream must win afterwards.
        torch.rand(19)
        random.random()
        rng_evidence = validate_restored_rng(self.state, self.saved["rng"])
        self.assertTrue(rng_evidence["restored"])
        self.assertEqual(self.state.restore_calls, 1)
        train(resumed, optimizers, self.batches[3:])
        for arm, head in resumed.items():
            self.assertEqual(self.state.state_fingerprint(head.state_dict()), expected[arm][0])
            self.assertEqual(self.state.state_fingerprint(optimizers[arm].state_dict()), expected[arm][1])
            self.assertEqual(evidence["fingerprints"][arm]["counters"]["step_min"], 3)
            self.assertEqual(evidence["fingerprints"][arm]["counters"]["step_max"], 3)
        # Optimizer load must not retain a live alias into the saved payload.
        self.assertEqual(self.state.state_fingerprint(self.saved), saved_hash)

    def test_accepts_diagnostic_checkpoint_schema(self):
        self.saved["format"] = "BRIGHT_SLOT_MEMORY_DIAGNOSTIC_CHECKPOINT_V1"
        heads, optimizers = fresh()
        evidence = restore_training_state(torch, heads, optimizers, self.saved, self.state)
        self.assertEqual(evidence["step"], 3)
        self.assertEqual(evidence["arms"], ["e", "d"])

    def test_rejects_wrong_arm_identity_or_order(self):
        for mutation in ("missing", "extra", "order"):
            with self.subTest(mutation=mutation):
                saved = copy.deepcopy(self.saved)
                if mutation == "missing":
                    del saved["heads"]["d"]
                elif mutation == "extra":
                    saved["heads"]["r"] = copy.deepcopy(saved["heads"]["d"])
                else:
                    saved["heads"] = OrderedDict(reversed(list(saved["heads"].items())))
                heads, optimizers = fresh()
                with self.assertRaisesRegex(ValueError, "arm identity/order"):
                    restore_training_state(torch, heads, optimizers, saved, self.state)

    def test_rejects_incomplete_or_nonintegral_optimizer_counters(self):
        for value in (2, 3.5, float("nan")):
            with self.subTest(value=value):
                saved = copy.deepcopy(self.saved)
                first = next(iter(saved["optimizers"]["e"]["state"].values()))
                first["step"].fill_(value)
                heads, optimizers = fresh()
                with self.assertRaises(ValueError):
                    restore_training_state(torch, heads, optimizers, saved, self.state)

    def test_rejects_nonfinite_moments(self):
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            with self.subTest(moment=name):
                saved = copy.deepcopy(self.saved)
                first = next(iter(saved["optimizers"]["e"]["state"].values()))
                first[name].reshape(-1)[0] = float("nan")
                heads, optimizers = fresh()
                with self.assertRaisesRegex(ValueError, "Nonfinite optimizer moment"):
                    restore_training_state(torch, heads, optimizers, saved, self.state)

    def test_rejects_missing_optimizer_moment_state(self):
        saved = copy.deepcopy(self.saved)
        saved["optimizers"]["e"]["state"].pop(next(iter(saved["optimizers"]["e"]["state"])))
        heads, optimizers = fresh()
        with self.assertRaisesRegex(ValueError, "every head parameter"):
            restore_training_state(torch, heads, optimizers, saved, self.state)

    def test_rejects_live_cross_arm_head_alias(self):
        heads, _ = fresh()
        heads["d"] = heads["e"]
        optimizers = optimizers_for(heads)
        with self.assertRaisesRegex(ValueError, "Cross-arm tensor storage alias"):
            restore_training_state(torch, heads, optimizers, self.saved, self.state)

    def test_rejects_cross_arm_moment_alias_even_when_values_match(self):
        saved = copy.deepcopy(self.saved)
        saved["optimizers"]["d"] = copy.deepcopy(saved["optimizers"]["e"])
        heads, optimizers = fresh()
        ordinary_load = optimizers["d"].load_state_dict

        def aliasing_load(payload):
            ordinary_load(payload)
            first_e = next(iter(optimizers["e"].state.values()))
            first_d = next(iter(optimizers["d"].state.values()))
            first_d["exp_avg"] = first_e["exp_avg"]

        with mock.patch.object(optimizers["d"], "load_state_dict", side_effect=aliasing_load):
            with self.assertRaisesRegex(ValueError, "Cross-arm tensor storage alias"):
                restore_training_state(torch, heads, optimizers, saved, self.state)

    def test_rejects_model_dtype_conversion_and_missing_weights(self):
        heads, optimizers = fresh()
        heads["e"].double()
        with self.assertRaisesRegex(ValueError, "Restored head fingerprint differs"):
            restore_training_state(torch, heads, optimizers, self.saved, self.state)
        heads, optimizers = fresh()
        saved = copy.deepcopy(self.saved)
        saved["heads"]["e"].pop(next(iter(saved["heads"]["e"])))
        with self.assertRaises(RuntimeError):
            restore_training_state(torch, heads, optimizers, saved, self.state)

    def test_rng_equality_failure_is_not_reported_as_restored(self):
        original = self.state.restore_rng

        def advancing_restore(value, **kwargs):
            original(value, **kwargs)
            torch.rand(1)

        with mock.patch.object(self.state, "restore_rng", side_effect=advancing_restore):
            with self.assertRaisesRegex(ValueError, "Restored RNG fingerprint differs"):
                validate_restored_rng(self.state, self.saved["rng"])


if __name__ == "__main__":
    unittest.main()
