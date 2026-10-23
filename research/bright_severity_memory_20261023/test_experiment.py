"""CPU integration contracts; none is CUDA/BRIGHT dataset evidence."""
from __future__ import annotations

import copy
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import experiment
from unittest.mock import patch


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if isinstance(value, dict):
        return repr([(str(k), fingerprint(v)) for k, v in value.items()])
    return repr(value)


STATE = SimpleNamespace(
    capture_rng=torch.random.get_rng_state,
    restore_rng=torch.random.set_rng_state,
    state_fingerprint=fingerprint,
    lr_at_step=lambda step, **kwargs: 0.0001,
)


class ExperimentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_weight_does_not_touch_targets_or_new_readout(self):
        value = torch.tensor(1.2, requires_grad=True)
        loss, stats = experiment.combined_loss(value, {}, object(), {}, {}, object(), read_weight=0)
        self.assertIs(loss, value)
        self.assertFalse(stats["read_loss_evaluated"])

    def test_training_uses_only_original_loss_and_detached_small_diagnostics(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.zeros(1, 4, 2, 2))
            def forward(self, **kwargs):
                self.seen = kwargs
                return {"probabilities": self.logit.softmax(1),
                        "memory_gain": self.logit.sum().detach(),
                        "severity_bias": self.logit.flatten()[:3].detach()}
        model = Model()
        target = torch.tensor([[[0, 1], [2, 3]]])
        example = SimpleNamespace(common={"feature": "unused"}, target=target)
        factory = SimpleNamespace(prepare=lambda example, arms: {"d": {"support": "Train-only"}})
        calls = []
        def loss(p, t):
            calls.append((p, t))
            return torch.nn.functional.nll_loss(p.log(), t)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        row = experiment.update(model, optimizer, example, SimpleNamespace(loss=loss),
                                factory, STATE, object(), step=1, read_weight=0, multiplier=.2)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], target)
        self.assertEqual(row["training_loss"], row["original_loss"])
        self.assertEqual(row["optimizer_step"], 1)
        self.assertEqual(model.seen, {"feature": "unused", "support": "Train-only"})
        self.assertEqual(row["read_loss"], 0)
        self.assertFalse(row["read_loss_evaluated"])
        self.assertIsInstance(row["severity_diagnostics"]["severity_bias"], list)
        json.dumps(row, allow_nan=False)

    def test_auxiliary_loss_cannot_be_silently_enabled(self):
        with self.assertRaisesRegex(ValueError, "zero auxiliary"):
            experiment.combined_loss(torch.tensor(1.), {}, None, {}, {}, None, read_weight=.01)
        for argv in (("plan", "--read-weight", ".01"), ("test",),
                     ("plan", "--variant", "read_aux"), ("resume",)):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                experiment.parser().parse_args(argv)

    def test_option_limits_and_defaults(self):
        for options in (("--steps", "5001"), ("--smoke-steps", "0"),
                        ("--post500-lr", "nan"), ("--seed", "-1")):
            args = experiment.parser().parse_args(["plan", *options])
            with self.assertRaises(ValueError):
                experiment.validate_options(args)
        args = experiment.parser().parse_args(["plan", "--variant", "severity_memory"])
        experiment.validate_options(args)
        self.assertEqual(args.steps, 2500)
        self.assertEqual(args.read_weight, 0)
        self.assertEqual(experiment.schedule(2500), [500, 1000, 1500, 2000, 2500])

    def test_source_identity_pins_adapter_and_vendored_reader(self):
        identity = experiment.code_identity()
        self.assertEqual(set(identity), {"experiment.py", "old_model_adapter.py", "severity_adapter.py"})
        self.assertTrue(all(len(v) == 64 for v in identity.values()))
        self.assertEqual(identity, experiment.code_identity())
        bridge = SimpleNamespace(check_package=lambda p: "addon", verify_ledgers=lambda source: None)
        plan = {"extension_files": identity, "addon_dir": "addon", "addon_sha256": "addon", "source": {}}
        experiment.verify_plan(plan, bridge)
        with patch.object(experiment, "code_identity", return_value={**identity, "severity_adapter.py": "wrong"}):
            with self.assertRaisesRegex(ValueError, "extension changed"):
                experiment.verify_plan(plan, bridge)

    def test_core_state_omits_only_new_calibration_parameters(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = torch.nn.Linear(2, 2)
                self.severity_bias_raw = torch.nn.Parameter(torch.zeros(3))
                self.memory_gain_raw = torch.nn.Parameter(torch.zeros(()))
        model = Model()
        core = experiment.core_state_dict(model)
        self.assertEqual(set(core), {"core.weight", "core.bias"})
        self.assertTrue(torch.equal(core["core.weight"], model.core.weight))

    def test_first_training_failure_survives_secondary_error(self):
        status = dict(phase="update_4", current_image="Train-image", last_complete_step=3)
        with tempfile.TemporaryDirectory() as folder:
            for message in ("first route failure", "secondary cleanup failure"):
                with self.assertRaisesRegex(RuntimeError, message):
                    with experiment.capture_training_failure(folder, status):
                        raise RuntimeError(message)
            record = json.loads((Path(folder) / "first_failure.json").read_text())
            self.assertEqual(record["error"], "RuntimeError: first route failure")
            self.assertEqual(record["last_complete_step"], 3)
            self.assertEqual(record["image"], "Train-image")
            self.assertFalse(record["automatic_retry"])

    def test_metric_definitions_and_absent_classes(self):
        cm = [[80, 10, 0, 0], [5, 20, 5, 0], [0, 5, 10, 0], [0, 0, 0, 0]]
        result = experiment.metrics(cm)
        expected = [80/95, 20/45, 10/20, None]
        self.assertEqual(result["iou_pct"][-1], None)
        self.assertAlmostEqual(result["miou4_pct"], 100*sum(v or 0 for v in expected)/4)
        self.assertAlmostEqual(result["building_iou_pct"], 100*40/55)

    def test_probability_validation_rejects_corruption(self):
        p = torch.ones(1, 4, 2, 3)/4
        target = torch.zeros(1, 2, 3, dtype=torch.long)
        experiment.validate_probability(p, target)
        broken = p.clone()
        broken[0, 0, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, "no values were repaired"):
            experiment.validate_probability(broken, target)
        self.assertEqual(float(broken[0, 0, 0, 0]), float(torch.tensor(.1)))

    def test_checkpoint_stores_mode_and_roundtrips_optimizer_and_rng(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        plan = dict(variant="severity_memory", effective_mode="severity_memory", read_weight=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoints"/"checkpoint_000001.pt"
            result = experiment.save_checkpoint(path, model, optimizer, STATE,
                                               SimpleNamespace(protocol_info={}), plan, 1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["effective_mode"], "severity_memory")
            self.assertEqual(payload["format"], "BRIGHT_SEVERITY_MEMORY_V1_CHECKPOINT")
            self.assertTrue(result["roundtrip_verified"])
            with self.assertRaises(FileExistsError):
                experiment.save_checkpoint(path, model, optimizer, STATE,
                                           SimpleNamespace(protocol_info={}), plan, 1)

    def evaluation_case(self, names):
        targets = {"a": torch.tensor([[[0, 1], [2, 3]]]),
                   "b": torch.tensor([[[0, 0], [255, 1]]])}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1, 4, 2, 2))

            def forward(self, **kwargs):
                assert not self.training and not torch.is_grad_enabled()
                return {"probabilities": (self.bias + torch.rand(()) * 0).softmax(1)}

        def prepare(name):
            return SimpleNamespace(image=name, event="event", target=targets[name],
                common={"base_probability_full": torch.full((1, 4, 2, 2), .25)})

        def cm(t, p):
            mask = t != 255
            return np.bincount(4*t[mask]+p[mask], minlength=16).reshape(4, 4)

        def transition(t, p, q):
            mask = t != 255
            return np.bincount(16*t[mask]+4*p[mask]+q[mask], minlength=64).reshape(4, 4, 4)

        model = Model()
        runtime = SimpleNamespace(val_names=["a", "b"], train_names=["train"],
            evaluation_batches=lambda requested, seed: iter(names), prepare=prepare,
            loss=lambda p, target: p.sum()*0)
        factory = SimpleNamespace(prepare=lambda example, arms: {"d": {}})
        original = SimpleNamespace(_assert_example=lambda example: None,
                                   _confusion=cm, _transition=transition)
        return runtime, model, factory, original

    def test_evaluation_restores_rng_modes_and_counts_valid_pixels(self):
        runtime, model, factory, original = self.evaluation_case(["a", "b"])
        before = STATE.capture_rng()
        with tempfile.TemporaryDirectory() as root:
            result = experiment.evaluate(runtime, model, factory, STATE, original, ["a", "b"],
                Path(root)/"eval", step=7, split="val", seed=3407, smoke=True, variant="original")
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, STATE.capture_rng()))
        self.assertEqual(sum(map(sum, result["metrics"]["original"]["confusion"])), 7)
        self.assertEqual(result["image_count"], 2)
        self.assertFalse(result["complete"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertTrue(result["head_state_unchanged"])

    def test_partial_or_reordered_evaluation_never_writes_completed_report(self):
        for order in (["a"], ["b", "a"]):
            runtime, model, factory, original = self.evaluation_case(order)
            with tempfile.TemporaryDirectory() as root:
                folder = Path(root)/"eval"
                with self.assertRaises(ValueError):
                    experiment.evaluate(runtime, model, factory, STATE, original, ["a", "b"],
                        folder, step=7, split="val", seed=3407, smoke=True, variant="original")
                self.assertFalse((folder/"result.json").exists())
                self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
