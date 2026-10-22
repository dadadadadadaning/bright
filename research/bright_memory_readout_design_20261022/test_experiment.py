"""CPU integration contracts; none is CUDA/BRIGHT dataset evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import experiment
from model_adapter import make_classifier
from test_model_adapter import fixture, load_reference


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
        cls.base, cls.slots = load_reference()

    def test_zero_weight_does_not_touch_targets_or_new_readout(self):
        value = torch.tensor(1.2, requires_grad=True)
        loss, stats = experiment.combined_loss(value, {}, object(), {}, {}, object(), read_weight=0)
        self.assertIs(loss, value)
        self.assertFalse(stats["read_loss_evaluated"])

    def test_auxiliary_step_reaches_query_key_but_not_gt_or_support_labels(self):
        inputs = fixture(self.slots)
        inputs["support_labels"].requires_grad_(True)
        common_names = ("pixel_features", "region_f", "region_t", "q", "ids", "base_probability_full")
        common = {k: inputs[k] for k in common_names}
        support = {k: v for k, v in inputs.items() if k not in common_names}
        model = make_classifier(self.base, "read_aux")(hidden=8, heads=2, chunk_size=2)
        # Isolate the auxiliary objective from the initially zero final residual.
        target = torch.tensor([[[0, 1, 2, 3, 255, 1]] * 4], dtype=torch.long)
        example = SimpleNamespace(common=common, target=target)
        factory = SimpleNamespace(prepare=lambda example, arms: {"d": support})
        runtime = SimpleNamespace(loss=lambda p, t: p.sum() * 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        target_before = target.clone()
        row = experiment.update(model, optimizer, example, runtime, factory, STATE, self.slots,
                                step=1, read_weight=.05, multiplier=.2)
        self.assertTrue(row["read_loss_evaluated"])
        self.assertGreater(row["read_loss"], 0)
        self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.key_projection.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(target, target_before))
        # The original visual path also receives the labels, but its zero scalar
        # loss contributes only zero gradient here; auxiliary labels are detached.
        self.assertTrue(inputs["support_labels"].grad is None or not inputs["support_labels"].grad.any())

    def test_options_reject_silent_non_aux_loss_and_nonfinite_weights(self):
        for options in (("--variant", "original", "--read-weight", "0.1"),
                        ("--variant", "read_aux", "--read-weight", "nan"),
                        ("--steps", "5001"), ("--smoke-steps", "0")):
            args = experiment.parser().parse_args(["plan", *options])
            with self.assertRaises(ValueError):
                experiment.validate_options(args)

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
        plan = dict(variant="no_reader", effective_mode="no_reader", read_weight=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoints"/"checkpoint_000001.pt"
            result = experiment.save_checkpoint(path, model, optimizer, STATE,
                                               SimpleNamespace(protocol_info={}), plan, 1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["effective_mode"], "no_reader")
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
