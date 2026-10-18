"""Paired-training CPU integration contracts, not BRIGHT/CUDA benefit evidence.

The optimizer/RNG/schedule helpers are loaded from the existing continuation
package. The tiny head is a test fixture exercising the inherited head API;
it is not an alternate implementation of the production dual-slot classifier.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import engine


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyReaderHead(torch.nn.Module):
    """Small residual classifier with stochastic training and named reader params."""
    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Conv2d(3, 4, 1)
        self.value_projection = torch.nn.Linear(4, 4, bias=False)
        self.label_embedding = torch.nn.Linear(4, 4, bias=False)
        self.context_projection = torch.nn.Conv2d(4, 4, 1)
        self.classifier = torch.nn.Conv2d(4, 4, 1)
        self.dropout = torch.nn.Dropout(.35)
        self.last_support_role = None
        self.last_slot_sums = None

    def forward(self, *, features, slot_features, base_probability_full,
                support_features, support_labels, support_role):
        if not self.training:
            raise AssertionError("Training engine must put inherited heads into train mode")
        # Both fixture slots and both support fields must reach the head intact.
        if slot_features.shape[1] != 2:
            raise AssertionError("Expected both input slot fields")
        self.last_slot_sums = slot_features.detach().sum(dim=(0, 2, 3, 4)).tolist()
        self.last_support_role = support_role
        reference = (self.value_projection(support_features) +
                     self.label_embedding(support_labels)).mean(dim=1)[:, :, None, None]
        fused = self.query_projection(features) + slot_features.sum(dim=1) + reference
        delta = self.classifier(torch.tanh(self.context_projection(self.dropout(fused))))
        logits = base_probability_full.log() + delta
        return {"logits": logits, "probabilities": logits.softmax(dim=1)}


class RuntimeFixture:
    def __init__(self, objectives, sentinel=False):
        self.objectives = objectives
        self.loss_calls = []
        self.sentinel = sentinel

    def loss(self, probability, target):
        self.loss_calls.append((probability.shape, target.detach().clone()))
        if self.sentinel:
            # A connected constant discriminates delegated runtime.loss from any
            # hidden/reimplemented objective in the engine.
            return probability.sum() * 0 + 7.0
        return self.objectives.objective_terms(probability, target)["loss"]


def _example():
    teacher = torch.tensor([.55, .25, .12, .08]).reshape(1, 4, 1, 1).expand(1, 4, 2, 3).clone()
    return SimpleNamespace(
        common={
            "features": torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 17,
            "slot_features": torch.stack((torch.full((1, 4, 2, 3), .2),
                                           torch.full((1, 4, 2, 3), -.1)), dim=1),
            "base_probability_full": teacher,
        },
        supports={
            "b": {"support_features": torch.tensor([[[.1, .2, .3, .4], [.3, .1, .5, .2]]]),
                  "support_labels": torch.tensor([[[.5, .3, .1, .1], [.2, .2, .5, .1]]]),
                  "support_role": "image-reference"},
            "d": {"support_features": torch.tensor([[[.6, .2, .1, .4], [.1, .5, .3, .7]]]),
                  "support_labels": torch.tensor([[[.1, .1, .7, .1], [.2, .6, .1, .1]]]),
                  "support_role": "train-memory"},
        },
        target=torch.tensor([[[0, 1, 2], [3, 255, 2]]]),
    )


class EngineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = Path(os.environ.get("BRIGHT_LOSS_HELPERS", str(HERE.parent / "joint_loss_after500_v1")))
        if not (helpers / "continuation.py").is_file() or not (helpers / "objectives.py").is_file():
            raise unittest.SkipTest("Pinned continuation/objective helpers required; bootstrap validates dependencies")
        cls.state = _load("anchor_engine_test_continuation", helpers / "continuation.py")
        cls.objectives = _load("anchor_engine_test_objectives", helpers / "objectives.py")
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        random.seed(321)
        np.random.seed(321)
        torch.manual_seed(321)
        # Produce a real 500-update shared prefix rather than relabelling Adam
        # counters or starting a weights-only continuation.
        cls.prototype = TinyReaderHead()
        cls.prototype_optimizer = torch.optim.AdamW(cls.prototype.parameters(), lr=1e-4, weight_decay=.005)
        cls.prefix_runtime = RuntimeFixture(cls.objectives)
        cls.prefix_example = _example()
        cls.prefix_rows = []
        for step in range(1, 501):
            rows = engine.train_step({"control": cls.prototype}, {"control": cls.prototype_optimizer},
                cls.prefix_example, cls.prefix_runtime, step=step, arm="d", state=cls.state,
                maximum_lr=1e-4, lr_multiplier=.2, anchor_weight=.1, prefix=True)
            if step in (1, 499, 500):
                cls.prefix_rows.append(rows["control"])
        cls.prefix_rng = cls.state.capture_rng()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "previous_threads"):
            torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.example = _example()
        self.runtime = RuntimeFixture(self.objectives)
        self.heads, self.optimizers, self.records = engine.fork_pair(
            self.prototype, self.prototype_optimizer, self.state)
        self.state.restore_rng(self.prefix_rng)

    def run_step(self, *, step=501, weight=.1, diagnose=False, arm="d"):
        return engine.train_step(self.heads, self.optimizers, self.example, self.runtime,
            step=step, arm=arm, state=self.state, maximum_lr=1e-4, lr_multiplier=.2,
            anchor_weight=weight, diagnose=diagnose)

    def pair_fingerprints(self):
        return {name: (self.state.state_fingerprint(head.state_dict()),
                       self.state.state_fingerprint(self.optimizers[name].state_dict()))
                for name, head in self.heads.items()}

    def test_fresh_shared_prefix_has_500_actual_original_objective_updates(self):
        self.assertEqual(len(self.prefix_runtime.loss_calls), 500)
        self.assertEqual([row["optimizer_step"] for row in self.prefix_rows], [1, 499, 500])
        self.assertTrue(all(row["trained_objective"] == "original" for row in self.prefix_rows))
        self.assertAlmostEqual(self.prefix_rows[0]["lr"], 1e-4 / 500, places=15)
        self.assertAlmostEqual(self.prefix_rows[-1]["lr"], 1e-4, places=15)
        self.assertTrue(all(int(item["step"].item()) == 500
                            for item in self.prototype_optimizer.state.values()))

    def test_fork_copies_weights_and_all_moments_without_shared_storage(self):
        self.assertEqual(self.records["control"], self.records["anchor"])
        source_params = list(self.prototype.parameters())
        control_params = list(self.heads["control"].parameters())
        anchor_params = list(self.heads["anchor"].parameters())
        for source, control, anchor in zip(source_params, control_params, anchor_params):
            self.assertEqual(len({source.data_ptr(), control.data_ptr(), anchor.data_ptr()}), 3)
            for key in ("exp_avg", "exp_avg_sq"):
                tensors = [opt.state[param][key] for opt, param in (
                    (self.prototype_optimizer, source), (self.optimizers["control"], control),
                    (self.optimizers["anchor"], anchor))]
                self.assertEqual(len({tensor.data_ptr() for tensor in tensors}), 3)
                torch.testing.assert_close(tensors[0], tensors[1], rtol=0, atol=0)
                torch.testing.assert_close(tensors[1], tensors[2], rtol=0, atol=0)
        with torch.no_grad():
            control_params[0].add_(1)
            self.optimizers["control"].state[control_params[0]]["exp_avg"].add_(1)
        torch.testing.assert_close(source_params[0], anchor_params[0], rtol=0, atol=0)
        torch.testing.assert_close(self.prototype_optimizer.state[source_params[0]]["exp_avg"],
                                   self.optimizers["anchor"].state[anchor_params[0]]["exp_avg"], rtol=0, atol=0)

    def test_zero_anchor_weight_keeps_dropout_forks_and_optimizer_states_identical(self):
        self.assertTrue(any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in self.heads["control"].modules()))
        for step in range(501, 505):
            rows = self.run_step(step=step, weight=0)
            self.assertEqual(self.pair_fingerprints()["control"], self.pair_fingerprints()["anchor"])
            self.assertEqual(rows["control"]["original_loss"], rows["anchor"]["original_loss"])
            self.assertEqual(rows["control"]["training_loss"], rows["anchor"]["training_loss"])
            self.assertEqual(rows["anchor"]["anchor"]["weighted_kl"], 0)
        self.assertEqual(len(self.runtime.loss_calls), 8)

    def test_positive_kl_changes_student_but_preserves_teacher_frozen_inputs_and_source(self):
        frozen_before = self.state.state_fingerprint((self.example.common, self.example.supports, self.example.target))
        source_before = self.state.state_fingerprint((self.prototype.state_dict(), self.prototype_optimizer.state_dict()))
        rows = self.run_step(weight=.5, diagnose=True)
        self.assertNotEqual(self.pair_fingerprints()["control"], self.pair_fingerprints()["anchor"])
        self.assertGreater(rows["anchor"]["anchor"]["weighted_kl"], 0)
        self.assertGreater(rows["anchor"]["reader_grad_abs_sum"], 0)
        self.assertGreater(rows["anchor"]["parameter_update_l2"], 0)
        self.assertEqual(rows["anchor"]["gradient_alignment"]["gradient_space"], "logits")
        self.assertEqual(frozen_before, self.state.state_fingerprint((self.example.common, self.example.supports, self.example.target)))
        self.assertEqual(source_before, self.state.state_fingerprint((self.prototype.state_dict(), self.prototype_optimizer.state_dict())))
        self.assertIsNone(self.example.common["base_probability_full"].grad)
        for name in self.heads:
            self.assertEqual(self.heads[name].last_support_role, "train-memory")
            self.assertEqual(self.heads[name].last_slot_sums,
                             self.example.common["slot_features"].sum(dim=(0, 2, 3, 4)).tolist())

    def test_b_support_is_passed_to_the_same_inherited_head(self):
        self.run_step(weight=0, arm="b")
        self.assertTrue(all(head.last_support_role == "image-reference" for head in self.heads.values()))
        self.assertEqual(len(self.runtime.loss_calls), 2)

    def test_runtime_loss_is_the_training_graph_not_an_internal_reimplementation(self):
        head = TinyReaderHead()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0)
        before = self.state.state_fingerprint(head.state_dict())
        runtime = RuntimeFixture(self.objectives, sentinel=True)
        rows = engine.train_step({"control": head}, {"control": optimizer}, self.example, runtime,
            step=1, arm="d", state=self.state, prefix=True, diagnose=True)
        self.assertEqual(len(runtime.loss_calls), 1)
        self.assertEqual(rows["control"]["original_loss"], 7)
        self.assertEqual(rows["control"]["training_loss"], 7)
        self.assertEqual(rows["control"]["reader_grad_abs_sum"], 0)
        self.assertEqual(rows["control"]["parameter_update_l2"], 0)
        self.assertEqual(before, self.state.state_fingerprint(head.state_dict()))

    def test_step501_keeps_original_horizon_and_scales_only_post500(self):
        rows = self.run_step()
        expected = 1e-4 * .5 * (1 + math.cos(math.pi / 4500)) * .2
        for name in ("control", "anchor"):
            self.assertAlmostEqual(rows[name]["lr"], expected, places=15)
            self.assertEqual(rows[name]["optimizer_step"], 501)
            self.assertTrue(all(int(item["step"].item()) == 501 for item in self.optimizers[name].state.values()))
        self.assertNotAlmostEqual(expected, 1e-4 * .2 / 500, places=10)  # no restarted warmup

    def test_paired_rng_advances_as_one_forward_not_two(self):
        before = self.state.capture_rng()
        one_head = copy.deepcopy(self.heads["control"])
        one_optimizer = torch.optim.AdamW(one_head.parameters(), lr=1e-4)
        one_optimizer.load_state_dict(copy.deepcopy(self.optimizers["control"].state_dict()))
        self.run_step(weight=.1)
        pair_after = self.state.state_fingerprint(self.state.capture_rng())
        self.state.restore_rng(before)
        engine.train_step({"control": one_head}, {"control": one_optimizer}, self.example,
            RuntimeFixture(self.objectives), step=501, arm="d", state=self.state,
            prefix=True, anchor_weight=.1)
        self.assertEqual(pair_after, self.state.state_fingerprint(self.state.capture_rng()))

    def test_diagnostics_do_not_change_updates_or_random_stream(self):
        before = self.state.capture_rng()
        self.run_step(weight=.1, diagnose=False)
        expected = self.pair_fingerprints()
        expected_rng = self.state.state_fingerprint(self.state.capture_rng())
        self.heads, self.optimizers, self.records = engine.fork_pair(self.prototype, self.prototype_optimizer, self.state)
        self.state.restore_rng(before)
        self.run_step(weight=.1, diagnose=True)
        self.assertEqual(expected, self.pair_fingerprints())
        self.assertEqual(expected_rng, self.state.state_fingerprint(self.state.capture_rng()))

    def test_branch_iteration_order_does_not_change_pair(self):
        before = self.state.capture_rng()
        self.run_step(weight=.1)
        expected = self.pair_fingerprints()
        self.heads, self.optimizers, self.records = engine.fork_pair(self.prototype, self.prototype_optimizer, self.state)
        self.heads = {name: self.heads[name] for name in ("anchor", "control")}
        self.optimizers = {name: self.optimizers[name] for name in ("anchor", "control")}
        self.state.restore_rng(before)
        self.run_step(weight=.1)
        self.assertEqual(expected, self.pair_fingerprints())

    def test_wrong_actual_optimizer_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Adam update count mismatch"):
            self.run_step(step=502)

    def test_wrong_branch_set_and_unsupported_arm_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Wrong paired/prefix model set"):
            engine.train_step({"control": self.heads["control"]}, self.optimizers, self.example,
                              self.runtime, step=501, arm="d", state=self.state)
        with self.assertRaisesRegex(ValueError, "Invalid arm"):
            self.run_step(arm="a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
