"""CPU contracts using the actual published model and route implementation.

These verify semantics and gradient paths, not CUDA stability or performance.
The old adapter is bundled beside this file. Set BRIGHT_REFERENCE_MODEL to the
installed original/model.py on the server. BRIGHT_READOUT_BASE_DIR optionally
overrides the bundled adapter with that directory's model_adapter.py.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import torch

from severity_adapter import MODES, make_classifier, preserve_foreground_mass, shuffled_label_copy


OLD_ADAPTER = (Path(os.environ["BRIGHT_READOUT_BASE_DIR"]) / "model_adapter.py"
               if os.environ.get("BRIGHT_READOUT_BASE_DIR") else
               Path(__file__).resolve().with_name("old_model_adapter.py"))
REFERENCE = Path(os.environ.get("BRIGHT_REFERENCE_MODEL",
    "/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2/deps/original/model.py"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference():
    package_name = "_severity_test_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REFERENCE.parent)]
    sys.modules[package_name] = package
    model = load_module(package_name + ".model", REFERENCE)
    return model.JointRegionMemoryClassifier, sys.modules[package_name + ".slots"]


def fixture(slots, *, empty_bank=False):
    torch.manual_seed(187)
    regions, members, height, width = 3, (0 if empty_bank else 4), 2, 3
    pixel = torch.randn(1, 256, height, width)
    q = torch.rand(1, 9, height * width)
    q = q / q.sum(1, keepdim=True)
    ids = torch.arange(9 * height * width).reshape(9, -1) % regions
    building = torch.rand(1, 1, height * 2, width * 2)
    prepared = slots.prepare_slots(pixel, q, ids, regions, (height, width), building, "soft")
    labels = torch.rand(members, 2, 4)
    if members:
        labels = labels / labels.sum(-1, keepdim=True)
    return dict(
        pixel_features=pixel, region_f=torch.randn(1, regions, 256),
        region_t=torch.randn(1, regions, 128), q=q, ids=ids,
        base_probability_full=torch.rand(1, 4, height * 2, width * 2).softmax(1),
        slot_f=prepared["slot_f"], slot_fraction=prepared["slot_fraction"],
        slot_valid=prepared["slot_valid"], route_q=prepared["route_q"],
        route_ids=prepared["route_ids"],
        support_ids=(torch.full((regions, 3), -1, dtype=torch.long) if empty_bank
                     else torch.tensor([[0, 1, -1], [1, 2, 3], [-1, -1, -1]])),
        support_f=torch.randn(members, 2, 256), support_t=torch.randn(members, 128),
        support_labels=labels, support_fraction=torch.full((members, 2), .5),
        support_valid=torch.ones(members, 2, dtype=torch.bool),
    )


class SeverityAdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base, cls.slots = load_reference()
        cls.old = load_module("_severity_old_adapter", OLD_ADAPTER)

    def model(self, mode, *, random_head=True):
        torch.manual_seed(73)
        model = make_classifier(self.base, self.old, self.slots, mode)(
            hidden=8, heads=2, chunk_size=2, shuffle_seed=3407)
        if random_head:
            with torch.no_grad():
                model.joint_head[-1].weight.normal_(0, .03)
                model.joint_head[-1].bias.normal_(0, .03)
        return model

    def clean(self, model):
        self.assertFalse(any(key.startswith("_readout_") for key in model.__dict__))
        self.assertEqual(len(model.label_embedding._forward_hooks), 0)

    def test_common_initialization_and_rng_are_identical(self):
        torch.manual_seed(51)
        base = self.base(hidden=8, heads=2, chunk_size=2)
        state = torch.get_rng_state().clone()
        for mode in MODES:
            torch.manual_seed(51)
            model = make_classifier(self.base, self.old, self.slots, mode)(
                hidden=8, heads=2, chunk_size=2)
            self.assertTrue(torch.equal(torch.get_rng_state(), state))
            for name, value in base.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), (mode, name))
            extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
            self.assertEqual(extra, 0 if mode == "original" else 4)

    def test_zero_initialization_is_bitwise_original_with_live_gain_gradient(self):
        reference = self.model("original")
        inputs = fixture(self.slots)
        expected = reference(**inputs)
        for mode in MODES:
            model = self.model(mode)
            result = model(**inputs)
            self.assertTrue(torch.equal(result["logits"], expected["logits"]), mode)
            self.assertTrue(torch.equal(result["probabilities"], expected["probabilities"]), mode)
            if mode != "original":
                loss = -result["probabilities"][:, 2].log().mean()
                loss.backward()
                self.assertGreater(float(model.severity_bias_raw.grad.abs().sum()), 0)
                if mode != "calibration":
                    self.assertGreater(float(model.memory_gain_raw.grad.abs()), 0)
                else:
                    self.assertEqual(float(model.memory_gain_raw.grad.abs()), 0)
            self.clean(model)

    def test_nonzero_correction_preserves_probability_mass_and_logits_contract(self):
        inputs = fixture(self.slots)
        reference = self.model("original")(**inputs)
        model = self.model("severity_memory")
        with torch.no_grad():
            model.severity_bias_raw.copy_(torch.tensor([.3, -.6, .9]))
            model.memory_gain_raw.fill_(.7)
        result = model(**inputs)
        self.assertFalse(torch.equal(result["probabilities"], reference["probabilities"]))
        torch.testing.assert_close(result["probabilities"][:, :1], reference["probabilities"][:, :1], atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(result["probabilities"][:, 1:].sum(1), reference["probabilities"][:, 1:].sum(1), atol=2e-7, rtol=1e-6)
        self.assertTrue(torch.equal(result["probabilities"], result["logits"].softmax(1)))
        self.assertTrue(bool(torch.isfinite(result["probabilities"]).all()))
        self.assertTrue(bool((result["probabilities"] >= 0).all()))
        self.assertLessEqual(abs(float(result["memory_gain"])), 2)
        self.assertEqual(result["severity_bias"].shape, (3,))
        for key in ("memory_gain", "severity_evidence_l1", "severity_memory_mass"):
            self.assertEqual(result[key].ndim, 0)
            self.assertFalse(result[key].requires_grad)
        self.clean(model)

    def test_foreground_mass_constraint_does_not_freeze_four_class_argmax_mask(self):
        logits = torch.tensor([.4, .3, .2, .1]).log().reshape(1, 4, 1, 1)
        correction = torch.tensor([2., -2., -2.]).reshape(1, 3, 1, 1)
        adjusted = preserve_foreground_mass(logits, correction).softmax(1)
        self.assertEqual(int(logits.argmax(1)), 0)
        self.assertEqual(int(adjusted.argmax(1)), 1)
        torch.testing.assert_close(adjusted[:, 0], logits.softmax(1)[:, 0])

    def test_memory_only_gradient_reaches_query_key_without_gt_or_label_gradient(self):
        inputs = fixture(self.slots)
        inputs["support_labels"].requires_grad_(True)
        model = self.model("severity_memory", random_head=False)
        with torch.no_grad():
            model.memory_gain_raw.fill_(.5)
        result = model(**inputs)
        (-result["probabilities"][:, 2].log().mean()).backward()
        self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.key_projection.weight.grad.abs().sum()), 0)
        # The visual Reader remains in the graph, but its downstream head is
        # zero initialized here; only the new probability path is active.
        self.assertEqual(float(model.value_projection.weight.grad.abs().sum()), 0)
        if inputs["support_labels"].grad is not None:
            self.assertEqual(float(inputs["support_labels"].grad.abs().sum()), 0)

    def test_empty_memory_has_exact_zero_effect(self):
        inputs = fixture(self.slots, empty_bank=True)
        expected = self.model("original")(**inputs)
        for mode in ("severity_memory", "shuffled_memory"):
            model = self.model(mode)
            with torch.no_grad():
                model.memory_gain_raw.fill_(.8)
            result = model(**inputs)
            self.assertTrue(torch.equal(result["probabilities"], expected["probabilities"]))
            self.assertEqual(float(result["severity_evidence_l1"]), 0)
            self.assertEqual(float(result["severity_memory_mass"]), 0)
            self.clean(model)

    def test_shuffling_only_added_readout_preserves_visual_reader_and_inputs(self):
        inputs = fixture(self.slots)
        before = {key: value.clone() for key, value in inputs.items()}
        normal = self.model("severity_memory")
        shuffled = self.model("shuffled_memory")
        with torch.no_grad():
            normal.memory_gain_raw.fill_(.5)
            shuffled.memory_gain_raw.fill_(.5)
        normal_result = normal(**inputs)
        rng = torch.get_rng_state().clone()
        shuffled_result = shuffled(**inputs)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(normal_result["memory_context"], shuffled_result["memory_context"]))
        self.assertTrue(torch.equal(normal_result["delta_grid"], shuffled_result["delta_grid"]))
        self.assertFalse(torch.equal(normal_result["read_probability"], shuffled_result["read_probability"]))
        self.assertFalse(torch.equal(normal_result["probabilities"], shuffled_result["probabilities"]))
        again = shuffled(**inputs)
        self.assertTrue(torch.equal(again["probabilities"], shuffled_result["probabilities"]))
        for key, value in before.items():
            self.assertTrue(torch.equal(inputs[key], value), key)
        self.clean(shuffled)

    def test_shuffle_preserves_valid_label_marginals_and_invalid_padding(self):
        inputs = fixture(self.slots)
        labels = inputs["support_labels"]
        valid = inputs["support_valid"]
        valid[0, 0] = False
        labels[0, 0].fill_(float("nan"))
        output = shuffled_label_copy(labels, valid, inputs["support_ids"], 3407)
        self.assertEqual(sorted(map(tuple, labels[valid].tolist())), sorted(map(tuple, output[valid].tolist())))
        self.assertTrue(torch.isnan(output[~valid]).all())
        self.assertTrue(torch.equal(labels[valid], inputs["support_labels"][valid]))

    def test_shuffle_cannot_import_unselected_query_or_event_labels(self):
        # Unselected rows deliberately contain a class never present in the
        # eligible pool. These could represent the excluded training query or
        # same-event bank rows; they must neither donate nor receive labels.
        labels = torch.zeros(5, 2, 4)
        labels[..., 0] = 1
        labels[1] = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.]])
        labels[3] = torch.tensor([[0., 0., 0., 1.], [0., .2, .3, .5]])
        valid = torch.ones(5, 2, dtype=torch.bool)
        support_ids = torch.tensor([[1, -1, 3], [3, 1, -1]])
        selected = torch.zeros_like(valid)
        selected[[1, 3]] = True
        before = labels.clone()
        rng = torch.get_rng_state().clone()
        output = shuffled_label_copy(labels, valid, support_ids, 3407)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(labels, before))
        self.assertTrue(torch.equal(output[~selected], before[~selected]))
        self.assertEqual(float(output[selected][:, 0].sum()), 0)
        self.assertEqual(sorted(map(tuple, before[selected].tolist())),
                         sorted(map(tuple, output[selected].tolist())))
        self.assertEqual(sorted(map(tuple, before.reshape(-1, 4).tolist())),
                         sorted(map(tuple, output.reshape(-1, 4).tolist())))
        self.assertTrue(torch.equal(output, shuffled_label_copy(labels, valid, support_ids, 3407)))

    def test_shuffle_with_no_selected_parent_cannot_use_unselected_bank(self):
        inputs = fixture(self.slots)
        selected = torch.full_like(inputs["support_ids"], -1)
        output = shuffled_label_copy(inputs["support_labels"], inputs["support_valid"], selected, 3407)
        self.assertTrue(torch.equal(output, inputs["support_labels"]))

    def test_calibration_cannot_use_probability_readout(self):
        inputs = fixture(self.slots)
        model = self.model("calibration")
        with torch.no_grad():
            model.memory_gain_raw.fill_(1.3)
        result = model(**inputs)
        expected = self.model("original")(**inputs)
        self.assertTrue(torch.equal(result["probabilities"], expected["probabilities"]))
        self.assertEqual(float(result["memory_gain"]), 0)

    def test_route_fault_and_query_gt_are_rejected_without_repair(self):
        for mode in MODES:
            model = self.model(mode)
            inputs = fixture(self.slots)
            with self.assertRaises(TypeError):
                model(**inputs, query_target=torch.zeros(1, 4, 6))
            self.clean(model)
            inputs["route_q"][0, 0, 0] *= .5
            before = inputs["route_q"].clone()
            with self.assertRaisesRegex(ValueError, "sum to one|preserve"):
                model(**inputs)
            self.assertTrue(torch.equal(inputs["route_q"], before))
            self.clean(model)

    def test_large_route_id_is_not_clipped(self):
        model = self.model("severity_memory")
        inputs = fixture(self.slots)
        inputs["ids"][0, 0] = 2 ** 51 + 25
        with self.assertRaisesRegex(ValueError, "out-of-range route ids"):
            model(**inputs)
        self.assertEqual(int(inputs["ids"][0, 0]), 2 ** 51 + 25)
        self.clean(model)

    def test_source_guard_rejects_changed_reader(self):
        class ChangedReader(self.base):
            def _read(self, *args, **kwargs):
                return super()._read(*args, **kwargs)
        with self.assertRaisesRegex(ValueError, "differs"):
            make_classifier(ChangedReader, self.old, self.slots, "severity_memory")

    def test_new_checkpoint_schema_roundtrips(self):
        model = self.model("severity_memory")
        with torch.no_grad():
            model.memory_gain_raw.fill_(-.5)
            model.severity_bias_raw.copy_(torch.tensor([.1, -.3, .2]))
        restored = self.model("severity_memory")
        restored.load_state_dict(model.state_dict(), strict=True)
        inputs = fixture(self.slots)
        self.assertTrue(torch.equal(model(**inputs)["probabilities"], restored(**inputs)["probabilities"]))
        self.assertLess(float(model(**inputs)["memory_gain"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
