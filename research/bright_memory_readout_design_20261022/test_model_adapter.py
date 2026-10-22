"""CPU contracts against the real published classifier, not a mocked Reader.

Set BRIGHT_REFERENCE_MODEL to the corresponding original/model.py on another
machine. These contracts do not certify CUDA execution or experimental gains.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import torch

from model_adapter import MODES, SUPPORTED_READ_AST_SHA256, make_classifier, reader_source_fingerprint


REFERENCE = Path(os.environ.get(
    "BRIGHT_REFERENCE_MODEL",
    "/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2/deps/original/model.py",
))


def load_reference(path=REFERENCE):
    package_name = "_readout_test_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.JointRegionMemoryClassifier, sys.modules[package_name + ".slots"]


def fixture(slots, *, empty_bank=False, zero_building=False):
    torch.manual_seed(187)
    regions, members, height, width = 3, (0 if empty_bank else 4), 2, 3
    pixel = torch.randn(1, 256, height, width)
    q = torch.rand(1, 9, height * width)
    q = q / q.sum(1, keepdim=True)
    ids = torch.arange(9 * height * width).reshape(9, -1) % regions
    building = (torch.zeros(1, 1, height * 2, width * 2) if zero_building
                else torch.rand(1, 1, height * 2, width * 2))
    prepared = slots.prepare_slots(pixel, q, ids, regions, (height, width), building, "soft")
    support_labels = torch.rand(members, 2, 4)
    if members:
        support_labels = support_labels / support_labels.sum(-1, keepdim=True)
    out = dict(
        pixel_features=pixel, region_f=torch.randn(1, regions, 256),
        region_t=torch.randn(1, regions, 128), q=q, ids=ids,
        base_probability_full=torch.rand(1, 4, height * 2, width * 2).softmax(1),
        slot_f=prepared["slot_f"], slot_fraction=prepared["slot_fraction"],
        slot_valid=prepared["slot_valid"], route_q=prepared["route_q"],
        route_ids=prepared["route_ids"],
        support_ids=(torch.full((regions, 3), -1, dtype=torch.long) if empty_bank
                     else torch.tensor([[0, 1, -1], [1, 2, 3], [-1, -1, -1]])),
        support_f=torch.randn(members, 2, 256), support_t=torch.randn(members, 128),
        support_labels=support_labels, support_fraction=torch.full((members, 2), .5),
        support_valid=torch.ones(members, 2, dtype=torch.bool),
    )
    return out


class ReadoutAdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base, cls.slots = load_reference()

    def models(self, mode):
        torch.manual_seed(73)
        base = self.base(hidden=8, heads=2, chunk_size=2)
        # Avoid the trivial initial equality caused by zero-initialized delta.
        with torch.no_grad():
            base.joint_head[-1].weight.normal_(0, .03)
            base.joint_head[-1].bias.normal_(0, .03)
        adapted = make_classifier(self.base, mode)(hidden=8, heads=2, chunk_size=2)
        adapted.load_state_dict(base.state_dict(), strict=True)
        return base, adapted

    def clean(self, model):
        self.assertFalse(any(k.startswith("_readout_") for k in model.__dict__))
        self.assertEqual(len(model.label_embedding._forward_hooks), 0)

    def test_source_and_parameter_schema(self):
        self.assertEqual(reader_source_fingerprint(self.base), SUPPORTED_READ_AST_SHA256)
        for mode in MODES:
            base, adapted = self.models(mode)
            self.assertEqual(list(base.state_dict()), list(adapted.state_dict()))
            self.assertEqual(sum(p.numel() for p in base.parameters()),
                             sum(p.numel() for p in adapted.parameters()))

    def test_original_path_is_identical(self):
        base, adapted = self.models("original")
        inputs = fixture(self.slots)
        reference, result = base(**inputs), adapted(**inputs)
        self.assertEqual(set(reference), set(result))
        for key in reference:
            self.assertTrue(torch.equal(reference[key], result[key]), key)
        self.clean(adapted)

    def test_observe_forward_gradient_and_inputs_unchanged(self):
        base, adapted = self.models("observe")
        inputs = fixture(self.slots)
        before = {key: value.clone() for key, value in inputs.items()}
        reference, result = base(**inputs), adapted(**inputs)
        for key in reference:
            self.assertTrue(torch.equal(reference[key], result[key]), key)
        reference["logits"].square().mean().backward()
        result["logits"].square().mean().backward()
        for (name, p), (other, q) in zip(base.named_parameters(), adapted.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                self.assertTrue(torch.equal(p.grad, q.grad), name)
        for key, value in inputs.items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.clean(adapted)

    def test_uniform_attention_label_readout_and_component_maxima(self):
        _, model = self.models("read_aux")
        with torch.no_grad():
            model.query_projection.weight.zero_()
            model.query_projection.bias.zero_()
        inputs = fixture(self.slots)
        result = model(**inputs)
        self.assertEqual(tuple(result["read_probability"].shape), (1, 3, 2, 4))
        for region, ids in enumerate(inputs["support_ids"]):
            legal = ids[ids >= 0]
            if len(legal):
                candidates = inputs["support_labels"][legal].flatten(0, 1)
                for role in range(2):
                    torch.testing.assert_close(result["read_probability"][0, region, role],
                                               candidates.mean(0), rtol=1e-6, atol=1e-7)
                    self.assertTrue(torch.equal(result["candidate_class_max"][0, region, role],
                                                candidates.amax(0)))
            else:
                self.assertFalse(bool(result["read_valid"][0, region].any()))
                self.assertEqual(float(result["read_probability"][0, region].abs().sum()), 0)
        self.clean(model)

    def test_auxiliary_gradient_reaches_queries_and_keys_not_labels_or_values(self):
        _, model = self.models("read_aux")
        inputs = fixture(self.slots)
        inputs["support_labels"].requires_grad_(True)
        result = model(**inputs)
        valid = result["read_valid"]
        loss = -result["read_probability"][..., 2][valid].log().mean()
        loss.backward()
        self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.key_projection.weight.grad.abs().sum()), 0)
        self.assertIsNone(inputs["support_labels"].grad)
        self.assertIsNone(model.value_projection.weight.grad)
        self.assertIsNone(model.label_embedding.weight.grad)
        self.clean(model)

    def test_empty_bank_is_zero_and_differentiable(self):
        base, model = self.models("read_aux")
        inputs = fixture(self.slots, empty_bank=True)
        result, reference = model(**inputs), base(**inputs)
        self.assertTrue(torch.equal(result["probabilities"], reference["probabilities"]))
        self.assertFalse(bool(result["read_valid"].any()))
        self.assertEqual(float(result["read_probability"].abs().sum()), 0)
        result["read_probability"].sum().backward()
        self.clean(model)

    def test_invalid_support_slots_are_masked_without_nan_leak(self):
        base, model = self.models("observe")
        inputs = fixture(self.slots)
        inputs["support_valid"].zero_()
        inputs["support_fraction"].zero_()
        inputs["support_labels"].fill_(float("nan"))
        inputs["support_f"].fill_(float("nan"))
        inputs["support_t"].fill_(float("nan"))
        result, reference = model(**inputs), base(**inputs)
        self.assertTrue(torch.equal(result["probabilities"], reference["probabilities"]))
        self.assertTrue(bool(torch.isfinite(result["read_probability"]).all()))
        self.assertFalse(bool(result["read_valid"].any()))
        self.assertEqual(float(result["candidate_class_max"].abs().sum()), 0)
        self.clean(model)

    def test_invalid_query_slots_have_no_readout(self):
        _, model = self.models("read_aux")
        inputs = fixture(self.slots, zero_building=True)
        result = model(**inputs)
        self.assertFalse(bool(result["read_valid"][..., 1].any()))
        self.assertEqual(float(result["read_probability"][:, :, 1].abs().sum()), 0)
        self.assertEqual(float(result["candidate_class_max"][:, :, 1].abs().sum()), 0)
        self.clean(model)

    def test_no_label_prediction_invariant_and_hook_removed(self):
        _, model = self.models("no_label")
        inputs = fixture(self.slots)
        first = model(**inputs)
        self.clean(model)
        changed = dict(inputs, support_labels=inputs["support_labels"].flip(-1))
        second = model(**changed)
        self.assertTrue(torch.equal(first["probabilities"], second["probabilities"]))
        self.assertTrue(torch.equal(first["memory_context"], second["memory_context"]))
        self.clean(model)

    def test_no_reader_prediction_invariant_to_all_support_content(self):
        _, model = self.models("no_reader")
        inputs = fixture(self.slots)
        first = model(**inputs)
        changed = dict(inputs, support_f=inputs["support_f"] * -3,
                       support_t=inputs["support_t"] + 20,
                       support_labels=inputs["support_labels"].flip(-1),
                       support_ids=torch.full_like(inputs["support_ids"], 2))
        second = model(**changed)
        self.assertTrue(torch.equal(first["probabilities"], second["probabilities"]))
        self.assertEqual(float(first["memory_context"].abs().sum()), 0)
        self.clean(model)

    def test_rejects_corrupt_routes_and_cleans_transient_state(self):
        for mode in MODES:
            _, model = self.models(mode)
            inputs = fixture(self.slots)
            inputs["route_q"][0, 0, 0] *= .5
            before = inputs["route_q"].clone()
            with self.assertRaisesRegex(ValueError, "sum to one|preserve"):
                model(**inputs)
            self.assertTrue(torch.equal(before, inputs["route_q"]))
            self.clean(model)

    def test_rejects_large_id_and_does_not_clip(self):
        for mode in MODES:
            _, model = self.models(mode)
            inputs = fixture(self.slots)
            inputs["ids"][0, 0] = 2 ** 51 + 25
            with self.assertRaisesRegex(ValueError, "out-of-range route ids"):
                model(**inputs)
            self.assertEqual(int(inputs["ids"][0, 0]), 2 ** 51 + 25)
            self.clean(model)

    def test_no_query_gt_argument_and_cleanup(self):
        _, model = self.models("read_aux")
        inputs = fixture(self.slots)
        with self.assertRaises(TypeError):
            model(**inputs, query_target=torch.zeros(1, 4, 6))
        self.clean(model)

    def test_two_forwards_keep_only_caller_owned_graphs(self):
        _, model = self.models("read_aux")
        inputs = fixture(self.slots)
        first = model(**inputs)
        self.clean(model)
        second = model(**inputs)
        self.clean(model)
        loss = first["read_probability"][..., 1].sum() + second["read_probability"][..., 2].sum()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(model.query_projection.weight.grad).all()))

    def test_changed_base_reader_is_rejected(self):
        class ChangedReader(self.base):
            def _read(self, *args, **kwargs):
                return super()._read(*args, **kwargs)
        with self.assertRaisesRegex(ValueError, "differs"):
            make_classifier(ChangedReader, "observe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
