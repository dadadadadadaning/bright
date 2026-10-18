"""Synthetic classifier checks; no empirical BRIGHT/CUDA performance claim."""
from __future__ import annotations

import unittest

import torch

try:
    from .model import JointRegionMemoryClassifier
    from .slots import prepare_slots
except ImportError:
    from model import JointRegionMemoryClassifier
    from slots import prepare_slots


def fixture():
    torch.manual_seed(19)
    height, width, regions, members = 2, 4, 3, 5
    pixel = torch.randn(1, 256, height, width)
    q = torch.softmax(torch.randn(1, 9, height * width), 1)
    ids = torch.randint(regions, (9, height * width))
    building = torch.rand(1, 1, height, width) * .8 + .1
    prepared = prepare_slots(pixel, q, ids, regions, (height, width), building, "soft")
    fraction = torch.rand(members, 2); fraction /= fraction.sum(-1, keepdim=True)
    return dict(pixel_features=pixel, region_f=torch.randn(1, regions, 256),
                region_t=torch.randn(1, regions, 128), q=q, ids=ids,
                base_probability_full=torch.softmax(torch.randn(1, 4, 4, 8), 1),
                slot_f=prepared["slot_f"], slot_fraction=prepared["slot_fraction"],
                slot_valid=prepared["slot_valid"], route_q=prepared["route_q"],
                route_ids=prepared["route_ids"], support_ids=torch.tensor([[0, 2, 4], [1, 2, -1], [3, 4, 0]]),
                support_f=torch.randn(members, 2, 256), support_t=torch.randn(members, 128),
                support_labels=torch.softmax(torch.randn(members, 2, 4), -1),
                support_fraction=fraction, support_valid=torch.ones(members, 2, dtype=torch.bool))


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.inputs = fixture()
        self.model = JointRegionMemoryClassifier(hidden=16, heads=4, chunk_size=2)

    def activate_head(self):
        with torch.no_grad():
            self.model.joint_head[-1].weight.normal_(0, .1)

    def test_zero_initial_output_matches_base(self):
        out = self.model(**self.inputs)
        torch.testing.assert_close(out["probabilities"], self.inputs["base_probability_full"], atol=1e-7, rtol=1e-6)
        self.assertEqual(out["delta_grid"].abs().sum().item(), 0)
        self.assertEqual(out["memory_context"].shape, (1, 3, 2, 16))
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in self.model.modules()))

    def test_chunk_outputs_and_gradients_equivalent(self):
        self.activate_head()
        out1 = self.model(**self.inputs, chunk_size=1)
        out1["logits"].square().mean().backward()
        grads1 = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad(set_to_none=True)
        out2 = self.model(**self.inputs, chunk_size=100)
        out2["logits"].square().mean().backward()
        torch.testing.assert_close(out1["probabilities"], out2["probabilities"], atol=1e-6, rtol=1e-5)
        for name, parameter in self.model.named_parameters():
            if name in grads1:
                torch.testing.assert_close(grads1[name], parameter.grad, atol=2e-6, rtol=1e-4)

    def test_candidate_order_invariance(self):
        self.activate_head()
        first = self.model(**self.inputs)
        self.inputs["support_ids"] = self.inputs["support_ids"][:, [2, 0, 1]]
        second = self.model(**self.inputs)
        torch.testing.assert_close(first["memory_context"], second["memory_context"], atol=1e-6, rtol=1e-5)

    def test_no_candidates_zero_context_not_nan(self):
        self.inputs["support_ids"].fill_(-1)
        self.inputs["support_valid"].zero_()
        for key in ("support_f", "support_t", "support_labels", "support_fraction"):
            self.inputs[key].fill_(float("nan"))
        self.activate_head()
        out = self.model(**self.inputs)
        self.assertEqual(out["memory_context"].abs().sum().item(), 0)
        out["logits"].square().mean().backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_empty_support_table(self):
        self.inputs["support_ids"].fill_(-1)
        for key in ("support_f", "support_t", "support_labels", "support_fraction", "support_valid"):
            self.inputs[key] = self.inputs[key][:0]
        out = self.model(**self.inputs)
        self.assertEqual(out["memory_context"].abs().sum().item(), 0)

    def test_invalid_child_is_masked(self):
        self.inputs["support_valid"][0, 0] = False
        self.inputs["support_f"][0, 0] = float("nan")
        self.inputs["support_labels"][0, 0] = float("nan")
        self.inputs["support_fraction"][0, 0] = float("nan")
        out = self.model(**self.inputs)
        self.assertTrue(bool(torch.isfinite(out["memory_context"]).all()))

    def test_gradients_to_both_roles_and_memory_labels(self):
        self.activate_head()
        self.inputs["support_f"].requires_grad_(True)
        self.inputs["support_labels"].requires_grad_(True)
        out = self.model(**self.inputs)
        target = torch.randint(4, (1, 4, 8))
        torch.nn.functional.cross_entropy(out["logits"], target).backward()
        for role in (0, 1):
            self.assertGreater(self.inputs["support_f"].grad[:, role].abs().sum().item(), 0)
            self.assertGreater(self.inputs["support_labels"].grad[:, role].abs().sum().item(), 0)
            self.assertGreater(self.model.role_embedding.weight.grad[role].abs().sum().item(), 0)
        for key in ("query_projection", "key_projection", "value_projection", "label_embedding", "context_projection"):
            self.assertGreater(getattr(self.model, key).weight.grad.abs().sum().item(), 0)

    def test_support_table_encoded_once(self):
        calls = []
        original = self.model._encode_slots
        def recorder(f, t, fraction, valid):
            calls.append(f.shape)
            return original(f, t, fraction, valid)
        self.model._encode_slots = recorder
        self.model(**self.inputs, chunk_size=1)
        self.assertEqual(calls, [torch.Size([3, 2, 256]), torch.Size([5, 2, 256])])

    def test_pixel_branch_keeps_local_differences(self):
        self.activate_head()
        first = self.model(**self.inputs)["delta_grid"]
        self.inputs["pixel_features"] = self.inputs["pixel_features"].clone()
        self.inputs["pixel_features"][:, :, 0, 0] += 5
        second = self.model(**self.inputs)["delta_grid"]
        self.assertGreater((first[:, :, 0, 0] - second[:, :, 0, 0]).abs().sum().item(), 0)
        torch.testing.assert_close(first[:, :, 1], second[:, :, 1])

    def test_query_role_and_parent_backprojection(self):
        # Replacing the correctly split route with two full-strength parent
        # copies must be refused, even if individual IDs are in range.
        self.inputs["route_q"] = torch.cat((self.inputs["q"], self.inputs["q"]), 1)
        with self.assertRaises(ValueError): self.model(**self.inputs)

    def test_bad_support_index_rejected(self):
        self.inputs["support_ids"][0, 0] = 5
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.model(**self.inputs)

    def test_bad_label_or_probability_rejected(self):
        self.inputs["support_labels"][0, 0, 0] = -1
        with self.assertRaises(ValueError): self.model(**self.inputs)
        self.inputs = fixture()
        self.inputs["base_probability_full"] *= 2
        with self.assertRaises(ValueError): self.model(**self.inputs)

    def test_first_update_unblocks_reader(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.1)
        target = torch.randint(4, (1, 4, 8))
        loss = torch.nn.functional.cross_entropy(self.model(**self.inputs)["logits"], target)
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(self.model(**self.inputs)["logits"], target).backward()
        self.assertGreater(self.model.query_projection.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
