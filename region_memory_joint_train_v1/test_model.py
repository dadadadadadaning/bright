"""Synthetic CPU contracts only: no real BRIGHT data, CUDA or training result."""

import copy
import unittest

import torch
from torch.nn import functional as F

from model import JointRegionMemoryClassifier, sparse_pool, sparse_unpool


def fixture():
    torch.manual_seed(3407)
    batch, regions, count, height, width = 2, 4, 3, 2, 3
    return dict(
        pixel_features=torch.randn(batch, 256, height, width),
        region_f=torch.randn(batch, regions, 256),
        region_t=torch.randn(batch, regions, 128),
        q=torch.softmax(torch.randn(batch, 2, height*width), dim=1),
        ids=torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]]),
        base_probability_full=torch.softmax(torch.randn(batch, 4, 4, 6), dim=1),
        support_f=torch.randn(batch, regions, count, 256),
        support_t=torch.randn(batch, regions, count, 128),
        support_labels=torch.softmax(torch.randn(batch, regions, count, 4), dim=-1),
        support_valid=torch.ones(batch, regions, count, dtype=torch.bool),
    )


def active_model():
    model = JointRegionMemoryClassifier(chunk_size=2)
    with torch.no_grad():
        model.joint_head[-1].weight.normal_(0.0, 0.02)
    return model


class JointContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_full_resolution_base_and_zero_delta(self):
        inputs = fixture()
        model = JointRegionMemoryClassifier()
        result = model(**inputs)
        self.assertEqual(result["logits"].shape, (2, 4, 4, 6))
        self.assertEqual(torch.count_nonzero(result["delta_grid"]).item(), 0)
        torch.testing.assert_close(result["probabilities"], inputs["base_probability_full"], atol=1e-7, rtol=1e-6)

    def test_zero_base_entries_finite_with_only_documented_log_clamp(self):
        inputs = fixture()
        inputs["base_probability_full"] = torch.zeros(2, 4, 4, 6)
        inputs["base_probability_full"][:, 0] = 1.0
        result = JointRegionMemoryClassifier()(**inputs)
        self.assertTrue(torch.isfinite(result["logits"]).all())
        torch.testing.assert_close(result["probabilities"], inputs["base_probability_full"], atol=4e-8, rtol=0)

    def test_zero_initialized_reader_starts_learning_after_first_head_update(self):
        inputs = fixture()
        model = JointRegionMemoryClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        target = torch.arange(48).reshape(2, 4, 6) % 4
        loss = F.cross_entropy(model(**inputs)["logits"], target)
        loss.backward()
        self.assertGreater(model.joint_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.query_projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(**inputs)["logits"], target).backward()
        for name in ("region_encoder.1.weight", "query_projection.weight", "key_projection.weight",
                     "value_projection.weight", "label_embedding.weight", "pixel_projection.0.weight"):
            self.assertGreater(dict(model.named_parameters())[name].grad.abs().sum().item(), 0, name)

    def test_all_padding_is_finite_and_zero_context(self):
        inputs = fixture()
        inputs["support_valid"].fill_(False)
        for key in ("support_f", "support_t", "support_labels"):
            inputs[key].fill_(float("nan"))
        model = active_model()
        result = model(**inputs)
        self.assertEqual(torch.count_nonzero(result["memory_context"]).item(), 0)
        self.assertTrue(torch.isfinite(result["logits"]).all())
        result["logits"].square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_padding_nan_inf_cannot_affect_prediction_or_gradient(self):
        inputs = fixture()
        inputs["support_valid"][:, :, -1] = False
        model = active_model()
        clean = model(**inputs)
        dirty = copy.deepcopy(inputs)
        dirty["support_f"][:, :, -1] = float("nan")
        dirty["support_t"][:, :, -1] = float("inf")
        dirty["support_labels"][:, :, -1] = float("nan")
        contaminated = model(**dirty)
        torch.testing.assert_close(clean["logits"], contaminated["logits"], atol=0, rtol=0)
        contaminated["logits"].square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_chunked_output_and_gradients_match(self):
        inputs = fixture()
        model = active_model()
        one = model(**inputs, chunk_size=1)
        one["logits"].square().mean().backward()
        one_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        all_regions = model(**inputs, chunk_size=4)
        all_regions["logits"].square().mean().backward()
        for key in one:
            torch.testing.assert_close(one[key], all_regions[key], atol=2e-6, rtol=2e-5)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(one_grads[name], parameter.grad, atol=2e-6, rtol=2e-5)

    def test_identical_support_tensors_produce_identical_result_independent_of_source(self):
        current_image_inputs = fixture()
        same_historical_inputs = copy.deepcopy(current_image_inputs)
        model = active_model()
        first, second = model(**current_image_inputs), model(**same_historical_inputs)
        torch.testing.assert_close(first["logits"], second["logits"], atol=0, rtol=0)

    def test_neighbor_order_permutation_and_invalid_extension_do_not_change_result(self):
        inputs = fixture()
        model = active_model()
        first = model(**inputs)
        reordered = copy.deepcopy(inputs)
        for key in ("support_f", "support_t", "support_labels", "support_valid"):
            reordered[key] = reordered[key][:, :, [2, 0, 1]]
        second = model(**reordered)
        torch.testing.assert_close(first["logits"], second["logits"], atol=2e-6, rtol=2e-5)
        for key in ("support_f", "support_t", "support_labels"):
            padding = torch.full_like(reordered[key][:, :, :1], float("nan"))
            reordered[key] = torch.cat((reordered[key], padding), dim=2)
        reordered["support_valid"] = torch.cat((reordered["support_valid"],
                                                 torch.zeros_like(reordered["support_valid"][:, :, :1])), dim=2)
        third = model(**reordered)
        torch.testing.assert_close(second["logits"], third["logits"], atol=2e-6, rtol=2e-5)

    def test_support_labels_and_features_have_independent_gradient_paths(self):
        inputs = fixture()
        for key in ("pixel_features", "region_f", "region_t", "support_f", "support_t", "support_labels"):
            inputs[key].requires_grad_(True)
        model = active_model()
        model(**inputs)["logits"].square().mean().backward()
        for key in ("pixel_features", "region_f", "region_t", "support_f", "support_t", "support_labels"):
            self.assertGreater(inputs[key].grad.abs().sum().item(), 0, key)
        changed = {k: v.detach().clone() for k, v in inputs.items()}
        changed["support_labels"] = changed["support_labels"].roll(1, dims=-1)
        self.assertGreater((model(**inputs)["logits"] - model(**changed)["logits"]).abs().max().item(), 1e-6)

    def test_pixel_outputs_need_not_be_constant_inside_one_region(self):
        inputs = fixture()
        inputs["ids"].fill_(0)
        inputs["base_probability_full"].fill_(0.25)
        result = active_model()(**inputs)
        self.assertGreater(result["delta_grid"].flatten(2).std(-1).max().item(), 1e-6)

    def test_sparse_unpool_constant_and_duplicate_weighted_ids(self):
        q = torch.tensor([[[0.2, 0.3], [0.3, 0.1], [0.5, 0.6]]])
        ids = torch.tensor([[0, 0], [0, 1], [1, 1]])
        values = torch.tensor([[[2.0], [10.0]]])
        torch.testing.assert_close(sparse_unpool(values, q, ids), torch.tensor([[[6.0, 7.6]]]))
        torch.testing.assert_close(sparse_unpool(torch.ones(1, 2, 3), q, ids), torch.ones(1, 3, 2))

    def test_sparse_pool_exact_mean_and_empty_region(self):
        q = torch.tensor([[[0.2, 0.3], [0.3, 0.1], [0.5, 0.6]]])
        ids = torch.tensor([[0, 0], [0, 1], [1, 1]])
        mean, mass = sparse_pool(torch.tensor([[[2.0, 10.0]]]), q, ids, regions=3)
        torch.testing.assert_close(mass, torch.tensor([[0.8, 1.2, 0.0]]))
        torch.testing.assert_close(mean, torch.tensor([[[5.0], [8.0/1.2], [0.0]]]))

    def test_invalid_q_and_ids_are_rejected(self):
        variants = []
        negative = fixture(); negative["q"][0, 0, 0] = -0.1; variants.append(negative)
        bad_sum = fixture(); bad_sum["q"] *= 0.5; variants.append(bad_sum)
        bad_id = fixture(); bad_id["ids"][0, 0] = 4; variants.append(bad_id)
        bad_dtype = fixture(); bad_dtype["ids"] = bad_dtype["ids"].float(); variants.append(bad_dtype)
        model = JointRegionMemoryClassifier()
        for inputs in variants:
            with self.assertRaises(ValueError):
                model(**inputs)

    def test_shape_dtype_labels_and_valid_nan_are_rejected(self):
        variants = []
        shape = fixture(); shape["region_t"] = shape["region_t"][:, :, :127]; variants.append(shape)
        dtype = fixture(); dtype["support_f"] = dtype["support_f"].half(); variants.append(dtype)
        labels = fixture(); labels["support_labels"] *= 2; variants.append(labels)
        valid_nan = fixture(); valid_nan["support_f"][0, 0, 0, 0] = float("nan"); variants.append(valid_nan)
        base = fixture(); base["base_probability_full"] *= 0.5; variants.append(base)
        model = JointRegionMemoryClassifier()
        for inputs in variants:
            with self.assertRaises(ValueError):
                model(**inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
