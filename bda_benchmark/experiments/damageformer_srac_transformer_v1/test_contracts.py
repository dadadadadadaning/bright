import unittest

import torch

from .build_neighbors import select_ids
from .model import (
    SRACTransformerFront,
    StructureTokenAdapter,
    SwinLocalSuperpixelAttention,
    project_region_probabilities,
    soft_assignment_boundary,
)
from .retrieval import JointMemoryReader
from .train_reader import _sample_query


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_local_attention_normalization_gradient_and_boundary(self):
        module = SwinLocalSuperpixelAttention(16, 16, stride=4, iterations=3)
        feature = torch.randn(2, 16, 8, 8, requires_grad=True)
        building = torch.rand(2, 1, 8, 8)
        result = module(feature, building)
        q, ids = result["q"], result["ids"]
        self.assertEqual(q.shape, (2, 9, 64))
        torch.testing.assert_close(q.sum(1), torch.ones(2, 64), atol=1e-6, rtol=1e-6)
        self.assertTrue((q >= 0).all())
        boundary = soft_assignment_boundary(q, ids, result["shape"])
        self.assertEqual(boundary.shape, (2, 1, 8, 8))
        self.assertTrue(((boundary >= 0) & (boundary <= 1)).all())
        (q.square().mean() + boundary.mean()).backward()
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)

    def test_structure_adapter_is_small_feature_update(self):
        adapter = StructureTokenAdapter(32, 16)
        feature = torch.randn(2, 32, 8, 8, requires_grad=True)
        building = torch.rand(2, 1, 8, 8)
        boundary = torch.rand_like(building)
        entropy = torch.rand_like(building)
        adapted, base = adapter(feature, building, boundary, entropy)
        self.assertEqual(adapted.shape, (2, 16, 8, 8))
        self.assertLess((adapted - base).abs().mean().item(), 0.01)
        adapted.square().mean().backward()
        self.assertIsNotNone(adapter.layer_scale.grad)
        self.assertGreater(feature.grad.abs().sum().item(), 0)

    def test_projection_is_probability_and_differentiable(self):
        module = SwinLocalSuperpixelAttention(8, 8, stride=4, iterations=1)
        feature = torch.randn(1, 8, 8, 8, requires_grad=True)
        result = module(feature, torch.rand(1, 1, 8, 8))
        logits = torch.randn(1, result["count"], 4, requires_grad=True)
        probability = project_region_probabilities(logits, result["q"], result["ids"], (8, 8), (16, 16))
        torch.testing.assert_close(probability.sum(1), torch.ones(1, 16, 16), atol=1e-6, rtol=1e-6)
        (-probability[:, 2].clamp_min(1e-8).log().mean()).backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertGreater(feature.grad.abs().sum().item(), 0)

    def test_reader_uses_direct_memory_and_safe_null_slots(self):
        reader = JointMemoryReader(16, num_events=3, slots=8).eval()
        query = torch.randn(5, 16)
        support = torch.randn(5, 8, 16)
        distribution = torch.rand(5, 8, 4).softmax(-1)
        valid = torch.ones(5, 8, dtype=torch.bool)
        metadata = dict(
            geometry=torch.rand(5, 8, 5),
            events=torch.zeros(5, 8, dtype=torch.long),
            disasters=torch.ones(5, 8, dtype=torch.long),
        )
        memory = reader(query, support, distribution, valid, mode="memory", metadata=metadata)
        nomemory = reader(query, mode="nomemory")
        self.assertEqual(memory.shape, (5, 4))
        self.assertFalse(torch.allclose(memory, nomemory))
        empty = torch.zeros_like(valid)
        a = reader(query, support, distribution, empty, mode="memory", metadata=metadata)
        b = reader(query, support + 1000, distribution.flip(-1), empty, mode="memory", metadata=metadata)
        torch.testing.assert_close(a, b)
        self.assertFalse(any("base" in name or "pixel_logits" in name for name, _ in reader.named_parameters()))

    def test_per_class_retrieval_has_equal_slots_without_query_labels(self):
        bank = dict(
            tokens=torch.eye(8),
            distribution=torch.eye(4).repeat_interleave(2, 0),
            events=torch.arange(8),
            images=torch.arange(8),
        )
        query = torch.eye(8)[:2]
        ids = select_ids(query, bank, event=-1, image=-1, topk=8, policy="per_class", exclusion="same_image")
        labels = bank["distribution"].argmax(-1)[ids]
        for row in labels:
            self.assertEqual(torch.bincount(row, minlength=4).tolist(), [2, 2, 2, 2])

    def test_class_balanced_query_sampling_returns_flat_indices(self):
        shard = {
            "weights": torch.ones(12),
            "distribution": torch.eye(4).repeat(3, 1),
        }
        ids = _sample_query(shard, 16, torch.Generator().manual_seed(3), "class_balanced")
        self.assertEqual(ids.shape, (16,))
        self.assertEqual(ids.dtype, torch.long)

    def test_full_front_preserves_a1_outputs_and_checkpoint_is_adapter_only(self):
        model = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1).eval()
        pre = torch.randn(1, 3, 64, 64)
        post = torch.randn_like(pre)
        with torch.no_grad():
            expected_loc, expected_pixel = model.base(pre, post)
            result = model(pre, post)
        torch.testing.assert_close(result["loc"], expected_loc)
        torch.testing.assert_close(result["old_pixel"], expected_pixel)
        self.assertEqual(result["prob"].shape, (1, 4, 64, 64))
        self.assertTrue(all(not name.startswith("base.") for name in model.adapter_state_dict()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
        self.assertFalse(model.base.training)

    def test_adapter_only_checkpoint_round_trip(self):
        source = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1)
        state = {name: value.detach().clone() for name, value in source.adapter_state_dict().items()}
        target = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1)
        target.load_adapter_state_dict(state)
        loaded = target.adapter_state_dict()
        self.assertEqual(set(loaded), set(state))
        for name in state:
            torch.testing.assert_close(loaded[name], state[name])


if __name__ == "__main__":
    unittest.main()

