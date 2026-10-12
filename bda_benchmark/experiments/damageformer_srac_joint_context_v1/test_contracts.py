import copy
import unittest

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import SRACTransformerFront

from .common import FrozenFrontCapture
from .model import ContextMemoryReader, JointContextDecoder, VARIANTS, unpool_region_tokens


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)

    @staticmethod
    def association():
        # Four pixels, two candidates per pixel, three global regions.
        ids = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
        q = torch.tensor([[[0.75, 0.50, 0.25, 0.00], [0.25, 0.50, 0.75, 1.00]]])
        return q, ids

    def test_unpool_uses_same_normalized_q_and_is_differentiable(self):
        q, ids = self.association()
        tokens = torch.randn(1, 3, 5, requires_grad=True)
        context = unpool_region_tokens(tokens, q, ids, (2, 2))
        expected_first = 0.75 * tokens[0, 0] + 0.25 * tokens[0, 1]
        torch.testing.assert_close(context[0, :, 0, 0], expected_first)
        context.square().mean().backward()
        self.assertGreater(tokens.grad.abs().sum().item(), 0)

    def test_joint_head_is_exact_a1_identity_at_step_zero(self):
        a1 = nn.Conv2d(256, 4, 1)
        decoder = JointContextDecoder(a1, dim=16, num_events=3, slots=4).eval()
        pixel = torch.randn(1, 256, 2, 2)
        tokens = torch.randn(1, 3, 16)
        q, ids = self.association()
        expected = a1(pixel)
        for variant in VARIANTS:
            kwargs = {}
            if variant == "j2":
                kwargs = dict(
                    support=torch.randn(3, 4, 16),
                    distribution=torch.rand(3, 4, 4).softmax(-1),
                    valid=torch.ones(3, 4, dtype=torch.bool),
                    metadata=dict(
                        geometry=torch.rand(3, 4, 5),
                        events=torch.zeros(3, 4, dtype=torch.long),
                        disasters=torch.ones(3, 4, dtype=torch.long),
                    ),
                )
            actual = decoder(pixel, tokens, q, ids, (2, 2), variant, **kwargs)["logits"]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(decoder.classifier.context.weight)), 0)

    def test_reader_returns_enhanced_tokens_not_logits(self):
        reader = ContextMemoryReader(16, num_events=3, heads=4, slots=4).eval()
        query = torch.randn(5, 16)
        support = torch.randn(5, 4, 16)
        distribution = torch.rand(5, 4, 4).softmax(-1)
        valid = torch.ones(5, 4, dtype=torch.bool)
        metadata = dict(
            geometry=torch.rand(5, 4, 5),
            events=torch.zeros(5, 4, dtype=torch.long),
            disasters=torch.ones(5, 4, dtype=torch.long),
        )
        memory = reader(query, support, distribution, valid, "memory", metadata)
        nomemory = reader(query, mode="nomemory")
        self.assertEqual(memory.shape, query.shape)
        self.assertEqual(nomemory.shape, query.shape)
        self.assertFalse(torch.allclose(memory, nomemory))
        self.assertFalse(any("classifier" in name for name, _ in reader.named_parameters()))

    def test_invalid_memory_is_replaced_by_learned_slots(self):
        reader = ContextMemoryReader(16, num_events=2, heads=4, slots=4).eval()
        query = torch.randn(3, 16)
        distribution = torch.rand(3, 4, 4).softmax(-1)
        invalid = torch.zeros(3, 4, dtype=torch.bool)
        metadata = dict(
            geometry=torch.rand(3, 4, 5),
            events=torch.zeros(3, 4, dtype=torch.long),
            disasters=torch.zeros(3, 4, dtype=torch.long),
        )
        a = reader(query, torch.randn(3, 4, 16), distribution, invalid, "memory", metadata)
        b = reader(query, torch.randn(3, 4, 16) + 999, distribution.flip(-1), invalid, "memory", metadata)
        torch.testing.assert_close(a, b)

    def test_variants_have_identical_parameterization_and_independent_storage(self):
        a1 = nn.Conv2d(256, 4, 1)
        template = JointContextDecoder(a1, dim=16, num_events=2, slots=4)
        models = [copy.deepcopy(template) for _ in VARIANTS]
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)), 1)
        first = dict(models[0].named_parameters())
        second = dict(models[1].named_parameters())
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name])
            self.assertNotEqual(first[name].data_ptr(), second[name].data_ptr())

    def test_region_branch_gets_gradient_after_context_columns_open(self):
        a1 = nn.Conv2d(256, 4, 1)
        decoder = JointContextDecoder(a1, dim=16, num_events=2, slots=4)
        with torch.no_grad():
            decoder.classifier.context.weight.normal_(std=0.01)
        pixel = torch.randn(1, 256, 2, 2)
        tokens = torch.randn(1, 3, 16)
        q, ids = self.association()
        logits = decoder(pixel, tokens, q, ids, (2, 2), "j1")["logits"]
        logits.square().mean().backward()
        reader_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in decoder.reader.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(reader_grad, 0)

    def test_real_front_capture_and_j0_reproduce_low_resolution_a1(self):
        front = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1).eval()
        capture = FrozenFrontCapture(front)
        pre = torch.randn(1, 3, 64, 64)
        post = torch.randn_like(pre)
        result, pixel = capture(pre, post)
        decoder = JointContextDecoder(front.base.clf_2, dim=16, num_events=2, slots=4).eval()
        actual = decoder(
            pixel,
            result["tokens"],
            result["q"],
            result["ids"],
            result["shape"],
            "j0",
        )["logits"]
        expected = front.base.clf_2(pixel)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(all(not parameter.requires_grad for parameter in front.parameters()))
        capture.close()


if __name__ == "__main__":
    unittest.main()

