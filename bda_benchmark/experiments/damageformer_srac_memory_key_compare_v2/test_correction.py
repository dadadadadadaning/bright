"""CPU contracts for identity, support fallback and matched learning."""
import argparse
import copy

import unittest
import torch
from torch.nn import functional as F

from .correction import (PixelMemoryCorrection, TRAIN_ARMS, correction_inputs,
                         make_heads, assert_evaluation)
from .train_correction import matched_update, validate_config, validate_trainable_heads


def inputs(seed=11):
    generator = torch.Generator().manual_seed(seed)
    return dict(feature=torch.randn(1, 256, 3, 4, generator=generator),
                base_probability=torch.randn(1, 4, 6, 8, generator=generator).softmax(1),
                evidence_grid=torch.randn(1, 10, 3, 4, generator=generator),
                coverage=torch.ones(1, 1, 3, 4))


class CorrectionTests(unittest.TestCase):
    def test_zero_initialization_is_exact_and_has_useful_delta_gradient(self):
        head = PixelMemoryCorrection(hidden=8)
        item = inputs()
        row = head(**item)
        assert torch.equal(row["probability"], item["base_probability"])
        assert torch.equal(row["gate"], torch.full_like(row["gate"], .5))
        target = torch.full((1, 6, 8), 2, dtype=torch.long)
        loss = F.nll_loss(row["probability"].clamp_min(1e-8).log(), target)
        loss.backward()
        assert head.delta.weight.grad is not None
        assert head.delta.weight.grad.abs().sum() > 0
        assert torch.isfinite(head.delta.weight.grad).all()
        with torch.no_grad():
            assert head(**item)["probability"] is item["base_probability"]


    def test_empty_and_off_support_return_original_probability_even_after_training(self):
        head = PixelMemoryCorrection(hidden=8)
        with torch.no_grad():
            head.delta.bias.copy_(torch.tensor([2., -1., 3., -2.]))
            head.gate.bias.fill_(3.)
        item = inputs()
        assert head(**item, force_off=True)["probability"] is item["base_probability"]
        item["coverage"].zero_()
        assert head(**item)["probability"] is item["base_probability"]


    def test_bound_normalization_and_pixel_support_fallback(self):
        head = PixelMemoryCorrection(hidden=8, bound=2.)
        with torch.no_grad():
            head.delta.bias.copy_(torch.tensor([100., -100., 20., -20.]))
            head.gate.bias.fill_(100.)
        item = inputs()
        item["coverage"][:, :, :, :2] = 0
        row = head(**item)
        assert row["delta"].abs().max() <= 2.
        assert torch.allclose(row["probability"].sum(1), torch.ones(1, 6, 8), atol=2e-6)
        assert torch.isfinite(row["probability"]).all()
        assert (row["probability"] >= 0).all()
        assert torch.equal(row["probability"][:, :, :, :2], item["base_probability"][:, :, :, :2])
        assert not torch.equal(row["probability"], item["base_probability"])


    def test_unpool_evidence_and_query_control_share_f256_coverage(self):
        feature = torch.randn(1, 256, 1, 2)
        base = torch.randn(1, 4, 2, 4).softmax(1)
        result = dict(shape=(1, 2), count=2, q=torch.ones(1, 1, 2), ids=torch.tensor([[0, 1]]))
        evidence = dict(posterior=torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.]]),
                        quality=torch.ones(2, 6), supported=torch.tensor([True, False]))
        normal = correction_inputs(feature, result, base, evidence)
        query = correction_inputs(feature, result, base, evidence, query_only=True)
        assert normal["evidence_grid"].shape == (1, 10, 1, 2)
        assert not query["evidence_grid"].any()
        assert torch.equal(normal["coverage"], query["coverage"])
        assert query["coverage"].flatten().tolist() == [1., 0.]
        assert query["base_probability"] is base
        assert all(not tensor.requires_grad for tensor in query.values())


    def test_matched_updates_are_finite_and_improve_a_supervised_example(self):
        torch.manual_seed(22)
        heads = make_heads(hidden=8)
        optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=.02, weight_decay=0)
                      for arm in TRAIN_ARMS}
        counts = validate_trainable_heads(heads, optimizers)
        assert len(set(counts.values())) == 1
        item = inputs()
        arm_inputs = {arm: copy.deepcopy(item) for arm in TRAIN_ARMS}
        arm_inputs["query_only"]["evidence_grid"].zero_()
        target = torch.full((1, 6, 8), 2, dtype=torch.long)
        def loss_fn(probability, labels):
            return F.nll_loss(probability.clamp_min(1e-8).log(), labels)
        initial = {arm: float(loss_fn(heads[arm](**arm_inputs[arm])["probability"], target)) for arm in TRAIN_ARMS}
        for _ in range(3):
            logs = matched_update(heads, optimizers, arm_inputs, target, loss_fn=loss_fn)
            assert all(torch.isfinite(torch.tensor(row["loss"])) and row["gradient_norm"] > 0 for row in logs.values())
        for arm in TRAIN_ARMS:
            current = heads[arm](**arm_inputs[arm])["probability"]
            assert float(loss_fn(current, target)) < initial[arm]
            assert heads[arm].gate.weight.grad.abs().sum() > 0
        assert not torch.equal(heads["query_only"].delta.weight, heads["f256_memory"].delta.weight)


    def test_failed_later_arm_advances_no_optimizer(self):
        heads = make_heads(hidden=8)
        optimizers = {arm: torch.optim.Adam(heads[arm].parameters(), lr=.01) for arm in TRAIN_ARMS}
        item = inputs()
        arm_inputs = {arm: copy.deepcopy(item) for arm in TRAIN_ARMS}
        arm_inputs["f256_memory"]["feature"][0, 0, 0, 0] = float("nan")
        before = {name: value.clone() for name, value in heads.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            matched_update(heads, optimizers, arm_inputs, torch.zeros(1, 6, 8, dtype=torch.long),
                           loss_fn=lambda probability, target: -probability[:, 0].clamp_min(1e-8).log().mean())
        assert all(torch.equal(value, heads.state_dict()[name]) for name, value in before.items())
        assert all(not optimizer.state for optimizer in optimizers.values())


    def test_detached_inputs_and_private_optimizer_required(self):
        heads = make_heads(hidden=8)
        optimizers = {arm: torch.optim.Adam(heads[arm].parameters(), lr=.01) for arm in TRAIN_ARMS}
        optimizers["t128_memory"] = optimizers["query_only"]
        with self.assertRaisesRegex(ValueError, "Optimizer"):
            validate_trainable_heads(heads, optimizers)
        item = inputs()
        item["feature"].requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached"):
            matched_update(heads, optimizers, {arm: item for arm in TRAIN_ARMS}, torch.zeros(1, 6, 8, dtype=torch.long))


    def test_fixed_budget_and_formal_reproduction_cannot_be_skipped(self):
        cli = argparse.Namespace(smoke=False, steps=4999, lr=1e-4, hidden=8,
                                 bound=2., temperature=.1, topk=8)
        with self.assertRaisesRegex(ValueError, "5000"):
            validate_config(cli)
        cli.smoke, cli.steps = True, 5000
        assert validate_config(cli) == 20
        report = dict(complete=True, immutable={"source": True}, native_a1_replay_ok=True,
                      pool_ok=True, image_count=450, full_validation_evidence=False,
                      reproduction={"passed": False}, test_accessed=False, smoke=False)
        with self.assertRaisesRegex(ValueError, "Val450"):
            assert_evaluation(report)


if __name__ == "__main__":
    unittest.main()
