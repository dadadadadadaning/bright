"""CPU numerical/training contracts; these are not BRIGHT validation results."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from residual_core import ARMS, DELTA_BOUND, make_heads, parameter_checks, prepare, probability
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
    feature_branches,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll


def fixture():
    torch.manual_seed(3407)
    feature = torch.randn(1, 256, 2, 3, requires_grad=True)
    q = torch.tensor([[[.8, .2, .7, .1, .4, .6], [.2, .8, .3, .9, .6, .4]]],
                     requires_grad=True)
    ids = torch.tensor([[0] * 6, [1] * 6], dtype=torch.long)
    head = nn.Conv2d(256, 4, 1)
    front = SimpleNamespace(base=SimpleNamespace(clf_2=head))
    result = dict(q=q, ids=ids, count=3, shape=(2, 3))
    return front, feature, result


class ResidualContracts(unittest.TestCase):
    def test_same_initialization_budget_and_no_parameter_aliases(self):
        front, _, _ = fixture()
        heads = make_heads(front)
        checks = parameter_checks(front, heads)
        self.assertTrue(checks["passed"], checks)
        self.assertEqual(checks["parameters_per_arm"], dict.fromkeys(ARMS, 33092))
        with torch.no_grad():
            heads["local_delta"].output.bias[0] = 1
        self.assertEqual(float(heads["region_delta"].output.bias[0]), 0.)
        self.assertFalse(parameter_checks(front, heads)["passed"])

    def test_initial_outputs_match_original_grid_probq_including_resize(self):
        front, feature, result = fixture()
        heads = make_heads(front)
        inputs = prepare(feature, result, front.base.clf_2)
        original, checks, _ = feature_branches(feature.detach(), result["q"].detach(),
                                               result["ids"], result["count"],
                                               front.base.clf_2, (7, 11))
        self.assertEqual(checks["empty_regions"], 1)
        for arm in ARMS:
            p = probability(arm, heads[arm], inputs, (7, 11))
            torch.testing.assert_close(p, original["grid_probq"], atol=3e-7, rtol=3e-6)
            torch.testing.assert_close(p.sum(1), torch.ones(1, 7, 11))
            self.assertTrue(bool((p >= 0).all()))

    def test_training_has_live_gradient_without_changing_frozen_inputs(self):
        front, feature, result = fixture()
        heads = make_heads(front)
        source_state = copy.deepcopy(front.base.clf_2.state_dict())
        feature_before, q_before = feature.detach().clone(), result["q"].detach().clone()
        inputs = prepare(feature, result, front.base.clf_2)
        target = torch.tensor([[[0, 1, 2], [3, 2, 255]]])
        for arm in ARMS:
            opt = torch.optim.AdamW(heads[arm].parameters(), lr=.01, weight_decay=0.)
            hidden_before = heads[arm].hidden.weight.detach().clone()
            for _ in range(2):
                opt.zero_grad()
                loss = region_nll(probability(arm, heads[arm], inputs, (2, 3)), target)
                loss.backward()
                self.assertGreater(float(heads[arm].output.weight.grad.abs().sum()), 0.)
                opt.step()
            self.assertFalse(torch.equal(hidden_before, heads[arm].hidden.weight))
        self.assertIsNone(feature.grad)
        self.assertIsNone(result["q"].grad)
        self.assertTrue(all(p.grad is None for p in front.base.clf_2.parameters()))
        torch.testing.assert_close(feature.detach(), feature_before, atol=0., rtol=0.)
        torch.testing.assert_close(result["q"].detach(), q_before, atol=0., rtol=0.)
        for name, value in source_state.items():
            torch.testing.assert_close(front.base.clf_2.state_dict()[name], value, atol=0., rtol=0.)

    def test_mixed_region_changes_context_and_outputs_with_matched_weights(self):
        front, feature, result = fixture()
        inputs = prepare(feature, result, front.base.clf_2)
        self.assertGreater(float((inputs["feature"] - inputs["mean_rebuilt"]).abs().mean()), .1)
        heads = make_heads(front)
        with torch.no_grad():
            heads["local_delta"].output.weight.normal_(0., .1)
        heads["region_delta"].load_state_dict(heads["local_delta"].state_dict())
        a = probability("local_delta", heads["local_delta"], inputs, (2, 3))
        b = probability("region_delta", heads["region_delta"], inputs, (2, 3))
        self.assertGreater(float((a - b).abs().max()), 1e-4)

    def test_off_is_baseline_and_bound_holds_after_large_weights(self):
        front, feature, result = fixture()
        heads = make_heads(front)
        inputs = prepare(feature, result, front.base.clf_2)
        with torch.no_grad():
            heads["region_delta"].output.weight.fill_(100.)
            heads["region_delta"].output.bias.copy_(torch.tensor([100., -100., 10., -10.]))
        delta = heads["region_delta"](inputs["feature"], inputs["mean_rebuilt"])
        self.assertLessEqual(float(delta.abs().max()), DELTA_BOUND)
        p = probability("region_delta", heads["region_delta"], inputs, (5, 8))
        self.assertTrue(bool(torch.isfinite(p).all()))
        torch.testing.assert_close(p.sum(1), torch.ones(1, 5, 8))

    def test_invalid_q_features_and_parent_fail_without_repair(self):
        front, feature, result = fixture()
        invalid = dict(result, q=result["q"] * .5)
        with self.assertRaisesRegex(ValueError, "normalized"):
            prepare(feature, invalid, front.base.clf_2)
        damaged = feature.detach().clone()
        damaged[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            prepare(damaged, result, front.base.clf_2)
        with self.assertRaisesRegex(ValueError, "original FP32"):
            prepare(feature[:, :128], result, nn.Conv2d(128, 4, 1))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            probability("wrong", make_heads(front)["local_delta"], {}, (2, 3))

    def test_ignore_255_has_zero_direct_loss_gradient(self):
        logits = torch.randn(1, 4, 2, 3, requires_grad=True)
        probability_values = logits.softmax(1)
        target = torch.tensor([[[0, 1, 2], [3, 2, 255]]])
        region_nll(probability_values, target).backward()
        torch.testing.assert_close(logits.grad[:, :, 1, 2], torch.zeros(1, 4), atol=0., rtol=0.)
        self.assertGreater(float(logits.grad[:, :, 0].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
