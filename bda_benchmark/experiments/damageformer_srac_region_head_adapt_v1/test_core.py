"""Numerical semantics and isolation of the new trainable heads."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from .core import (assert_formal_report, head_probability, initial_parameter_checks,
                   make_heads, prepare_inputs)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import feature_branches
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import project_region_probabilities


def fixture():
    torch.manual_seed(39)
    front = SimpleNamespace(base=SimpleNamespace(clf_2=nn.Conv2d(256, 4, 1)),
                            region_classifier=nn.Linear(128, 4))
    front.base.clf_2.requires_grad_(False)
    front.region_classifier.requires_grad_(False)
    feature = torch.randn(1, 256, 2, 3, requires_grad=True)
    # Repeated global IDs in separate slots must accumulate normally.
    ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]])
    q = torch.rand(1, 2, 6)
    q = (q / q.sum(1, keepdim=True)).requires_grad_(True)
    result = dict(q=q, ids=ids, count=4, shape=(2, 3),
                  tokens=torch.randn(1, 4, 128, requires_grad=True))
    return front, feature, result


class HeadCoreTests(unittest.TestCase):
    def test_initial_p_r_match_original_probe_formulas(self):
        front, feature, result = fixture()
        heads = make_heads(front)
        self.assertTrue(initial_parameter_checks(front, heads)["passed"])
        inputs = prepare_inputs(feature, result)
        expected, checks, _ = feature_branches(feature.detach(), result["q"].detach(),
            result["ids"], result["count"], front.base.clf_2, (5, 7))
        for arm, reference in (("p", "pixel_grid"), ("r", "region_mean")):
            actual = head_probability(arm, heads[arm], inputs, (5, 7))
            torch.testing.assert_close(actual, expected[reference], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(actual.sum(1), torch.ones(1, 5, 7))
        self.assertTrue(checks["affine_commutation"]["passed"])

    def test_t_is_original_region_classifier_readout(self):
        front, feature, result = fixture()
        heads = make_heads(front)
        inputs = prepare_inputs(feature, result)
        old = project_region_probabilities(front.region_classifier(result["tokens"]),
            result["q"], result["ids"], result["shape"], (5, 7))
        torch.testing.assert_close(head_probability("t", heads["t"], inputs, (5, 7)), old)

    def test_r_backward_updates_head_not_feature_q_or_other_heads(self):
        front, feature, result = fixture()
        heads = make_heads(front)
        inputs = prepare_inputs(feature, result)
        p = head_probability("r", heads["r"], inputs, (2, 3))
        loss = -p[:, 2].clamp_min(1e-8).log().mean()
        loss.backward()
        self.assertGreater(float(heads["r"].weight.grad.abs().sum()), 0)
        self.assertIsNone(feature.grad)
        self.assertIsNone(result["q"].grad)
        self.assertIsNone(result["tokens"].grad)
        for module in (heads["p"], heads["t"], front.base.clf_2):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_copies_do_not_alias_and_original_remains_frozen(self):
        front, _, _ = fixture()
        before = copy.deepcopy(front.base.clf_2.state_dict())
        heads = make_heads(front)
        with torch.no_grad():
            heads["p"].weight.add_(1)
        torch.testing.assert_close(heads["r"].weight, before["weight"], rtol=0, atol=0)
        torch.testing.assert_close(front.base.clf_2.weight, before["weight"], rtol=0, atol=0)
        self.assertFalse(front.base.clf_2.weight.requires_grad)
        self.assertFalse(initial_parameter_checks(front, heads)["passed"])

    def test_empty_region_does_not_break_projection(self):
        front, feature, result = fixture()
        inputs = prepare_inputs(feature, result)
        self.assertEqual(float(inputs["mass"][0, 3]), 0)
        for arm, head in make_heads(front).items():
            self.assertTrue(bool(torch.isfinite(head_probability(arm, head, inputs, (2, 3))).all()))

    def test_formal_gate_never_promotes_smoke_or_failed_reproduction(self):
        record = dict(complete=True, image_count=450, full_validation_evidence=True,
                      immutable={"models": True}, native_a1_replay_ok=True,
                      reproduction={"passed": True})
        assert_formal_report(record)
        for change in ({"image_count": 2}, {"reproduction": {"passed": False}},
                       {"immutable": {}}, {"native_a1_replay_ok": False}):
            with self.assertRaises(ValueError):
                assert_formal_report({**record, **change})
        debug = {**record, "image_count": 2, "full_validation_evidence": False,
                 "reproduction": {"passed": None}}
        assert_formal_report(debug, smoke=True)
        with self.assertRaises(ValueError):
            assert_formal_report(debug)


if __name__ == "__main__":
    unittest.main()
