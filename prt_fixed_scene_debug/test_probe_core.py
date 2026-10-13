import copy
import json
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

import probe_core
from probe_core import run_fixed_probe


class ManualRouting:
    builds = 0

    def __init__(self, ids, count):
        self.ids, self.count = ids.clone(), count
        ManualRouting.builds += 1


def manual_pool(feature, q, ids, count, *, routing):
    if routing.count != count or not torch.equal(routing.ids, ids):
        raise ValueError("Wrong route plan")
    batch, channels, _, _ = feature.shape
    mean = feature.new_zeros(batch, count, channels)
    mass = feature.new_zeros(batch, count)
    flat = feature.flatten(2)
    for region in range(count):
        slot, pixel = (ids == region).nonzero(as_tuple=True)
        weights = q[:, slot, pixel]
        mass[:, region] = weights.sum(-1)
        numerator = (flat[:, :, pixel] * weights[:, None]).sum(-1)
        denominator = torch.where(mass[:, region] > 0, mass[:, region], torch.ones_like(mass[:, region]))
        mean[:, region] = numerator / denominator[:, None]
    return mean, mass


def scene_fixture():
    torch.manual_seed(33)
    head = nn.Conv2d(3, 4, 1)
    feature = torch.rand(1, 3, 2, 3)
    q = torch.tensor([[[.2, .4, .6, .8, .3, .7], [.8, .6, .4, .2, .7, .3]]])
    ids = torch.tensor([[0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0]])
    mean, mass = manual_pool(feature, q, ids, 2, routing=ManualRouting(ids, 2))
    with torch.no_grad():
        p_logits = head(feature)
        r_logits = F.linear(mean, head.weight[:, :, 0, 0], head.bias)
        probability = F.interpolate(p_logits.softmax(1), (4, 6), mode="bilinear", align_corners=False)
        probability /= probability.sum(1, keepdim=True)
    return dict(image="fixture", count=2, shape=(2, 3), output_size=(4, 6),
                parent_classifier_state={key: value.detach().clone() for key, value in head.state_dict().items()},
                private_heads_state={arm + "." + key: value.detach().clone()
                    for arm in ("p", "r") for key, value in head.state_dict().items()},
                tensors=dict(feature=feature, q=q, ids=ids,
                    original_mean_tokens=mean, recomputed_mean_tokens=mean.clone(),
                    original_mass=mass, recomputed_mass=mass.clone(),
                    p_original_logits=p_logits, p_private_logits=p_logits.clone(),
                    r_original_logits=r_logits, r_private_logits=r_logits.clone(),
                    pixel_grid=probability, p=probability.clone()))


class FixedSceneTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene_fixture()
        ManualRouting.builds = 0

    def test_real_affine_calls_manual_pool_and_fixed_counts(self):
        calls = []
        def counted(*args, **kwargs):
            calls.append(kwargs["routing"])
            return manual_pool(*args, **kwargs)
        report = run_fixed_probe(self.scene, "cpu", counted, ManualRouting)
        self.assertEqual(report["operator_calls"], {"p_conv": 6, "r_linear": 12, "mean_pool": 6})
        self.assertEqual(len(report["p"]), 6)
        self.assertEqual([len(v) for v in report["r"].values()], [6, 6])
        self.assertEqual(len(calls), 6)
        self.assertEqual(len({id(plan) for plan in calls[:3]}), 1)
        self.assertEqual(len({id(plan) for plan in calls[3:]}), 3)
        self.assertEqual(ManualRouting.builds, 4)
        self.assertTrue(report["fixed_inputs_and_heads_unchanged"])
        self.assertFalse(report["full_validation_evidence"])
        self.assertFalse(report["execution_grad_enabled"])
        self.assertIn("detached state_dict metadata", report["parameter_policy"])
        for role, state in report["head_layout"].items():
            self.assertTrue(all(value["requires_grad"] is (role != "parent") for value in state.values()))
        for call in report["p"]:
            self.assertTrue(call["vs_first_parent"]["passed"])
            self.assertFalse(call["logits_requires_grad"])
        for calls in report["r"].values():
            self.assertTrue(all(call["logits_requires_grad"] is False for call in calls))
        for call in report["pool"]["shared_plan"]:
            self.assertTrue(call["mean_vs_first_pool"]["passed"])
            self.assertNotIn("logit_argmax_differences", call["mean_vs_first_pool"])
        json.dumps(report, allow_nan=False)

    def test_cpu_references_and_counterfactual_mean_change(self):
        self.scene["tensors"]["p_original_logits"][0, 2, 1, 2] += .2
        self.scene["tensors"]["original_mean_tokens"][0, 1, 2] += .1
        report = run_fixed_probe(self.scene, "cpu", manual_pool, ManualRouting)
        p = report["p_cpu_fp64_reference"]
        self.assertEqual(p["index"], [0, 2, 1, 2])
        self.assertGreater(p["original_abs_error"], .199)
        self.assertLess(p["private_abs_error"], 1e-6)
        self.assertIn("dot_abs_sum", p)
        self.assertIn("condition_ratio", p)
        for call in report["p"]:
            self.assertEqual(call["at_saved_worst_point"]["index"], p["index"])
            self.assertLess(call["at_saved_worst_point"]["abs_error_vs_fp64"], 1e-6)
        mean = report["mean_cpu_fp64_reference"]
        self.assertEqual(mean["index"], [0, 1, 2])
        self.assertGreater(mean["route_occurrences"], 0)
        self.assertGreater(mean["original_abs_error"], .099)
        self.assertLess(mean["recomputed_abs_error"], 1e-6)
        for calls in report["pool"].values():
            for call in calls:
                self.assertEqual(call["at_saved_worst_point"]["index"], mean["index"])
                self.assertLess(call["at_saved_worst_point"]["abs_error_vs_fp64"], 1e-6)
        self.assertFalse(report["r_input_effect"]["mean_difference"]["passed"])
        for case in report["r"].values():
            for call in case:
                self.assertTrue(call["vs_first_parent_same_input"]["passed"])

    def test_saved_cpu_scene_bytes_do_not_change(self):
        before = {key: value.clone() for key, value in self.scene["tensors"].items()}
        run_fixed_probe(self.scene, "cpu", manual_pool, ManualRouting)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.scene["tensors"][key]), key)

    def test_unequal_weight_rejected_before_any_pool(self):
        self.scene["private_heads_state"]["p.weight"][0, 0, 0, 0] += 1
        with mock.patch("probe_core.nn.Conv2d") as constructor:
            with self.assertRaisesRegex(ValueError, "Copied head differs"):
                run_fixed_probe(self.scene, "cpu", manual_pool, ManualRouting)
            constructor.assert_not_called()
        self.assertEqual(ManualRouting.builds, 0)

    def test_nonfinite_input_rejected_before_any_operator(self):
        self.scene["tensors"]["feature"][0, 0, 0, 0] = float("nan")
        with mock.patch("probe_core.nn.Conv2d") as constructor:
            with self.assertRaisesRegex(ValueError, "finite validated CPU"):
                run_fixed_probe(self.scene, "cpu", manual_pool, ManualRouting)
            constructor.assert_not_called()

    def test_pool_error_propagates_without_after_hash_or_retry(self):
        observed = {}
        with mock.patch("probe_core._fixed_hash", wraps=probe_core._fixed_hash) as hashing:
            def fail(*args, **kwargs):
                observed["hashes"] = hashing.call_count
                raise RuntimeError("primary CUDA operator failure")
            with self.assertRaisesRegex(RuntimeError, "primary CUDA operator failure"):
                run_fixed_probe(self.scene, "cpu", fail, ManualRouting)
            self.assertEqual(hashing.call_count, observed["hashes"])

    def test_mutating_operator_fails_input_hash_and_preserves_old_scene(self):
        before = self.scene["tensors"]["feature"].clone()
        def mutating(feature, *args, **kwargs):
            result = manual_pool(feature, *args, **kwargs)
            feature.add_(.1)
            return result
        with self.assertRaisesRegex(RuntimeError, "Fixed input/classifier state changed"):
            run_fixed_probe(self.scene, "cpu", mutating, ManualRouting)
        self.assertTrue(torch.equal(before, self.scene["tensors"]["feature"]))

    def test_cannot_repeat_until_pass(self):
        for repeats in (1, 2, 4, True):
            with self.assertRaisesRegex(ValueError, "Exactly three"):
                run_fixed_probe(self.scene, "cpu", manual_pool, ManualRouting, repeats=repeats)


if __name__ == "__main__":
    unittest.main()
