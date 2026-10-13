"""CPU contracts for fresh paired keys and post-hoc label evidence."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from . import paired


def fixture():
    q = torch.tensor([[[0.6, 0.2, 0.4, 0.7],
                       [0.3, 0.5, 0.1, 0.2],
                       [0.1, 0.3, 0.5, 0.1]]], dtype=torch.float32)
    # Candidate occurrences include duplicate region 0 at pixel 0 and region 2
    # at pixel 3. Region 3 is deliberately unused.
    ids = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 2], [1, 2, 0, 1]])
    mass = torch.zeros(1, 4)
    for slot in range(3):
        for pixel in range(4):
            mass[0, ids[slot, pixel]] += q[0, slot, pixel]
    result = dict(q=q, ids=ids, count=4, shape=(2, 2),
                  tokens=torch.arange(512, dtype=torch.float32).reshape(1, 4, 128)+1,
                  mass=mass, valid_regions=torch.tensor([[True, True, False, False]]))
    feature = torch.arange(1, 1025, dtype=torch.float32).reshape(1, 256, 2, 2)
    target = torch.tensor([[[0, 1], [2, 3]]])
    return result, feature, target


def dense_reference(result, feature, target):
    """Explicit KxN mathematical operator independent of production helpers."""
    k, n = result["count"], result["q"].shape[-1]
    routes = torch.zeros(k, n, dtype=torch.float64)
    for slot in range(result["ids"].shape[0]):
        for pixel in range(n):
            routes[result["ids"][slot, pixel], pixel] += result["q"][0, slot, pixel].double()
    mass = routes.sum(-1)
    f = feature[0].flatten(1).T.double()
    keys = (routes @ f) / torch.where(mass > 0, mass, torch.ones_like(mass))[:, None]
    valid = (target >= 0) & (target < 4)
    labels = F.one_hot(target.clamp(0, 3), 4).permute(0, 3, 1, 2).double() * valid[:, None]
    fractions = F.interpolate(labels, result["shape"], mode="area")[0].flatten(1).T
    counts = routes @ fractions
    label_mass = counts.sum(-1)
    distribution = counts / torch.where(label_mass > 0, label_mass,
                                        torch.ones_like(label_mass))[:, None]
    return keys, mass, distribution, label_mass


class PairedTests(unittest.TestCase):
    def test_current_forward_has_aligned_actual_fp32_keys(self):
        result, feature, target = fixture()
        out = paired.extract_pair(result, feature, target)
        self.assertEqual(out["t128"].shape, (4, 128))
        self.assertEqual(out["f256"].shape, (4, 256))
        self.assertEqual(out["t128"].dtype, torch.float32)
        self.assertEqual(out["f256"].dtype, torch.float32)
        self.assertTrue(torch.equal(out["t128"], result["tokens"][0]))
        self.assertEqual(out["active"].tolist(), [True, True, False, False])
        self.assertTrue(out["q_check"]["passed"])
        self.assertTrue(out["label_check"]["passed"])

    def test_fractional_duplicate_routes_agree_with_dense_fp64_reference(self):
        result, feature, target = fixture()
        out = paired.extract_pair(result, feature, target)
        expected = dense_reference(result, feature, target)
        for name, reference in zip(("f256", "mass", "distribution", "label_mass"), expected):
            self.assertTrue(torch.allclose(out[name].double(), reference, atol=2e-5, rtol=2e-5), name)
        report = out["label_check"]
        self.assertEqual(report["reference_dtype"], "torch.float64")
        self.assertEqual(report["reference_device"], "cpu")
        self.assertEqual(report["routes_checked"], 12)
        self.assertEqual(report["regions_checked"], 4)

    def test_empty_mass_and_empty_labels_are_exact_zero(self):
        result, feature, target = fixture()
        out = paired.extract_pair(result, feature, torch.full_like(target, 255))
        self.assertEqual(float(out["mass"][3]), 0)
        self.assertTrue(torch.equal(out["f256"][3], torch.zeros(256)))
        self.assertTrue(torch.equal(out["label_mass"], torch.zeros(4)))
        self.assertTrue(torch.equal(out["distribution"], torch.zeros(4, 4)))
        self.assertTrue(out["label_check"]["passed"])

    def test_tiny_positive_mass_is_normalized_without_epsilon_clamp(self):
        result, _, _ = fixture()
        result.update(q=torch.tensor([[[1.0], [1e-8]]]), ids=torch.tensor([[0], [1]]),
                      count=2, shape=(1, 1), tokens=torch.ones(1, 2, 128),
                      mass=torch.tensor([[1.0, 1e-8]]), valid_regions=torch.ones(1, 2, dtype=torch.bool))
        out = paired.extract_pair(result, torch.ones(1, 256, 1, 1)*7,
                                  torch.tensor([[[2]]]))
        self.assertTrue(torch.equal(out["f256"], torch.full((2, 256), 7.0)))
        self.assertEqual(out["distribution"].tolist(), [[0, 0, 1, 0], [0, 0, 1, 0]])
        self.assertAlmostEqual(float(out["label_mass"][1]), 1e-8, places=14)
        self.assertTrue(out["active"].all())

    def test_invalid_labels_contribute_no_mass(self):
        result, feature, target = fixture()
        target[0, 0, 1] = -1
        target[0, 1, 1] = 255
        out = paired.extract_pair(result, feature, target)
        expected = dense_reference(result, feature, target)
        self.assertTrue(torch.allclose(out["distribution"].double(), expected[2], atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(out["label_mass"].double(), expected[3], atol=2e-5, rtol=2e-5))
        self.assertEqual(out["label_check"]["invalid_target_pixels"], 2)

    def test_area_resizing_fractional_labels_matches_dense_reference(self):
        result, feature, _ = fixture()
        target = torch.tensor([[[0, 0, 1, 255], [1, 2, 2, 3],
                                [3, -1, 0, 1], [2, 3, 2, 1]]])
        out = paired.extract_pair(result, feature, target)
        expected = dense_reference(result, feature, target)
        self.assertTrue(torch.allclose(out["distribution"].double(), expected[2], atol=2e-5, rtol=2e-5))
        self.assertTrue(out["label_check"]["passed"])

    def test_changing_ground_truth_cannot_change_keys_mass_or_active(self):
        result, feature, target = fixture()
        no_labels = paired.extract_pair(result, feature)
        first = paired.extract_pair(result, feature, target)
        second = paired.extract_pair(result, feature, (target+1) % 4)
        for name in ("t128", "f256", "mass", "active"):
            self.assertTrue(torch.equal(no_labels[name], first[name]), name)
            self.assertTrue(torch.equal(first[name], second[name]), name)
        self.assertNotIn("distribution", no_labels)
        self.assertFalse(torch.equal(first["distribution"], second["distribution"]))

    def test_posthoc_labels_do_not_pool_f256_or_forward_model(self):
        result, _, target = fixture()
        original = paired.mean_pool
        widths = []
        def tracked_pool(feature, *args, **kwargs):
            widths.append(feature.shape[1])
            return original(feature, *args, **kwargs)
        with mock.patch.object(paired, "mean_pool", side_effect=tracked_pool):
            out = paired.extract_labels(target, result)
        self.assertEqual(widths, [4])
        self.assertTrue(out["label_check"]["passed"])

    def test_pair_labels_reuse_the_current_route_plan(self):
        result, feature, target = fixture()
        with mock.patch.object(paired, "GroupedRouting", wraps=paired.GroupedRouting) as factory:
            paired.extract_pair(result, feature, target)
        self.assertEqual(factory.call_count, 1)

    def test_reference_is_independent_of_grouped_pool(self):
        result, feature, target = fixture()
        out = paired.extract_pair(result, feature, target)
        with mock.patch.object(paired, "mean_pool", side_effect=AssertionError("shared pool used")), \
             mock.patch.object(paired, "GroupedRouting", side_effect=AssertionError("shared route used")):
            report = paired.validate_label_reference(target, result, out["distribution"], out["label_mass"])
        self.assertTrue(report["passed"])

    def test_all_256_feature_channels_match_independent_random_dense_reference(self):
        result, _, target = fixture()
        generator = torch.Generator().manual_seed(3407)
        feature = torch.randn(1, 256, 2, 2, generator=generator)
        out = paired.extract_pair(result, feature)
        with mock.patch.object(paired, "mean_pool", side_effect=AssertionError("shared pool used")), \
             mock.patch.object(paired, "GroupedRouting", side_effect=AssertionError("shared route used")):
            reference = dense_reference(result, feature, target)[0]
        self.assertTrue(torch.allclose(out["f256"].double(), reference, atol=2e-5, rtol=2e-5))

    def test_nondivisible_area_resize_matches_independent_reference(self):
        result, feature, _ = fixture()
        target = torch.arange(63).reshape(1, 7, 9) % 4
        target[0, 2:4, 3:5] = 255
        out = paired.extract_pair(result, feature, target)
        reference = dense_reference(result, feature, target)
        self.assertTrue(torch.allclose(out["distribution"].double(), reference[2], atol=2e-5, rtol=2e-5))
        self.assertTrue(out["label_check"]["passed"])

    def test_double_q_does_not_change_actual_fp32_key_output(self):
        result, feature, _ = fixture()
        expected = paired.extract_pair(result, feature)
        result["q"] = result["q"].double()
        actual = paired.extract_pair(result, feature)
        self.assertEqual(actual["f256"].dtype, torch.float32)
        self.assertTrue(torch.equal(expected["f256"], actual["f256"]))

    def test_reference_detects_distribution_and_mass_corruption(self):
        result, feature, target = fixture()
        out = paired.extract_pair(result, feature, target)
        changed = out["distribution"].roll(1, dims=-1)
        self.assertFalse(paired.validate_label_reference(target, result, changed,
                                                         out["label_mass"])["passed"])
        self.assertFalse(paired.validate_label_reference(target, result, out["distribution"],
                                                         out["label_mass"]+0.01)["passed"])

    def test_failed_reference_is_a_hard_error_without_tolerance_retry(self):
        result, feature, target = fixture()
        with mock.patch.object(paired, "validate_label_reference", return_value={"passed": False}) as ref:
            with self.assertRaisesRegex(ValueError, "FP64 reference failed"):
                paired.extract_pair(result, feature, target)
        self.assertEqual(ref.call_count, 1)

    def test_half_tokens_and_half_features_are_rejected_before_upcasting(self):
        result, feature, _ = fixture()
        result["tokens"] = result["tokens"].half()
        with self.assertRaisesRegex(ValueError, "FP32 front tokens"):
            paired.extract_pair(result, feature)
        result, feature, _ = fixture()
        with self.assertRaisesRegex(ValueError, "FP32 captured F256"):
            paired.extract_pair(result, feature.half())

    def test_zero_keys_are_inactive_regardless_of_labels(self):
        result, feature, target = fixture()
        result["tokens"][0, 0] = 0
        out = paired.extract_pair(result, feature, target)
        self.assertFalse(out["active"][0])
        feature.zero_()
        self.assertFalse(paired.extract_pair(result, feature)["active"].any())

    def test_tiny_nonzero_keys_use_the_shared_retrieval_norm_threshold(self):
        result, feature, _ = fixture()
        # A single nonzero coordinate controls the exact L2 norm independently
        # of the widths 128 and 256; summing tiny values across every channel
        # would otherwise change the intended boundary.
        result["tokens"].zero_()
        result["tokens"][0, 0, 0] = 5e-9
        result["tokens"][0, 1, 0] = 2e-8
        self.assertEqual(paired.extract_pair(result, feature)["active"].tolist(),
                         [False, True, False, False])
        result["tokens"][0, :2, 0] = 2e-8
        feature.zero_()
        feature[0, 0] = 5e-9
        self.assertFalse(paired.extract_pair(result, feature)["active"].any())
        feature[0, 0] = 2e-8
        self.assertEqual(paired.extract_pair(result, feature)["active"].tolist(),
                         [True, True, False, False])
        result["tokens"][0, 0, 0] = 1e-8
        self.assertFalse(paired.extract_pair(result, feature)["active"][0])

    def test_inputs_and_current_forward_are_not_mutated(self):
        result, feature, target = fixture()
        copied = copy.deepcopy(result)
        feature_before, target_before = feature.clone(), target.clone()
        paired.extract_pair(result, feature, target)
        for name, old in copied.items():
            if isinstance(old, torch.Tensor):
                self.assertTrue(torch.equal(result[name], old), name)
            else:
                self.assertEqual(result[name], old)
        self.assertTrue(torch.equal(feature, feature_before))
        self.assertTrue(torch.equal(target, target_before))

    def test_invalid_front_route_and_feature_inputs_fail(self):
        cases = (
            lambda r, f: r["q"].fill_(float("nan")),
            lambda r, f: r["q"].fill_(-1),
            lambda r, f: r["q"].zero_(),
            lambda r, f: r["ids"].fill_(4),
            lambda r, f: r["ids"].fill_(-1),
            lambda r, f: r.update(ids=r["ids"].float()),
            lambda r, f: r.update(count=True),
            lambda r, f: r.update(shape=(4, 4)),
            lambda r, f: r.update(shape=(2.0, 2)),
            lambda r, f: r.update(tokens=r["tokens"].expand(2, -1, -1)),
            lambda r, f: r["tokens"].fill_(float("inf")),
            lambda r, f: r.update(valid_regions=r["valid_regions"].long()),
            lambda r, f: r["mass"].fill_(-1),
            lambda r, f: r["mass"].fill_(float("nan")),
            lambda r, f: r["mass"].add_(1),
            lambda r, f: f.fill_(float("nan")),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                result, feature, _ = fixture()
                mutate(result, feature)
                with self.assertRaises((ValueError, FloatingPointError)):
                    paired.extract_pair(result, feature)
        result, _, _ = fixture()
        with self.assertRaises(ValueError):
            paired.extract_pair(result, torch.zeros(1, 128, 2, 2))

    def test_invalid_target_and_reference_output_inputs_fail(self):
        result, feature, target = fixture()
        for bad_target in (target.float(), target.bool(), target[0], target.expand(2, -1, -1)):
            with self.subTest(shape=bad_target.shape, dtype=bad_target.dtype):
                with self.assertRaises(ValueError):
                    paired.extract_pair(result, feature, bad_target)
        out = paired.extract_pair(result, feature, target)
        with self.assertRaises(ValueError):
            paired.validate_label_reference(target, result, out["distribution"][:2], out["label_mass"])
        with self.assertRaises(ValueError):
            paired.validate_label_reference(target, result, out["distribution"]*float("nan"), out["label_mass"])


if __name__ == "__main__":
    unittest.main()
