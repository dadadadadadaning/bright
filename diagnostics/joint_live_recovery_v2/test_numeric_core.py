"""CPU numerical contracts. Synthetic cancellation is not a server diagnosis."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import torch

import numeric_core as core


def snapshot(batch=2, channels=5):
    generator = torch.Generator(device="cpu").manual_seed(19)
    height, width, count = 3, 4, 7
    return {"format": "native_pool_input_v1", "token_count": count,
            "content": torch.randn(batch, channels, height, width, generator=generator),
            "weights": torch.rand(batch, 9, height*width, generator=generator)*.3,
            "ids": torch.arange(9*height*width).reshape(9, height*width).remainder(count)}


def outputs(value):
    return {key: value.clone() for key in core.OUTPUT_KEYS}


def literal(value, **kwargs):
    return core.pool_literal(value["content"], value["weights"], value["ids"],
                             value["token_count"], **kwargs)


class PoolMathContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.patches = []
        for name in ("_lazy_init", "is_available", "device_count"):
            patched = mock.patch.object(torch.cuda, name,
                        side_effect=AssertionError("CPU helper queried CUDA: " + name))
            patched.start()
            self.patches.append(patched)

    def tearDown(self):
        for patched in reversed(self.patches):
            patched.stop()

    def test_fp32_literal_matches_actual_original_source_exactly(self):
        projects = [Path(os.environ.get("NUMERIC_PROJECT", "/workspace/bright")),
                    Path("/root/shared-nvme/BRIGHT-master")]
        relative = "bda_benchmark/experiments/damageformer_a1_cross_modal_superpixel_v1/association.py"
        source = next((root/relative for root in projects if (root/relative).is_file()), None)
        if source is None:
            self.skipTest("Original association source is absent in this CPU workspace")
        specification = importlib.util.spec_from_file_location("original_pool_reference", source)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        value = snapshot()
        actual = literal(value)
        expected = module.soft_pool_unpool(value["content"], value["weights"],
                                          value["ids"], value["token_count"])
        for key in core.OUTPUT_KEYS:
            self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_cpu_contiguous_indices_preserve_values_and_original_inputs(self):
        value = snapshot()
        before = copy.deepcopy(value)
        native = literal(value)
        candidate = literal(value, index_layout="contiguous")
        for key in core.OUTPUT_KEYS:
            self.assertTrue(torch.equal(native[key], candidate[key]))
        for key in ("content", "weights", "ids"):
            self.assertTrue(torch.equal(value[key], before[key]))

    def test_grouped_double_reference_does_not_call_literal_or_scatter(self):
        value = snapshot()
        with mock.patch.object(core, "pool_literal", side_effect=AssertionError("literal used")), \
             mock.patch.object(torch.Tensor, "scatter_add_", side_effect=AssertionError("scatter used")), \
             mock.patch.object(torch.Tensor, "index_add_", side_effect=AssertionError("index_add used")):
            reference = core.cpu_reference64(value, channel_chunk=2, max_work_elements=17)
        self.assertTrue(all(reference[key].dtype == torch.float64 for key in core.OUTPUT_KEYS))

    def test_grouped_double_agrees_with_double_scatter_auxiliary(self):
        value = snapshot()
        grouped = core.cpu_reference64(value, channel_chunk=2, max_work_elements=17)
        scatter = literal(value, precision="fp64")
        for key in core.OUTPUT_KEYS:
            torch.testing.assert_close(grouped[key], scatter[key], atol=1e-12, rtol=1e-12)

    def test_channel_and_route_chunking_preserves_double_reference(self):
        value = snapshot()
        full = core.cpu_reference64(value)
        bounded = core.cpu_reference64(value, channel_chunk=1, max_work_elements=2)
        for key in core.OUTPUT_KEYS:
            torch.testing.assert_close(full[key], bounded[key], atol=1e-12, rtol=1e-12)

    def test_duplicate_routes_and_unnormalized_weights_are_retained(self):
        value = {"format": "native_pool_input_v1", "token_count": 3,
                 "content": torch.tensor([[[[1., 3., 5., 7.]]]]),
                 "weights": torch.zeros(1, 9, 4),
                 "ids": torch.tensor([[0, 0, 1, 1]]).expand(9, 4)}
        value["weights"][:, 0] = .2
        value["weights"][:, 1] = .3
        result = core.cpu_reference64(value, channel_chunk=1, max_work_elements=2)
        torch.testing.assert_close(result["tokens"][0, :, 0], torch.tensor([2., 6., 0.], dtype=torch.float64))
        expected_mass = value["weights"][:, :2, :2].double().sum()
        self.assertEqual(float(result["mass"][0, 0]), float(expected_mass))
        # Original unpool keeps total route weight .5; it does not normalize it.
        expected = torch.tensor([1., 1., 3., 3.], dtype=torch.float64)
        torch.testing.assert_close(result["reconstruction"].flatten(), expected, atol=1e-7, rtol=1e-7)

    def test_empty_regions_and_positive_mass_below_clamp(self):
        value = {"format": "native_pool_input_v1", "token_count": 4,
                 "content": torch.tensor([[[[2., 4., 6., 8.]]]]),
                 "weights": torch.full((1, 9, 4), 1e-9),
                 "ids": torch.zeros(9, 4, dtype=torch.long)}
        result = core.cpu_reference64(value, max_work_elements=3)
        numerator = (value["content"].double().flatten()*value["weights"][0, 0].double()).sum()*9
        self.assertLess(float(result["mass"][0, 0]), 1e-6)
        self.assertAlmostEqual(float(result["tokens"][0, 0, 0]), float(numerator/1e-6), places=12)
        self.assertTrue(torch.equal(result["tokens"][0, 1:], torch.zeros(3, 1, dtype=torch.float64)))
        self.assertTrue(bool(torch.isfinite(result["reconstruction"]).all()))

    def test_promote_saved_square_values_without_resquaring(self):
        value = {"format": "native_pool_input_v1", "token_count": 1,
                 "content": torch.tensor([[[[2., 3., 4., 5.]], [[9., 9., 9., 9.]]]]),
                 "weights": torch.zeros(1, 9, 4),
                 "ids": torch.zeros(9, 4, dtype=torch.long)}
        value["weights"][:, 0] = 1
        result = core.cpu_reference64(value)
        self.assertEqual(float(result["tokens"][0, 0, 1]), 9.)
        self.assertEqual(float(result["tokens"][0, 0, 0]), 3.5)

    def test_synthetic_signed_cancellation_distinguishes_fp32_and_fp64(self):
        value = {"format": "native_pool_input_v1", "token_count": 1,
                 "content": torch.tensor([[[[1e8, 1., -1e8, 1.]]]]),
                 "weights": torch.zeros(1, 9, 4),
                 "ids": torch.zeros(9, 4, dtype=torch.long)}
        value["weights"][:, 0] = 1
        fp32 = literal(value)
        fp64 = core.cpu_reference64(value)
        self.assertEqual(float(fp64["tokens"][0, 0, 0]), .5)
        self.assertEqual(float(fp32["tokens"][0, 0, 0]), .25)
        self.assertFalse(core.compare(fp32, fp64)["passed"])

    def test_broadcast_source_layout_has_same_grouped_math(self):
        value = snapshot(batch=1, channels=3)
        value["ids"] = torch.arange(12).remainder(7).unsqueeze(0).expand(9, 12)
        value["weights"] = torch.tensor([[[.1]]]).expand(1, 9, 12)
        grouped = core.cpu_reference64(value)
        native = literal(value, precision="fp64")
        for key in core.OUTPUT_KEYS:
            torch.testing.assert_close(grouped[key], native[key], atol=1e-12, rtol=1e-12)

    def test_bad_saved_inputs_rejected_without_clamping(self):
        value = snapshot()
        value["ids"][0, 0] = value["token_count"]
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            core.cpu_reference64(value)
        value = snapshot()
        value["weights"][0, 0, 0] = -.1
        with self.assertRaisesRegex(ValueError, "Negative"):
            core.validate_snapshot(value)
        value = snapshot()
        value["content"][0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            core.validate_snapshot(value)


class ComparisonContracts(unittest.TestCase):
    def test_max_abs_alone_does_not_determine_tolerance_pass(self):
        reference = outputs(torch.tensor([1e6, 0.], dtype=torch.float64))
        actual = outputs(torch.tensor([1e6+1, 3e-5], dtype=torch.float64))
        result = core.compare(actual, reference, work_elements=1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["tokens"]["max_abs"], 1.)
        self.assertEqual(result["tokens"]["elements_outside_tolerance"], 1)
        self.assertAlmostEqual(result["tokens"]["max_normalized_error"], 1.5)
        self.assertEqual(result["tokens"]["worst_failing_examples"][0]["index"], [1])

    def test_fixed_relative_tolerance_is_asymmetric(self):
        actual = outputs(torch.tensor([1.0000400004], dtype=torch.float64))
        reference = outputs(torch.tensor([1.], dtype=torch.float64))
        self.assertFalse(core.compare(actual, reference)["passed"])
        self.assertTrue(core.compare(reference, actual)["passed"])

    def test_large_scale_error_can_pass_relative_term(self):
        reference = outputs(torch.tensor([1e6], dtype=torch.float64))
        actual = outputs(torch.tensor([1e6+1], dtype=torch.float64))
        result = core.compare(actual, reference)
        self.assertTrue(result["passed"])
        self.assertEqual(result["mass"]["max_abs"], 1.)
        self.assertLess(result["mass"]["max_normalized_error"], 1)

    def test_noncontiguous_chunk_comparison_has_correct_worst_coordinates(self):
        actual = torch.zeros(2, 3, 4).transpose(1, 2)
        expected = torch.zeros_like(actual).double()
        actual[1, 2, 0] = 1
        actual[0, 3, 2] = .25
        result = core.compare(outputs(actual), outputs(expected), max_examples=1, work_elements=3)
        row = result["tokens"]
        self.assertEqual(row["total_elements"], 24)
        self.assertEqual(row["outside_tolerance"], 2)
        self.assertEqual(row["worst_failing_examples"][0]["index"], [1, 2, 0])
        self.assertEqual(row["worst_failing_examples"][0]["flat_index"], 18)

    def test_chunk_metrics_agree_with_direct_double_difference(self):
        generator = torch.Generator().manual_seed(44)
        reference = torch.randn(2, 3, 7, generator=generator).double()
        actual = reference + torch.randn(2, 3, 7, generator=generator).double()*1e-4
        result = core.compare(outputs(actual), outputs(reference), max_examples=3, work_elements=5)
        difference = (actual-reference).abs()
        normalized = difference/(core.ATOL + core.RTOL*reference.abs())
        row = result["tokens"]
        self.assertAlmostEqual(row["max_abs"], float(difference.max()), places=14)
        self.assertAlmostEqual(row["mean_abs"], float(difference.mean()), places=14)
        self.assertAlmostEqual(row["max_normalized_error"], float(normalized.max()), places=12)
        self.assertEqual(row["outside_tolerance"], int((normalized > 1).sum()))
        self.assertEqual(len(row["worst_failing_examples"]), 3)

    def test_fp32_vs_fp64_does_not_downcast_reference(self):
        source = torch.tensor([1.], dtype=torch.float32)
        reference = outputs(source.double() + 5e-5)
        actual = outputs(source)
        result = core.compare(actual, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(result["mass"]["actual_dtype"], "torch.float32")
        self.assertEqual(result["mass"]["reference_dtype"], "torch.float64")

    def test_nonfinite_report_is_json_safe_and_fails(self):
        actual = outputs(torch.tensor([torch.nan, torch.inf, 1.]))
        reference = outputs(torch.tensor([1., 1., 1.]))
        result = core.compare(actual, reference, work_elements=1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["tokens"]["nonfinite_elements"], 2)
        self.assertIsNone(result["tokens"]["max_abs"])
        json.dumps(result, allow_nan=False)

    def test_shape_mismatch_fails_with_reason(self):
        actual = outputs(torch.ones(2))
        reference = outputs(torch.ones(3))
        self.assertFalse(core.compare(actual, reference)["passed"])

    def test_tolerances_cannot_be_overridden_by_caller(self):
        with self.assertRaises(TypeError):
            core.compare(outputs(torch.ones(1)), outputs(torch.ones(1)), atol=1)


if __name__ == "__main__":
    unittest.main()
