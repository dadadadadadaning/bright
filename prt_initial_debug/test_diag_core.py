import copy
import json
import unittest
from unittest import mock

import torch
from torch import nn

from diag_core import (backend_flags, close_report, compare_fixed_heads,
                       temporary_backend_flags, tensor_digest, tensor_meta)


class FixedHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.head = nn.Conv2d(3, 4, 1).float().eval()
        self.copy = copy.deepcopy(self.head)

    def test_equal_p_is_repeatable_and_json_safe_without_mutation(self):
        x = torch.randn(1, 3, 2, 3)
        before = x.clone()
        result = compare_fixed_heads("p", x, self.head, self.copy, (4, 6))
        self.assertTrue(result["initial_parameter_equal"])
        self.assertTrue(result["parameters_unchanged"])
        self.assertTrue(result["backend_restored"])
        self.assertTrue(result["diagnostic_only"])
        self.assertTrue(self.copy.weight.requires_grad)
        self.assertTrue(torch.equal(x, before))
        for mode in result["modes"].values():
            for comparison in mode["comparisons"].values():
                self.assertTrue(all(stage["passed"] for stage in comparison.values()))
        json.dumps(result, allow_nan=False)

    def test_bias_perturbation_is_localized_in_logits_and_reference(self):
        x = torch.randn(1, 3, 2, 3)
        with torch.no_grad():
            self.copy.bias[2] += 0.125
        result = compare_fixed_heads("p", x, self.head, self.copy, (2, 3),
                                     run_backend_probes=False)
        self.assertFalse(result["initial_parameter_equal"])
        stages = result["modes"]["legacy"]["comparisons"]["clone_1_vs_original_1"]
        self.assertFalse(stages["logits"]["passed"])
        self.assertEqual(stages["logits"]["max_index"][1], 2)
        self.assertAlmostEqual(stages["logits"]["max_abs"], .125, places=6)
        point = stages["logits"]
        self.assertAlmostEqual(point["cpu_float64_actual_head"]["logit"] -
                               point["cpu_float64_reference_head"]["logit"], .125, places=6)
        self.assertFalse(stages["softmax"]["passed"])

    def test_r_uses_original_conv_weight_as_linear_and_injected_unpool(self):
        x = torch.randn(1, 6, 3)
        calls = []
        def unpool(prob):
            calls.append(tuple(prob.shape))
            return prob.transpose(1, 2).reshape(1, 4, 2, 3)
        result = compare_fixed_heads("r", x, self.head, self.copy, (4, 6),
                                     unpool=unpool, run_backend_probes=False)
        self.assertEqual(calls, [(1, 6, 4)] * 6)
        for stages in result["modes"]["legacy"]["comparisons"].values():
            self.assertTrue(stages["logits"]["passed"])
            self.assertTrue(stages["unpool"]["passed"])
        dot = result["modes"]["legacy"]["comparisons"]["clone_1_vs_original_1"]
        self.assertAlmostEqual(dot["logits"]["actual_at_max"],
                               dot["logits"]["cpu_float64_actual_head"]["logit"], places=6)

    def test_captured_legacy_difference_is_not_hidden_by_head_repeat_agreement(self):
        x = torch.randn(1, 3, 2, 3)
        captured = self.head(x).detach().clone()
        captured[0, 1, 1, 2] += .25
        result = compare_fixed_heads("p", x, self.head, self.copy, (2, 3),
                                     captured_stages={"logits": captured},
                                     run_backend_probes=False)
        self.assertFalse(result["captured_vs_fixed"]["logits"]["passed"])
        self.assertEqual(result["captured_vs_fixed"]["logits"]["max_index"], [0, 1, 1, 2])
        repeat = result["modes"]["legacy"]["comparisons"]["original_2_vs_original_1"]
        self.assertTrue(repeat["logits"]["passed"])

    def test_flags_restore_on_exception(self):
        before = backend_flags()
        with self.assertRaisesRegex(RuntimeError, "diagnostic exception"):
            with temporary_backend_flags(cudnn_allow_tf32=not before["cudnn_allow_tf32"],
                                         cudnn_enabled=not before["cudnn_enabled"]):
                raise RuntimeError("diagnostic exception")
        self.assertEqual(backend_flags(), before)

    def test_unpool_exception_propagates_and_does_not_mutate_flags_or_parameters(self):
        before = backend_flags()
        weights = self.head.weight.detach().clone()
        def fail(_):
            raise RuntimeError("unpool exception")
        with self.assertRaisesRegex(RuntimeError, "unpool exception"):
            compare_fixed_heads("r", torch.randn(1, 6, 3), self.head, self.copy,
                                (2, 3), unpool=fail)
        self.assertEqual(backend_flags(), before)
        self.assertTrue(torch.equal(weights, self.head.weight))

    def test_tensor_operations_stop_after_primary_device_error(self):
        observed = {}
        with mock.patch("diag_core.tensor_digest", wraps=tensor_digest) as hashing, \
                mock.patch("diag_core.torch.equal", wraps=torch.equal) as equality:
            def fail(_):
                observed.update(hashes=hashing.call_count, equals=equality.call_count)
                raise RuntimeError("simulated primary CUDA error")
            with self.assertRaisesRegex(RuntimeError, "primary CUDA error"):
                compare_fixed_heads("r", torch.randn(1, 6, 3), self.head, self.copy,
                                    (2, 3), unpool=fail)
            self.assertEqual(hashing.call_count, observed["hashes"])
            self.assertEqual(equality.call_count, observed["equals"])

    def test_metadata_preserves_offset_and_noncontiguous_layout(self):
        base = torch.randn(1, 3, 3, 4)
        sliced = base[:, :, 1:, 1:]
        meta = tensor_meta(sliced)
        self.assertEqual(meta["shape"], [1, 3, 2, 3])
        self.assertEqual(meta["stride"], list(base.stride()))
        self.assertGreater(meta["storage_offset"], 0)
        self.assertFalse(meta["is_contiguous"])
        self.assertEqual(meta["version"], sliced._version)

    def test_close_report_locates_sparse_difference_and_keeps_gate(self):
        reference = torch.ones(1, 4, 2, 3)
        actual = reference.clone()
        actual[0, 3, 1, 2] += .001
        result = close_report(actual, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outside_tolerance"], 1)
        self.assertEqual(result["max_index"], [0, 3, 1, 2])
        self.assertEqual(result["atol"], 2e-5)
        actual[0, 3, 1, 2] = 1.00001
        self.assertTrue(close_report(actual, reference)["passed"])


if __name__ == "__main__":
    unittest.main()
