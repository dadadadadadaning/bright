"""Full-image CPU regression for independent FP32 NLL reductions.

The fixture uses the inherited objective and actual evaluation aggregation.
The alternate scalar reduction EMULATES CUDA arithmetic on CPU; this is not a
GPU execution, a training replay, or BRIGHT validation evidence.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import copy
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import evidence


ROOT = Path(__file__).resolve().parents[1]
IMAGE_COUNT = 450


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_metrics(path):
    """Run exact inherited metric helpers without importing CLI/model state."""
    names = {"_confusion", "_metrics"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Missing inherited confusion/metric helper")
    scope = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def _cuda_nll_order_emulated_on_cpu(logp, target):
    """Emulate PyTorch's unweighted 2-D CUDA NLL mean, using CPU FP32 sums.

    PyTorch v2.1.2 and v2.2.0 have the same 2-D reduction in
    aten/src/ATen/native/cuda/Loss.cu:
    https://github.com/pytorch/pytorch/blob/v2.1.2/aten/src/ATen/native/cuda/Loss.cu
    https://github.com/pytorch/pytorch/blob/v2.2.0/aten/src/ATen/native/cuda/Loss.cu
    NLL_LOSS_THREADS == 32; nll_loss_forward_reduce_cuda_kernel_2d has
    32 strided sequential accumulators, then thread 0 sums their partials.
    Each valid label has weight 1, so this fixture's denominator is exact.
    """
    if (logp.device.type != "cpu" or logp.dtype != torch.float32
            or logp.ndim != 2 or target.ndim != 1 or len(target) % 32):
        raise AssertionError("Fixture requires a CPU FP32 table and 32-aligned count")
    losses = -logp[torch.arange(len(target)), target].numpy()
    partials = np.cumsum(losses.reshape(-1, 32), axis=0, dtype=np.float32)[-1]
    total = np.cumsum(partials, dtype=np.float32)[-1]
    return torch.tensor(total / np.float32(len(target)), dtype=torch.float32)


class NLLReductionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference = Path(os.environ.get(
            "BRIGHT_ABLATION_REFERENCE", ROOT / "joint_anchor_fresh_chain_v2"))
        helper_path = reference / "deps/helpers/objectives.py"
        if not helper_path.is_file():
            raise RuntimeError(
                "Set BRIGHT_ABLATION_REFERENCE to joint_anchor_fresh_chain_v2")
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, original_threads)
        objectives = _load_module("_nll_regression_objectives", helper_path)
        evaluator = _load_module(
            "_nll_regression_evaluation", ROOT / "slot_memory_ablation_v1/evaluation.py")
        original = _original_metrics(reference / "deps/original/train.py")

        target = torch.arange(1024 * 1024).reshape(1, 1024, 1024) % 4
        probability = torch.full((1, 4, 1024, 1024), 0.1 / 3)
        probability.scatter_(1, target[:, None], 0.9)
        target[:, -16:] = 255
        matrix = original._confusion(target.numpy(), probability.argmax(1).numpy())
        cls.valid_pixels = int(matrix.sum()) * IMAGE_COUNT
        cls.metrics, cls.objectives = {}, {}
        for mode in ("native_cpu", "cuda_order_emulated_on_cpu"):
            replacement = (
                nullcontext() if mode == "native_cpu" else
                mock.patch.object(objectives, "F", SimpleNamespace(
                    nll_loss=_cuda_nll_order_emulated_on_cpu)))
            with torch.no_grad(), replacement:
                terms = objectives.serialized_terms(
                    objectives.objective_terms(probability, target))
            values = {key: terms[key] for key in (
                "nll", "lovasz", "class_nll_contribution",
                "class_lovasz_contribution", "class_valid_counts", "valid_count")}
            values["original_loss"] = terms["loss"]
            values["class_present_image_counts"] = [
                int(n > 0) for n in terms["class_valid_counts"]]
            totals = {}
            # Repeated synthetic sufficient statistics exercise the actual
            # 450-image FP64 aggregation without evaluating 450 large images.
            for _ in range(IMAGE_COUNT):
                evaluator._add_terms(totals, values)
            cls.metrics[mode] = evaluator._metrics(
                matrix * IMAGE_COUNT, totals, IMAGE_COUNT, original)
            cls.objectives[mode] = values

    def _validate(self, metric):
        return evidence._metrics(
            {evidence.BASELINE: metric}, [evidence.BASELINE],
            IMAGE_COUNT, self.valid_pixels)

    def test_native_cpu_passes_original_cross_reduction_tolerance(self):
        metric = self.metrics["native_cpu"]
        self.assertEqual(self.objectives["native_cpu"]["valid_count"], 1_032_192)
        self.assertEqual(
            self.objectives["native_cpu"]["class_valid_counts"], [258_048] * 4)
        self.assertTrue(math.isclose(
            sum(metric["mean_image_class_nll_contribution"]),
            metric["mean_image_nll"], rel_tol=2e-6, abs_tol=2e-7))
        self.assertEqual(self._validate(metric), self.valid_pixels)

    def test_cuda_order_gap_is_accepted_without_rewriting_any_metric(self):
        metric = copy.deepcopy(self.metrics["cuda_order_emulated_on_cpu"])
        before = copy.deepcopy(metric)
        scalar = metric["mean_image_nll"]
        class_sum = sum(metric["mean_image_class_nll_contribution"])
        self.assertFalse(math.isclose(
            class_sum, scalar, rel_tol=2e-6, abs_tol=2e-7))
        self.assertGreater(abs(class_sum - scalar), 1e-5)
        # This is the exact failing gate used before the validator fix.
        with self.assertRaisesRegex(ValueError, "Inconsistent class nll sum"):
            evidence._near(class_sum, scalar, "class nll sum")
        self.assertEqual(self._validate(metric), self.valid_pixels)
        self.assertEqual(metric, before)
        for key in ("original_loss", "nll", "lovasz", "class_nll_contribution",
                    "class_lovasz_contribution"):
            self.assertEqual(
                metric["mean_image_" + key],
                self.objectives["cuda_order_emulated_on_cpu"][key])

    def test_corrupted_scalar_still_fails_original_loss_decomposition(self):
        metric = copy.deepcopy(self.metrics["cuda_order_emulated_on_cpu"])
        metric["mean_image_nll"] += 0.01
        with self.assertRaisesRegex(ValueError, "Inconsistent original loss"):
            self._validate(metric)


if __name__ == "__main__":
    unittest.main()
