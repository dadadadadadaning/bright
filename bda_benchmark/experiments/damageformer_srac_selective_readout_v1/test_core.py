"""Small CPU-only contracts for selective readout and the boundary audit.

Run from the repository root:
    python -m unittest bda_benchmark.experiments.damageformer_srac_selective_readout_v1.test_core -v
"""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from typing import Tuple
import unittest

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.boundary import boundary_maps
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.core import LocalSelectiveReadout, normalize_routes


def _old_boundary_reference():
    """Load just the existing pure function without importing its backbone."""
    path = Path(__file__).parent.parent / "damageformer_srac_transformer_v1" / "model.py"
    source = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "soft_assignment_boundary")
    namespace = {"torch": torch, "F": F, "Tuple": Tuple}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["soft_assignment_boundary"]


def _fixture():
    torch.manual_seed(124)
    batch, height, width, regions, supports = 2, 3, 5, 7, 4
    pixel = torch.randn(batch, 6, height, width)
    tokens = torch.randn(batch, regions, 5)
    q = torch.rand(batch, supports, height * width)
    q[:, 0, ::3] = 0
    q /= q.sum(1, keepdim=True)
    ids = torch.stack([(torch.arange(height * width) + offset) % regions for offset in range(supports)])
    valid = torch.ones(batch, regions, dtype=torch.bool)
    valid[0, 2] = False
    valid[1, 5] = False
    return pixel, tokens, q, ids, valid


class NormalizeRoutesTests(unittest.TestCase):
    def test_sparse_masking_and_normalization(self):
        q = torch.tensor([[[0.0, 0.2], [0.4, 0.3], [0.6, 0.5]]])
        valid = torch.tensor([[[True, True], [False, True], [True, False]]])
        scores = torch.tensor([[[1000.0, 1.0], [1000.0, -1.0], [2.0, 1000.0]]])
        routes = normalize_routes(q, valid, scores)
        expected_second = torch.tensor([0.2 * math.exp(1), 0.3 * math.exp(-1), 0.0])
        expected_second /= expected_second.sum()
        self.assertTrue(torch.equal(routes[:, :, 0], torch.tensor([[0.0, 0.0, 1.0]])))
        torch.testing.assert_close(routes[0, :, 1], expected_second)
        self.assertTrue(torch.equal(routes[~(valid & (q > 0))], torch.zeros_like(routes[~(valid & (q > 0))])))
        torch.testing.assert_close(routes.sum(1), torch.ones(1, 2))

    def test_all_invalid_and_zero_mass_are_safe(self):
        q = torch.tensor([[[0.2, 0.0], [0.8, 0.0]]], requires_grad=True)
        scores = torch.randn_like(q, requires_grad=True)
        valid = torch.tensor([[[False, True], [False, True]]])
        routes = normalize_routes(q, valid, scores)
        self.assertTrue(torch.equal(routes, torch.zeros_like(routes)))
        routes.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(float(scores.grad.abs().sum()), 0)

    def test_extreme_scores_and_tiny_positive_q(self):
        q = torch.tensor([[[1e-35], [1.0], [0.0]]])
        scores = torch.tensor([[[1e4], [-1e4], [1e20]]])
        routes = normalize_routes(q, torch.ones_like(q, dtype=torch.bool), scores)
        self.assertTrue(torch.equal(routes, torch.tensor([[[1.0], [0.0], [0.0]]])))

    def test_subnormal_surviving_q_mass_has_finite_score_gradients(self):
        q = torch.tensor([[[1e-40], [2e-40], [1.0]]])
        valid = torch.tensor([[[True], [True], [False]]])
        scores = torch.zeros_like(q, requires_grad=True)
        routes = normalize_routes(q, valid, scores)
        torch.testing.assert_close(routes[:, :2, 0], torch.tensor([[1 / 3, 2 / 3]]), atol=2e-6, rtol=1e-5)
        routes[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.abs().sum()), 0)
        self.assertEqual(float(scores.grad[:, 2].abs().sum()), 0)

    def test_nonfinite_and_negative_q_rejected(self):
        valid = torch.ones(1, 2, 1, dtype=torch.bool)
        for bad in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError):
                normalize_routes(torch.tensor([[[bad], [1.0]]]), valid)


class LocalReadoutTests(unittest.TestCase):
    def test_initial_identity_batch_two_and_input_immutability(self):
        inputs = _fixture()
        saved = [tensor.clone() for tensor in inputs]
        readout = LocalSelectiveReadout(6, 5, 3, chunk_size=4)
        self.assertEqual(float(readout.phi.weight.detach().abs().sum()), 0)
        self.assertGreater(float(readout.psi.weight.detach().abs().sum()), 0)
        q_context, q_stats, q_routes = readout(*inputs, mode="q", return_routes=True)
        content, stats, routes = readout(*inputs, mode="content", return_routes=True)
        self.assertTrue(torch.equal(content, q_context))
        self.assertTrue(torch.equal(routes, q_routes))
        self.assertEqual(stats["max_route_delta"], 0)
        self.assertEqual(stats["score_abs_max"], 0)
        self.assertEqual(stats, q_stats)
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in stats.values()))
        for tensor, unchanged in zip(inputs, saved):
            self.assertTrue(torch.equal(tensor, unchanged))

    def test_matches_dense_oracle_without_opening_routes(self):
        pixel, tokens, q, ids, valid = _fixture()
        readout = LocalSelectiveReadout(6, 5, 3, chunk_size=4)
        with torch.no_grad():
            readout.phi.weight.normal_(std=0.3)
        context, stats, routes = readout(pixel, tokens, q, ids, valid, "content", return_routes=True)
        dense_q = q.new_zeros(len(pixel), q.shape[-1], tokens.shape[1])
        dense_q.scatter_add_(2, ids.T[None].expand(len(pixel), -1, -1), q.transpose(1, 2))
        dense_q *= valid[:, None]
        scores = torch.einsum("bnd,bkd->bnk", readout.phi(pixel.flatten(2).transpose(1, 2)), readout.psi(tokens)) * readout.scale
        dense_routes = normalize_routes(dense_q.transpose(1, 2), valid[:, :, None].expand_as(dense_q.transpose(1, 2)), scores.transpose(1, 2)).transpose(1, 2)
        expected = torch.bmm(dense_routes, tokens).transpose(1, 2).reshape_as(context)
        torch.testing.assert_close(context, expected, atol=1e-6, rtol=1e-5)
        active = valid[:, ids] & (q > 0)
        self.assertTrue(torch.equal(routes[~active], torch.zeros_like(routes[~active])))
        self.assertEqual(stats["invalid_route_mass"], 0)
        self.assertGreater(stats["route_l1_mean"], 0)

    def test_chunk_invariance_outputs_and_gradients(self):
        inputs = _fixture()
        first = LocalSelectiveReadout(6, 5, 3, chunk_size=1)
        with torch.no_grad():
            first.phi.weight.normal_(std=0.2)
        second = copy.deepcopy(first)
        second.chunk_size = 1000
        out1, stats1, route1 = first(*inputs, mode="content", return_routes=True)
        out2, stats2, route2 = second(*inputs, mode="content", return_routes=True)
        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(route1, route2, atol=1e-6, rtol=1e-5)
        for name in stats1:
            self.assertAlmostEqual(stats1[name], stats2[name], places=5)
        out1.square().mean().backward()
        out2.square().mean().backward()
        for one, two in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(one.grad, two.grad, atol=1e-6, rtol=1e-5)

    def test_scorer_is_not_dead_at_initialization(self):
        inputs = _fixture()
        readout = LocalSelectiveReadout(6, 5, 3, chunk_size=4)
        optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
        context, _ = readout(*inputs, mode="content")
        context.square().mean().backward()
        for parameter in readout.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(readout.phi.weight.grad.abs().sum()), 0)
        self.assertEqual(float(readout.psi.weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        context, stats = readout(*inputs, mode="content")
        context.square().mean().backward()
        self.assertGreater(float(readout.phi.weight.grad.abs().sum()), 0)
        self.assertGreater(float(readout.psi.weight.grad.abs().sum()), 0)
        self.assertGreater(stats["route_l1_mean"], 0)
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in readout.parameters()))

    def test_all_invalid_regions_zero_context_and_finite_gradients(self):
        pixel, tokens, q, ids, valid = _fixture()
        valid.zero_()
        readout = LocalSelectiveReadout(6, 5, 3, chunk_size=4)
        for mode in ("q", "content"):
            context, stats, routes = readout(pixel, tokens, q, ids, valid, mode, return_routes=True)
            self.assertTrue(torch.equal(context, torch.zeros_like(context)))
            self.assertTrue(torch.equal(routes, torch.zeros_like(routes)))
            self.assertEqual(stats["empty_fraction"], 1)
            self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        context.sum().backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in readout.parameters()))

    def test_candidate_permutation_invariance(self):
        pixel, tokens, q, ids, valid = _fixture()
        readout = LocalSelectiveReadout(6, 5, 3, chunk_size=4)
        with torch.no_grad():
            readout.phi.weight.normal_()
        permutation = torch.stack([torch.randperm(q.shape[1]) for _ in range(q.shape[2])], dim=1)
        permuted_q = q.gather(1, permutation[None].expand(len(pixel), -1, -1))
        permuted_ids = ids.gather(0, permutation)
        original, _, routes = readout(pixel, tokens, q, ids, valid, "content", return_routes=True)
        permuted, _, permuted_routes = readout(pixel, tokens, permuted_q, permuted_ids, valid, "content", return_routes=True)
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(routes.gather(1, permutation[None].expand(len(pixel), -1, -1)), permuted_routes)


class BoundaryTests(unittest.TestCase):
    def test_uniform_identical_distribution_old_eight_ninths_change_zero(self):
        ids = torch.arange(9)[:, None].expand(-1, 6)
        q = torch.full((2, 9, 6), 1 / 9)
        maps = boundary_maps(q, ids, (2, 3))
        torch.testing.assert_close(maps["old"], torch.full((2, 1, 2, 3), 8 / 9))
        torch.testing.assert_close(maps["change"], torch.zeros_like(maps["change"]), atol=1e-7, rtol=0)
        self.assertTrue(torch.equal(maps["zero"], torch.zeros_like(maps["old"])))

    def test_one_hot_same_and_different_regions(self):
        q = torch.ones(1, 1, 4)
        ids = torch.tensor([[0, 0, 1, 1]])
        maps = boundary_maps(q, ids, (1, 4))
        expected = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
        self.assertTrue(torch.equal(maps["old"], expected))
        self.assertTrue(torch.equal(maps["change"], expected))

    def test_candidate_permutation_and_global_id_relabeling(self):
        _, _, q, ids, _ = _fixture()
        permutation = torch.stack([torch.randperm(q.shape[1]) for _ in range(q.shape[2])], dim=1)
        changed_ids = ids.gather(0, permutation) * 101 + 17
        changed_q = q.gather(1, permutation[None].expand(q.shape[0], -1, -1))
        original = boundary_maps(q, ids, (3, 5))
        permuted = boundary_maps(changed_q, changed_ids, (3, 5))
        for name in original:
            torch.testing.assert_close(original[name], permuted[name], atol=1e-6, rtol=1e-5)

    def test_changed_candidate_topology_with_overlapping_support(self):
        # Uniform slots are identical locally, but their global support differs.
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
        q = torch.full((1, 3, 2), 1 / 3)
        maps = boundary_maps(q, ids, (1, 2))
        torch.testing.assert_close(maps["old"], torch.full((1, 1, 1, 2), 7 / 9))
        torch.testing.assert_close(maps["change"], torch.full((1, 1, 1, 2), 1 / 9))

    def test_duplicate_ids_aggregate_before_squared_norm(self):
        # Both pixels have global distribution [0.5, 0.5], different slot splits.
        ids = torch.tensor([[0, 0], [0, 1], [1, 1]])
        q = torch.tensor([[[0.2, 0.5], [0.3, 0.1], [0.5, 0.4]]])
        maps = boundary_maps(q, ids, (1, 2))
        torch.testing.assert_close(maps["old"], torch.full((1, 1, 1, 2), 0.5))
        torch.testing.assert_close(maps["change"], torch.zeros(1, 1, 1, 2), atol=1e-7, rtol=0)

    def test_old_is_bitwise_existing_implementation_and_mean_is_per_image(self):
        _, _, q, ids, _ = _fixture()
        old = _old_boundary_reference()(q, ids, (3, 5))
        maps = boundary_maps(q, ids, (3, 5))
        self.assertTrue(torch.equal(maps["old"], old))
        expected_mean = old.mean((2, 3), keepdim=True).expand_as(old)
        self.assertTrue(torch.equal(maps["mean"], expected_mean))
        self.assertFalse(torch.equal(expected_mean[0], expected_mean[1]))

    def test_change_matches_dense_global_oracle_and_has_finite_gradients(self):
        _, _, q, ids, _ = _fixture()
        q = q.clone().requires_grad_()
        dense = q.new_zeros(q.shape[0], int(ids.max()) + 1, q.shape[-1])
        dense.scatter_add_(1, ids[None].expand(len(q), -1, -1), q)
        dense = dense.reshape(len(q), -1, 3, 5)
        horizontal = 0.5 * (dense[:, :, :, :-1] - dense[:, :, :, 1:]).square().sum(1, keepdim=True)
        vertical = 0.5 * (dense[:, :, :-1] - dense[:, :, 1:]).square().sum(1, keepdim=True)
        expected = torch.stack((F.pad(horizontal, (0, 1, 0, 0)), F.pad(horizontal, (1, 0, 0, 0)), F.pad(vertical, (0, 0, 0, 1)), F.pad(vertical, (0, 0, 1, 0)))).amax(0)
        maps = boundary_maps(q, ids, (3, 5))
        torch.testing.assert_close(maps["change"], expected, atol=1e-6, rtol=1e-5)
        (maps["old"].mean() + maps["change"].mean()).backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_single_pixel_and_validation(self):
        q = torch.tensor([[[0.2], [0.8]]])
        ids = torch.tensor([[0], [1]])
        self.assertTrue(all(torch.equal(value, torch.zeros_like(value)) for value in boundary_maps(q, ids, (1, 1)).values()))
        with self.assertRaises(ValueError):
            boundary_maps(q * 2, ids, (1, 1))
        with self.assertRaises(ValueError):
            boundary_maps(q, ids - 2, (1, 1))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
