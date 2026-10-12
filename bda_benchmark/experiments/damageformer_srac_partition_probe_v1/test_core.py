"""CPU contracts for isolated partition interventions; no BRIGHT data required.

These tests establish numerical/interface contracts, not evidence that any
intervention improves BRIGHT.  They never instantiate or load the A1 backbone.
"""
from __future__ import annotations

import math
import unittest

import torch

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map,
    soft_pool_unpool,
)
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.boundary import boundary_maps
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    SuperpixelEmbedding,
    SwinLocalSuperpixelAttention,
    project_region_probabilities,
)
from .core import (
    building_log_prior,
    candidate_map_radius,
    embedding_with_partition,
    run_partition,
    sparse_pool,
)


class PartitionProbeContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        self.device = torch.device("cpu")

    def make_module(self, **overrides):
        settings = dict(in_dim=4, attn_dim=4, stride=2, iterations=3,
                        temperature=0.2, spatial_weight=0.1, building_weight=0.5)
        settings.update(overrides)
        return SwinLocalSuperpixelAttention(**settings).eval()

    def inputs(self, batch=2, height=7, width=9):
        return (torch.randn(batch, 4, height, width),
                torch.rand(batch, 1, height * 2, width * 2))

    def check_partition(self, result, expected_supports):
        q = result["q"]
        height, width = result["shape"]
        self.assertEqual(q.shape[1:], (expected_supports, height * width))
        self.assertEqual(result["ids"].shape, q.shape[1:])
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q >= 0).all())
        self.assertTrue(torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=2e-6, rtol=2e-6))
        self.assertEqual(result["count"], math.prod(result["grid_shape"]))
        self.assertTrue(torch.isfinite(result["entropy_map"]).all())

    def test_candidate_radius_one_is_exact_original(self):
        for height, width, stride in ((1, 1, 8), (7, 9, 2), (13, 15, 3)):
            expected = candidate_map(height, width, stride, self.device)
            actual = candidate_map_radius(height, width, stride, 1, self.device)
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))
            self.assertEqual(actual[2:], expected[2:])

    def test_radius_two_expands_supports_not_seed_budget(self):
        one = candidate_map_radius(13, 15, 3, 1, self.device)
        two = candidate_map_radius(13, 15, 3, 2, self.device)
        self.assertEqual(one[0].shape, (9, 195))
        self.assertEqual(two[0].shape, (25, 195))
        self.assertEqual(one[2:], two[2:])
        self.assertTrue(torch.equal(one[0][4], two[0][12]))
        self.assertTrue(two[1][12].all())
        self.assertTrue((two[0] >= 0).all())
        self.assertTrue((two[0] < two[2] * two[3]).all())

    def test_default_partition_reproduces_original_module(self):
        module = self.make_module()
        embedding, building = self.inputs()
        expected = module(embedding, building)
        actual = run_partition(module, embedding, building)
        for key in ("q", "ids", "entropy_map"):
            self.assertTrue(torch.equal(actual[key], expected[key]), msg=key)
        for key in ("shape", "grid_shape", "count"):
            self.assertEqual(actual[key], expected[key])

    def test_iteration_counts_and_candidate_masks_are_valid(self):
        module = self.make_module()
        embedding, building = self.inputs(batch=1)
        for radius in (1, 2):
            for iterations in (1, 3, 5):
                result = run_partition(module, embedding, building, radius=radius, iterations=iterations)
                self.check_partition(result, (2 * radius + 1) ** 2)
                _, valid, _, _ = candidate_map_radius(7, 9, module.stride, radius, self.device)
                self.assertEqual(torch.count_nonzero(result["q"].masked_select(~valid[None])).item(), 0)
                self.assertTrue(torch.equal(result["candidate_valid"], valid))
                self.assertEqual(len(result["trace"]), iterations)
                for index, row in enumerate(result["trace"]):
                    self.assertEqual(row["iteration"], index + 1)
                    self.assertAlmostEqual(float(row["mass_sum"]), 63.0, places=4)
                    self.assertLess(float(row["row_sum_error"]), 2e-6)
                    self.assertEqual(float(row["invalid_assignment_max"]), 0)
                if radius == 2:
                    # The outer 16 positions must really participate, not be
                    # silently truncated by the old nine-slot center helper.
                    outer = torch.tensor([i for i in range(25)
                                          if i // 5 in (0, 4) or i % 5 in (0, 4)])
                    self.assertGreater(result["q"][:, outer].sum().item(), 0)

    def test_entropy_diagnostic_uses_available_candidates_not_global_nine(self):
        module = self.make_module(spatial_weight=0, building_weight=0)
        with torch.no_grad():
            module.query.weight.zero_()
        embedding, building = self.inputs(batch=1)
        for radius in (1, 2):
            result = run_partition(module, embedding, building, radius=radius)
            available = result["candidate_valid"].sum(0).reshape(1, 1, 7, 9)
            expected_legacy = available.float().log() / math.log(9.0)
            self.assertTrue(torch.allclose(result["entropy_map"], expected_legacy, atol=2e-6, rtol=2e-6))
            self.assertTrue(torch.allclose(result["entropy_valid_map"], torch.ones_like(expected_legacy), atol=2e-6, rtol=2e-6))
        one_seed_embedding, one_seed_building = self.inputs(batch=1, height=1, width=1)
        one_seed = run_partition(module, one_seed_embedding, one_seed_building, radius=2)
        self.assertEqual(one_seed["entropy_valid_map"].item(), 0.0)

    def test_overrides_do_not_mutate_parameters_buffers_or_config(self):
        module = self.make_module()
        state_before = {key: value.detach().clone() for key, value in module.state_dict().items()}
        names = ("stride", "iterations", "temperature", "spatial_weight", "building_weight", "training")
        config_before = {name: getattr(module, name) for name in names}
        embedding, building = self.inputs(batch=1)
        run_partition(module, embedding, building, iterations=5, radius=2, stride=1,
                      prior_mode="bounded", prior_strength=2.0, leakage=0.1)
        self.assertEqual({name: getattr(module, name) for name in names}, config_before)
        self.assertEqual(set(module.state_dict()), set(state_before))
        for key, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[key]), msg=key)

    def test_partition_gradients_are_finite_and_building_is_detached(self):
        module = self.make_module()
        embedding, building = self.inputs(batch=1)
        embedding.requires_grad_(True)
        building.requires_grad_(True)
        result = run_partition(module, embedding, building, radius=2, iterations=3,
                               prior_mode="bounded", prior_strength=2.0)
        objective = (result["q"] * torch.randn_like(result["q"])).sum()
        objective.backward()
        self.assertIsNotNone(embedding.grad)
        self.assertTrue(torch.isfinite(embedding.grad).all())
        self.assertGreater(embedding.grad.abs().sum().item(), 0)
        self.assertIsNone(building.grad)
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)

    def test_one_seed_and_empty_slot_cases_do_not_create_nan(self):
        module = self.make_module(stride=8)
        embedding, building = self.inputs(batch=1, height=1, width=1)
        result = run_partition(module, embedding, building, radius=2, iterations=5)
        self.check_partition(result, 25)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["q"][0, 12, 0].item(), 1.0)
        pooled = sparse_pool(embedding, result["q"], result["ids"], 3)
        self.assertTrue(torch.isfinite(pooled["tokens"]).all())
        self.assertTrue(torch.equal(pooled["tokens"][:, 1:], torch.zeros_like(pooled["tokens"][:, 1:])))
        self.assertTrue(torch.allclose(pooled["reconstruction"], embedding))

    def test_legacy_building_prior_matches_original_formula(self):
        pixel = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0])[:, None]
        center = torch.tensor([0.0, 0.3, 1.0])[None, :]
        c = center.clamp(1e-4, 1 - 1e-4)
        expected = 0.5 * (pixel * c + (1 - pixel) * (1 - c)).clamp_min(1e-4).log()
        actual = building_log_prior(pixel, center, mode="legacy", strength=0.5)
        self.assertTrue(torch.equal(actual, expected))

    def test_bounded_prior_is_finite_and_not_hard_exclusion(self):
        pixel = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])[:, None]
        center = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])[None, :]
        strength, leakage = 2.0, 0.1
        prior = building_log_prior(pixel, center, mode="bounded", strength=strength, leakage=leakage)
        self.assertTrue(torch.isfinite(prior).all())
        self.assertTrue((prior <= 1e-7).all())
        self.assertGreaterEqual(prior.min().item(), strength * math.log(leakage) - 1e-6)
        self.assertGreater(prior[0, 0].item(), prior[0, -1].item())
        self.assertGreater(prior[-1, -1].item(), prior[-1, 0].item())
        self.assertTrue(torch.equal(prior[2], torch.zeros_like(prior[2])))
        self.assertTrue(torch.equal(prior[:, 2], torch.zeros_like(prior[:, 2])))
        # Equal content affinities still leave nonzero cross-building probability.
        probabilities = prior.softmax(-1)
        self.assertGreater(probabilities[0, -1].item(), 0)
        self.assertGreater(probabilities[-1, 0].item(), 0)

    def test_nine_support_pool_and_embedding_match_existing_code(self):
        module = self.make_module()
        embedding, building = self.inputs(batch=1)
        partition = run_partition(module, embedding, building)
        q, ids, count = partition["q"], partition["ids"], partition["count"]
        content = torch.randn(1, 5, 7, 9)
        expected_pool = soft_pool_unpool(content, q, ids, count)
        actual_pool = sparse_pool(content, q, ids, count)
        for key in ("tokens", "mass", "reconstruction"):
            self.assertTrue(torch.equal(actual_pool[key], expected_pool[key]), msg=key)
        readout = SuperpixelEmbedding(content_dim=5, adapted_dim=4, dim=8).eval()
        expected = readout(embedding, content, q, ids, partition["shape"], count)
        actual = embedding_with_partition(readout, embedding, content, partition)
        for original, replay in zip(expected, actual):
            self.assertTrue(torch.equal(original, replay))

    def test_twenty_five_support_pool_matches_dense_oracle_and_backpropagates(self):
        height, width, stride = 7, 9, 2
        ids, valid, gh, gw = candidate_map_radius(height, width, stride, 2, self.device)
        logits = torch.randn(2, 25, height * width, requires_grad=True)
        q = logits.masked_fill(~valid[None], -torch.inf).softmax(1)
        content = torch.randn(2, 3, height, width, requires_grad=True)
        result = sparse_pool(content, q, ids, gh * gw)
        dense = q.new_zeros(2, gh * gw, height * width)
        dense.scatter_add_(1, ids[None].expand(2, -1, -1), q)
        mass = dense.sum(-1)
        tokens = torch.bmm(dense, content.flatten(2).transpose(1, 2)) / mass.clamp_min(1e-6)[..., None]
        reconstruction = torch.bmm(dense.transpose(1, 2), tokens).transpose(1, 2).reshape_as(content)
        self.assertTrue(torch.allclose(result["mass"], mass, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(result["tokens"], tokens, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(result["reconstruction"], reconstruction, atol=2e-6, rtol=2e-6))
        result["reconstruction"].square().mean().backward()
        for gradient in (logits.grad, content.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        constants = torch.full((2, 3, height, width), 2.5)
        constant_result = sparse_pool(constants, q.detach(), ids, gh * gw)
        self.assertTrue(torch.allclose(constant_result["reconstruction"], constants, atol=2e-6, rtol=2e-6))

    def test_twenty_five_support_embedding_and_probability_projection_are_valid(self):
        module = self.make_module()
        adapted, building = self.inputs(batch=1)
        partition = run_partition(module, adapted, building, radius=2)
        raw = torch.randn(1, 5, 7, 9)
        readout = SuperpixelEmbedding(content_dim=5, adapted_dim=4, dim=8).eval()
        tokens, mass, geometry, valid = embedding_with_partition(readout, adapted, raw, partition)
        self.assertEqual(tokens.shape, (1, partition["count"], 8))
        self.assertEqual(geometry.shape, (1, partition["count"], 5))
        self.assertTrue(torch.equal(valid, mass > 1e-6))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertTrue(torch.isfinite(geometry).all())
        logits = torch.randn(1, partition["count"], 4)
        probability = project_region_probabilities(logits, partition["q"], partition["ids"], partition["shape"], (14, 18))
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue(torch.allclose(probability.sum(1), torch.ones(1, 14, 18), atol=2e-6, rtol=2e-6))

    def test_boundary_separates_identical_uncertainty_from_distribution_change(self):
        q = torch.full((1, 2, 2), 0.5)
        ids = torch.tensor([[0, 0], [1, 1]])
        maps = boundary_maps(q, ids, (1, 2))
        self.assertTrue(torch.equal(maps["old"], torch.full((1, 1, 1, 2), 0.5)))
        self.assertTrue(torch.equal(maps["change"], torch.zeros(1, 1, 1, 2)))
        disjoint = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        maps = boundary_maps(disjoint, ids, (1, 2))
        self.assertTrue(torch.equal(maps["old"], torch.ones(1, 1, 1, 2)))
        self.assertTrue(torch.equal(maps["change"], torch.ones(1, 1, 1, 2)))

    def test_boundary_aligns_global_ids_under_per_pixel_permutations(self):
        q = torch.tensor([[[0.2, 0.7], [0.3, 0.2], [0.5, 0.1]]])
        ids = torch.tensor([[0, 0], [1, 1], [2, 2]])
        expected = boundary_maps(q, ids, (1, 2))
        permutation = torch.tensor([[2, 1], [0, 2], [1, 0]])
        actual = boundary_maps(q.gather(1, permutation[None]), ids.gather(0, permutation), (1, 2))
        for key in expected:
            self.assertTrue(torch.allclose(actual[key], expected[key], atol=1e-7, rtol=1e-6), msg=key)

    def test_boundary_duplicate_ids_and_zero_invalid_slots_do_not_distort_change(self):
        # Both pixels represent global distribution {0: .5, 1: .5}, but the
        # first uses two slots for region 0 and the second one zero-weight slot.
        q = torch.tensor([[[0.2, 0.5], [0.3, 0.5], [0.5, 0.0]]])
        ids = torch.tensor([[0, 0], [0, 1], [1, 0]])
        maps = boundary_maps(q, ids, (1, 2))
        self.assertTrue(torch.allclose(maps["old"], torch.full((1, 1, 1, 2), 0.5)))
        self.assertTrue(torch.allclose(maps["change"], torch.zeros(1, 1, 1, 2), atol=1e-7))


if __name__ == "__main__":
    unittest.main()
