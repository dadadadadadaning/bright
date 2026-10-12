"""Grouped-reduction contracts. Synthetic checks are not real BRIGHT evidence."""
from __future__ import annotations

import unittest

import torch

from .feature_core import mean_pool, unpool
from .grouped_pool import GroupedRouting


def dense_reference(feature, q, ids, count):
    """Independent CPU dense routing oracle, suitable only for small fixtures."""
    feature, q, ids = feature.detach().cpu(), q.detach().cpu(), ids.detach().cpu()
    dtype = torch.float64 if feature.dtype == torch.float64 else torch.float32
    rows = feature.to(dtype).flatten(2).transpose(1, 2)
    dense = torch.zeros(q.shape[0], q.shape[2], count, dtype=dtype)
    for support in range(q.shape[1]):
        for pixel in range(q.shape[2]):
            dense[:, pixel, int(ids[support, pixel])] += q[:, support, pixel].to(dtype)
    mass = dense.sum(1)
    numerator = dense.transpose(1, 2).bmm(rows)
    return numerator / torch.where(mass > 0, mass, torch.ones_like(mass))[..., None], mass


def cpu_scatter_reference(feature, q, ids, count):
    """CPU-only independent oracle for the realistic-size CUDA test."""
    feature, q, ids = feature.detach().cpu(), q.detach().cpu(), ids.detach().cpu()
    rows = feature.float().flatten(2).transpose(1, 2)
    numerator = torch.zeros(rows.shape[0], count, rows.shape[-1])
    mass = torch.zeros(rows.shape[0], count, 1)
    for support in range(q.shape[1]):
        index = ids[support][None, :, None]
        weight = q[:, support, :, None].float()
        numerator.scatter_add_(1, index.expand_as(rows), rows*weight)
        mass.scatter_add_(1, index.expand_as(weight), weight)
    return numerator / torch.where(mass > 0, mass, torch.ones_like(mass)), mass[..., 0]


def fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(2718)
    # Transposes retain noncontiguous strides for all three input tensors.
    feature = torch.randn(2, 7, 4, 3, dtype=dtype, generator=generator).transpose(2, 3)
    q = torch.rand(2, 12, 9, dtype=dtype, generator=generator).transpose(1, 2)
    q = q / q.sum(1, keepdim=True)
    ids = torch.randint(0, 8, (12, 9), generator=generator).transpose(0, 1)
    return feature, q, ids, 11


class GroupedPoolContracts(unittest.TestCase):
    def test_noncontiguous_batch_two_matches_dense_float32_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                feature, q, ids, count = fixture(dtype)
                self.assertFalse(feature.is_contiguous())
                self.assertFalse(ids.is_contiguous())
                plan = GroupedRouting(ids, count, max_work_elements=256, regions_per_block=2)
                actual, mass = mean_pool(feature, q, ids, count, routing=plan)
                expected, expected_mass = dense_reference(feature, q, ids, count)
                tolerance = 2e-6 if dtype == torch.float32 else 1e-12
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(mass, expected_mass, atol=tolerance, rtol=tolerance)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, feature.device)

    def test_all_candidate_routes_collapse_to_one_region_and_empty_stays_zero(self):
        feature, q, ids, count = fixture()
        ids.zero_()
        plan = GroupedRouting(ids, count, max_work_elements=128, regions_per_block=2)
        token, mass = mean_pool(feature, q, ids, count, routing=plan)
        torch.testing.assert_close(token[:, 0], feature.mean((2, 3)))
        torch.testing.assert_close(mass[:, 0], torch.full((2,), 12.))
        self.assertTrue(bool((token[:, 1:] == 0).all()))
        self.assertTrue(bool((mass[:, 1:] == 0).all()))

    def test_zero_and_tiny_positive_mass_are_not_clamped_or_discarded(self):
        feature = torch.tensor([[[[2., 6., 10., 14.]], [[-1., 3., -2., 8.]]]])
        tiny = torch.full((1, 4), 1e-12)
        q = torch.stack((1-tiny, tiny, torch.zeros_like(tiny)), 1)
        ids = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        plan = GroupedRouting(ids, 4, max_work_elements=16, regions_per_block=1)
        token, mass = mean_pool(feature, q, ids, 4, routing=plan)
        self.assertGreater(float(mass[0, 1]), 0.)
        self.assertLess(float(mass[0, 1]), 1e-6)
        torch.testing.assert_close(token[0, 1], feature.mean((0, 2, 3)))
        self.assertTrue(bool((mass[:, 2:] == 0).all()))
        self.assertTrue(bool((token[:, 2:] == 0).all()))

    def test_edge_clamped_duplicate_candidates_preserve_constants(self):
        ids = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
                            [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
                            [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]])
        q = torch.full((1, 9, 4), 1/9)
        feature = torch.full((1, 5, 2, 2), 7.)
        token, mass = mean_pool(feature, q, ids, 3)
        torch.testing.assert_close(token[:, :2], torch.full((1, 2, 5), 7.))
        torch.testing.assert_close(unpool(token, q, ids, (2, 2)), feature)
        torch.testing.assert_close(mass.sum(1), torch.tensor([4.]))

    def test_fixed_plan_repeated_reductions_are_bitwise_equal_and_read_only(self):
        feature, q, ids, count = fixture()
        originals = tuple(value.clone() for value in (feature, q, ids))
        plan = GroupedRouting(ids, count, max_work_elements=256, regions_per_block=2)
        first = mean_pool(feature, q, ids, count, routing=plan)
        for _ in range(3):
            result = mean_pool(feature, q, ids, count, routing=plan)
            self.assertTrue(torch.equal(first[0], result[0]))
            self.assertTrue(torch.equal(first[1], result[1]))
        for original, current in zip(originals, (feature, q, ids)):
            self.assertTrue(torch.equal(original, current))

    def test_route_plan_must_match_the_current_ids(self):
        feature, q, ids, count = fixture()
        plan = GroupedRouting(ids, count)
        changed = ids.clone()
        changed[0, 0] = (int(changed[0, 0])+1) % count
        with self.assertRaises(ValueError):
            mean_pool(feature, q, changed, count, routing=plan)

    def test_route_plan_does_not_alias_in_place_edited_cpu_ids(self):
        feature, q, ids, count = fixture()
        ids = ids.contiguous()
        plan = GroupedRouting(ids, count)
        ids[0, 0] = (int(ids[0, 0])+1) % count
        with self.assertRaises(ValueError):
            mean_pool(feature, q, ids, count, routing=plan)

    def test_invalid_routes_are_rejected_before_reduction(self):
        feature, q, ids, count = fixture()
        for invalid in (-1, count):
            bad = ids.clone()
            bad[0, 0] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    mean_pool(feature, q, bad, count)
        q_bad = q.clone()
        q_bad[0, 0, 0] = -.1
        with self.assertRaises(ValueError):
            mean_pool(feature, q_bad, ids, count)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; server synthetic probe still required')
class GroupedPoolCUDAContracts(unittest.TestCase):
    def test_realistic_256_channel_grid_deterministic_repeat_and_cpu_oracle(self):
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            generator = torch.Generator().manual_seed(31415)
            side, stride = 256, 8
            rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
            seed_y, seed_x = rows.flatten()//stride, cols.flatten()//stride
            grid = side//stride
            ids = torch.stack([((seed_y+dy).clamp(0, grid-1)*grid+
                                (seed_x+dx).clamp(0, grid-1))
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            feature = torch.randn(1, 256, side, side, generator=generator)
            feature[:, 0].fill_(7.)
            q = torch.rand(1, 9, side*side, generator=generator)
            q /= q.sum(1, keepdim=True)
            expected, expected_mass = cpu_scatter_reference(feature[:, :3], q, ids, grid*grid)
            feature, q, ids = feature.cuda(), q.cuda(), ids.cuda()
            plan = GroupedRouting(ids, grid*grid, max_work_elements=2**22, regions_per_block=32)
            first, mass = mean_pool(feature, q, ids, grid*grid, routing=plan)
            torch.cuda.synchronize()
            torch.testing.assert_close(first[..., :3].cpu(), expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(mass.cpu(), expected_mass, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(first[..., 0], torch.full_like(first[..., 0], 7.),
                                       atol=2e-5, rtol=2e-5)
            for _ in range(2):
                repeated, repeated_mass = mean_pool(feature, q, ids, grid*grid, routing=plan)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(first, repeated))
                self.assertTrue(torch.equal(mass, repeated_mass))
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)


if __name__ == '__main__':
    unittest.main()
