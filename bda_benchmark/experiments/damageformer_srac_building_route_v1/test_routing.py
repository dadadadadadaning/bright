"""CPU synthetic contracts for sparse 9/18-support building routes."""
import unittest

import torch

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool

from .routing import pool_routes, read_regions, route_from_q


class RoutingTests(unittest.TestCase):
    @staticmethod
    def two_pixels():
        q = torch.zeros(1, 9, 2)
        q[:, 4] = 1.0
        ids = torch.zeros(9, 2, dtype=torch.long)
        building = torch.tensor([[[[0.2, 0.8]]]])
        return q, ids, building

    def test_modes_preserve_original_q_and_probability_mass(self):
        q, ids, building = self.two_pixels()
        before = q.clone()
        routes = [route_from_q(q, ids, 1, (1, 2), building, mode)
                  for mode in ("unsplit", "hard", "soft")]
        self.assertTrue(torch.equal(q, before))
        self.assertEqual([r["q"].shape[1] for r in routes], [9, 18, 18])
        self.assertEqual([r["count"] for r in routes], [1, 2, 2])
        for route in routes:
            torch.testing.assert_close(route["q"].sum(1), torch.ones(1, 2), atol=1e-6, rtol=0)
        hard = routes[1]
        self.assertEqual(hard["ids"][0, 4, 0].item(), 0)
        self.assertEqual(hard["ids"][0, 13, 1].item(), 1)
        torch.testing.assert_close(hard["q"][0, 4], torch.tensor([1., 0.]))
        torch.testing.assert_close(hard["q"][0, 13], torch.tensor([0., 1.]))

    def test_hard_split_separates_opposite_pixels(self):
        q, ids, building = self.two_pixels()
        feature = torch.tensor([[[[1., 9.]]]])
        route = route_from_q(q, ids, 1, (1, 2), building, "hard")
        pooled = pool_routes(feature, route["q"], route["ids"], route["count"])
        torch.testing.assert_close(pooled["tokens"].reshape(-1), torch.tensor([1., 9.]))
        torch.testing.assert_close(pooled["mass"].reshape(-1), torch.tensor([1., 1.]))

    def test_capacity_control_matches_split_capacity_without_using_building(self):
        # One original seed, two spatial halves in each direction: 2K slots,
        # 18 supports, and exactly the same route for contradictory B maps.
        q = torch.zeros(1, 9, 64)
        q[:, 4] = 1.
        ids = torch.zeros(9, 64, dtype=torch.long)
        low_b = torch.zeros(1, 1, 8, 8)
        high_b = torch.ones_like(low_b)
        low = route_from_q(q, ids, 1, (8, 8), low_b, "capacity_control")
        high = route_from_q(q, ids, 1, (8, 8), high_b, "capacity_control")
        self.assertEqual((low["count"], low["q"].shape[1]), (2, 18))
        torch.testing.assert_close(low["q"], high["q"])
        torch.testing.assert_close(low["ids"], high["ids"])
        torch.testing.assert_close(low["q"].sum(1), torch.ones(1, 64), atol=1e-6, rtol=0)
        split = low["q"][0, 13].reshape(8, 8)
        self.assertEqual((split[0, 0].item(), split[0, 4].item(),
                          split[4, 0].item(), split[4, 4].item()), (0., 1., 1., 0.))
        feature = torch.ones(1, 1, 8, 8)
        pooled = pool_routes(feature, low["q"], low["ids"], low["count"])
        torch.testing.assert_close(pooled["mass"], torch.tensor([[32., 32.]]))
        logits = torch.tensor([[[5., 0., 0., 0.], [0., 5., 0., 0.]]])
        torch.testing.assert_close(read_regions(logits, low, (8, 8)),
                                   read_regions(logits, high, (8, 8)))

    def test_soft_split_is_continuous_and_reads_same_route(self):
        q, ids, building = self.two_pixels()
        feature = torch.tensor([[[[1., 9.]]]])
        route = route_from_q(q, ids, 1, (1, 2), building, "soft")
        pooled = pool_routes(feature, route["q"], route["ids"], route["count"])
        # Background receives 0.8*1+0.2*9, building receives 0.2*1+0.8*9.
        torch.testing.assert_close(pooled["tokens"].reshape(-1), torch.tensor([2.6, 7.4]))
        logits = torch.tensor([[[4., 0., 0., 0.], [0., 4., 0., 0.]]])
        probability = read_regions(logits, route, (1, 2))
        self.assertEqual(tuple(probability.shape), (1, 4, 1, 2))
        torch.testing.assert_close(probability.sum(1), torch.ones(1, 1, 2), atol=1e-6, rtol=0)
        self.assertGreater(probability[0, 0, 0, 0], probability[0, 0, 0, 1])

    def test_unsplit_pool_matches_original_nine_support_pool(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.rand(2, 9, 6, generator=generator)
        q = q / q.sum(1, keepdim=True)
        ids = torch.randint(0, 3, (9, 6), generator=generator)
        feature = torch.randn(2, 3, 2, 3, generator=generator)
        route = route_from_q(q, ids, 3, (2, 3), torch.full((2, 1, 2, 3), .5), "unsplit")
        old = soft_pool_unpool(feature, q, ids, 3)
        new = pool_routes(feature, route["q"], route["ids"], route["count"])
        torch.testing.assert_close(new["mass"], old["mass"], atol=1e-6, rtol=0)
        torch.testing.assert_close(new["tokens"], old["tokens"], atol=1e-6, rtol=0)

    def test_rejects_invalid_or_nonnormalized_q(self):
        q, ids, building = self.two_pixels()
        with self.assertRaises(ValueError):
            route_from_q(q * .5, ids, 1, (1, 2), building, "hard")
        with self.assertRaises(ValueError):
            route_from_q(q, ids, 1, (1, 2), building, "soft", soft_floor=.5)


if __name__ == "__main__":
    unittest.main()
