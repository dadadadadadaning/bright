import unittest

import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.objectives import (
    GeometryLossWeights,
    PseudoInstanceJointObjective,
    aligned_equivariance_loss,
    anchor_kl_loss,
    compose_region_quality,
    quality_weighted_geometry_loss,
    quality_weighted_region_loss,
)


class ObjectiveTests(unittest.TestCase):
    def test_quality_is_product_and_tierable(self):
        self.assertAlmostEqual(
            compose_region_quality(shape=0.5, center=0.8, boundary=0.5, semantic=1.0),
            0.2,
        )

    def test_geometry_loss_is_instance_equal_not_area_weighted(self):
        result = quality_weighted_geometry_loss(
            center_losses=[torch.tensor(1.0), torch.tensor(3.0)],
            offset_losses=[torch.tensor(10.0), torch.tensor(10.0)],
            boundary_losses=[torch.tensor(10.0), torch.tensor(10.0)],
            qualities=[1.0, 1.0],
            weights=GeometryLossWeights(center=1.0),
        )
        self.assertAlmostEqual(result.item(), 2.0)

    def test_equivariance_is_zero_for_identical_logits(self):
        logits = torch.randn(2, 1, 5, 5)
        self.assertAlmostEqual(
            aligned_equivariance_loss(logits, logits.clone()).item(), 0.0, places=7
        )

    def test_region_loss_rewards_the_region_label(self):
        mask = torch.ones((3, 3), dtype=torch.bool)
        good = torch.zeros((4, 3, 3))
        bad = torch.zeros((4, 3, 3))
        good[2] = 4.0
        bad[1] = 4.0
        good_loss = quality_weighted_region_loss(good, [mask], [2], [1.0])
        bad_loss = quality_weighted_region_loss(bad, [mask], [2], [1.0])
        self.assertLess(good_loss.item(), bad_loss.item())

    def test_zero_quality_returns_differentiable_zero(self):
        logits = torch.randn(4, 2, 2, requires_grad=True)
        loss = quality_weighted_region_loss(
            logits, [torch.ones((2, 2), dtype=torch.bool)], [1], [0.0]
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(logits.grad)

    def test_identical_teacher_has_zero_anchor_loss(self):
        logits = torch.randn(1, 4, 3, 3)
        unsafe = torch.ones((1, 3, 3), dtype=torch.bool)
        self.assertAlmostEqual(
            anchor_kl_loss(logits, logits.clone(), unsafe).item(), 0.0, places=6
        )

    def test_joint_objective_has_finite_backward(self):
        logits = torch.randn(1, 4, 4, 4, requires_grad=True)
        target = torch.randint(0, 4, (1, 4, 4))
        mask = torch.zeros((4, 4), dtype=torch.bool)
        mask[1:3, 1:3] = True
        center1 = torch.randn(1, 1, 2, 2, requires_grad=True)
        center2 = torch.randn(1, 1, 2, 2, requires_grad=True)
        losses = PseudoInstanceJointObjective()(
            semantic_logits=logits,
            semantic_target=target,
            teacher_logits=logits.detach(),
            unsafe_mask=torch.ones((1, 4, 4), dtype=torch.bool),
            region_masks=[[mask]],
            region_labels=[[2]],
            region_qualities=[[0.8]],
            center_logits_view1=center1,
            center_logits_view2_aligned=center2,
        )
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
