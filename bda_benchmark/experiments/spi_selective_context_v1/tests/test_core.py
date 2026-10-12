from __future__ import annotations

import unittest

import torch

from bda_benchmark.experiments.spi_selective_context_v1.core import (
    SelectiveContextConfig,
    foreground_region_distribution,
    interior_weight,
    normalized_entropy,
    selective_context_refinement,
)


class SelectiveContextTests(unittest.TestCase):
    def test_entropy_is_normalized(self):
        uniform = torch.full((4, 2, 2), 0.25)
        confident = torch.zeros((4, 2, 2))
        confident[1] = 1.0
        self.assertTrue(torch.allclose(normalized_entropy(uniform), torch.ones(2, 2)))
        self.assertLess(float(normalized_entropy(confident).max()), 1e-5)

    def test_foreground_context_cannot_create_background(self):
        probabilities = torch.full((4, 2, 2), 0.05)
        probabilities[0] = 0.80
        probabilities[2] = 0.10
        context = foreground_region_distribution(
            probabilities, torch.ones((2, 2), dtype=torch.bool)
        )
        self.assertEqual(float(context[0]), 0.0)
        self.assertAlmostEqual(float(context.sum()), 1.0, places=6)

    def test_boundary_is_downweighted(self):
        mask = torch.ones((5, 5), dtype=torch.bool)
        weight = interior_weight(mask, radius=2, boundary_floor=0.2)
        self.assertAlmostEqual(float(weight[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(weight[2, 2]), 1.0, places=6)

    def test_bbox_interior_weight_matches_full_image_reference(self):
        mask = torch.zeros((17, 19), dtype=torch.bool)
        mask[4:13, 6:15] = True
        mask[4:7, 6:9] = False
        radius = 3
        floor = 0.2
        expected = torch.zeros_like(mask, dtype=torch.float32)
        expected[mask] = floor
        current = mask.float()[None, None]
        for depth in range(1, radius + 1):
            current = (
                torch.nn.functional.avg_pool2d(
                    current, 3, stride=1, padding=1
                )
                == 1.0
            ).float()
            expected[current[0, 0].bool()] = (
                floor + (1.0 - floor) * depth / radius
            )
        observed = interior_weight(mask, radius=radius, boundary_floor=floor)
        self.assertTrue(torch.equal(observed, expected))

    def test_refinement_uses_no_target_and_changes_uncertain_pixel(self):
        probabilities = torch.zeros((4, 5, 5))
        probabilities[1] = 0.80
        probabilities[2] = 0.10
        probabilities[3] = 0.10
        probabilities[:, 2, 2] = torch.tensor([0.0, 0.34, 0.36, 0.30])
        prediction = probabilities.argmax(dim=0)
        mask = torch.ones((5, 5), dtype=torch.bool)
        config = SelectiveContextConfig(
            minimum_purity=0.70,
            minimum_margin=0.10,
            blend_strength=1.0,
            uncertainty_threshold=0.2,
            uncertainty_temperature=0.05,
            interior_radius=1,
        )
        output, rows = selective_context_refinement(
            prediction, probabilities, [mask], config=config, mode="uncertain_interior"
        )
        self.assertTrue(rows[0]["accepted"])
        self.assertEqual(int(output[2, 2]), 1)


if __name__ == "__main__":
    unittest.main()
