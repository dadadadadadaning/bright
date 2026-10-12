import unittest

import torch

from bda_benchmark.experiments.spi_structural_diffusion_v1.core import (
    StructuralDiffusionConfig,
    residual_to_full_logits,
    structural_diffusion,
)


class StructuralDiffusionTests(unittest.TestCase):
    def setUp(self):
        self.config = StructuralDiffusionConfig(iterations=2, maximum_alpha=0.75)

    @staticmethod
    def probabilities():
        values = torch.tensor(
            [[
                [[0.10, 0.10, 0.10]],
                [[0.80, 0.45, 0.10]],
                [[0.05, 0.40, 0.80]],
                [[0.05, 0.05, 0.00]],
            ]],
            dtype=torch.float32,
        )
        return values / values.sum(1, keepdim=True)

    def test_uniform_diffusion_changes_uncertain_middle_pixel(self):
        probabilities = self.probabilities()
        labels = torch.ones((1, 1, 3), dtype=torch.long)
        refined, _ = structural_diffusion(
            probabilities, labels, mode="uniform", config=self.config
        )
        self.assertFalse(torch.allclose(refined[:, :, 0, 1], probabilities[:, :, 0, 1]))

    def test_different_loccc_labels_block_propagation(self):
        probabilities = self.probabilities()
        labels = torch.tensor([[[1, 0, 2]]])
        refined, _ = structural_diffusion(
            probabilities, labels, mode="uniform", config=self.config
        )
        self.assertTrue(torch.allclose(refined, probabilities, atol=1e-7))

    def test_pixels_outside_loccc_remain_exact(self):
        probabilities = self.probabilities()
        labels = torch.tensor([[[1, 1, 0]]])
        refined, _ = structural_diffusion(
            probabilities, labels, mode="uniform", config=self.config
        )
        self.assertTrue(torch.equal(refined[:, :, 0, 2], probabilities[:, :, 0, 2]))

    def test_modality_disagreement_suppresses_update(self):
        probabilities = self.probabilities()
        labels = torch.ones((1, 1, 3), dtype=torch.long)
        fused = torch.tensor([[[[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]]])
        rgb = fused.clone()
        sar_agree = fused.clone()
        sar_disagree = -fused.clone()
        agree, _ = structural_diffusion(
            probabilities, labels, fused_feature=fused, rgb_feature=rgb,
            sar_feature=sar_agree, mode="modality", config=self.config
        )
        disagree, _ = structural_diffusion(
            probabilities, labels, fused_feature=fused, rgb_feature=rgb,
            sar_feature=sar_disagree, mode="modality", config=self.config
        )
        agree_change = (agree - probabilities).abs().sum()
        disagree_change = (disagree - probabilities).abs().sum()
        self.assertGreater(float(agree_change), float(disagree_change))

    def test_semantic_barrier_preserves_discontinuity_more_than_uniform(self):
        probabilities = self.probabilities()
        labels = torch.ones((1, 1, 3), dtype=torch.long)
        feature = torch.ones((1, 2, 1, 3))
        uniform, _ = structural_diffusion(
            probabilities, labels, mode="uniform", config=self.config
        )
        semantic, _ = structural_diffusion(
            probabilities, labels, fused_feature=feature, rgb_feature=feature,
            sar_feature=feature, mode="semantic", config=self.config
        )
        original_gap = probabilities[0, 1, 0, 0] - probabilities[0, 1, 0, 2]
        uniform_gap = uniform[0, 1, 0, 0] - uniform[0, 1, 0, 2]
        semantic_gap = semantic[0, 1, 0, 0] - semantic[0, 1, 0, 2]
        self.assertLess(float(uniform_gap), float(original_gap))
        self.assertGreater(float(semantic_gap), float(uniform_gap))

    def test_full_residual_is_zero_outside_support(self):
        low = self.probabilities()
        refined = low.clone()
        refined[:, 1, 0, 0] -= 0.1
        refined[:, 2, 0, 0] += 0.1
        full_logits = low.log()
        support = torch.tensor([[[[False, False, False]]]])
        output = residual_to_full_logits(low, refined, full_logits, support)
        self.assertTrue(torch.equal(output, full_logits))


if __name__ == "__main__":
    unittest.main()
