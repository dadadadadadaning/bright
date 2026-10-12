import unittest

import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.heads import MultiScaleCenterHead


class MultiScaleCenterHeadTests(unittest.TestCase):
    def test_all_ablation_modes_use_high_resolution_output(self):
        low = torch.randn(2, 8, 4, 4)
        high = torch.randn(2, 6, 8, 8)
        for mode in ("low", "high", "dual"):
            with self.subTest(mode=mode):
                head = MultiScaleCenterHead(8, 6, hidden_channels=4, mode=mode)
                output = head(low, high)
                self.assertEqual(output["center_logits"].shape, (2, 1, 8, 8))

    def test_dual_alpha_is_bounded_and_trainable(self):
        head = MultiScaleCenterHead(8, 6, hidden_channels=4, mode="dual", alpha=0.3)
        output = head(torch.randn(1, 8, 4, 4), torch.randn(1, 6, 8, 8))
        self.assertAlmostEqual(output["center_alpha"].item(), 0.3, places=6)
        output["center_logits"].mean().backward()
        self.assertIsNotNone(head.alpha_logit.grad)


if __name__ == "__main__":
    unittest.main()
