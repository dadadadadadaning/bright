import unittest

import torch

from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import RegionMaps
from bda_benchmark.experiments.spi_s2_distillation_v1.core import (
    CorrectedGBC,
    DistillationConfig,
    StructuralResidualStudent,
    build_structural_batch,
    outside_anchor_kl,
)


class S2DistillationCoreTests(unittest.TestCase):
    @staticmethod
    def regions(labels):
        zeros = torch.zeros_like(labels)
        return RegionMaps(labels=labels, core_labels=zeros, ring_labels=zeros,
                          region_count=int(labels.max().item()))

    def test_confident_region_builds_soft_teacher(self):
        logits = torch.full((1, 4, 2, 2), -3.0)
        logits[:, 2] = 3.0
        feature = torch.randn(1, 8, 2, 2)
        labels = torch.ones((1, 2, 2), dtype=torch.long)
        result = build_structural_batch(logits, feature, self.regions(labels), DistillationConfig())
        self.assertEqual(result.accepted_regions, 1)
        self.assertTrue(torch.all(result.accepted_support == 1))
        self.assertGreater(float(result.teacher_probability[:, 2].mean()), 0.9)

    def test_ambiguous_region_is_not_distilled(self):
        logits = torch.zeros((1, 4, 2, 2))
        feature = torch.randn(1, 8, 2, 2)
        labels = torch.ones((1, 2, 2), dtype=torch.long)
        result = build_structural_batch(logits, feature, self.regions(labels), DistillationConfig())
        self.assertEqual(result.accepted_regions, 0)
        self.assertTrue(torch.allclose(result.teacher_probability, logits.softmax(1), atol=1e-6))

    def test_zero_initialization_preserves_p3(self):
        student = StructuralResidualStudent(feature_channels=8, hidden=8)
        logits = torch.randn(1, 4, 2, 2)
        feature = torch.randn(1, 8, 2, 2)
        labels = torch.ones((1, 2, 2), dtype=torch.long)
        structural = build_structural_batch(logits, feature, self.regions(labels), DistillationConfig())
        residual = student(feature, logits.softmax(1), structural)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_student_cannot_modify_outside_regions(self):
        student = StructuralResidualStudent(feature_channels=8, hidden=8)
        torch.nn.init.ones_(student.output.weight)
        logits = torch.full((1, 4, 2, 2), -3.0)
        logits[:, 1] = 3.0
        feature = torch.randn(1, 8, 2, 2)
        labels = torch.tensor([[[1, 1], [0, 0]]])
        structural = build_structural_batch(logits, feature, self.regions(labels), DistillationConfig())
        residual = student(feature, logits.softmax(1), structural)
        self.assertTrue(torch.equal(residual[:, :, 1], torch.zeros_like(residual[:, :, 1])))

    def test_corrected_gbc_is_2d_and_backward_safe(self):
        module = CorrectedGBC(8)
        value = torch.randn(2, 8, 5, 7, requires_grad=True)
        output = module(value)
        self.assertEqual(output.shape, value.shape)
        output.mean().backward()
        self.assertIsNotNone(value.grad)

    def test_outside_anchor_is_zero_for_identical_logits(self):
        logits = torch.randn(1, 4, 3, 3)
        support = torch.zeros(1, 1, 3, 3)
        loss = outside_anchor_kl(logits, logits, support)
        self.assertAlmostEqual(float(loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
