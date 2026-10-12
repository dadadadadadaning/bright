from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.DamageFormer_S7_SIRM_M2B import RetrievalConfig, StaticBuildingInstanceMemory
from model.DamageFormer_S7_SIRM_M3 import (
    ProjectionClassifier,
    ProjectionConfig,
    ProjectedStaticBuildingInstanceMemory,
    ResidualMetricProjection,
    load_projection_checkpoint,
    save_projection_checkpoint,
)


class S7SIRMM3Tests(unittest.TestCase):
    def test_projection_shape_norm_and_gradient(self):
        torch.manual_seed(1)
        config = ProjectionConfig(input_dim=8, hidden_dim=4, dropout=0.0)
        projector = ResidualMetricProjection(config)
        features = torch.randn(6, 8)
        projected = projector(features)
        self.assertEqual(projected.shape, features.shape)
        self.assertTrue(
            torch.allclose(projected.norm(dim=1), torch.ones(6), atol=1e-6)
        )
        loss = -projected[:, 0].mean() + 0.01 * projector.gamma.square()
        loss.backward()
        self.assertIsNotNone(projector.fc2.weight.grad)
        self.assertTrue(torch.isfinite(projector.fc2.weight.grad).all())

    def test_projected_memory_projects_keys_and_queries(self):
        config = ProjectionConfig(input_dim=3, hidden_dim=2, dropout=0.0)
        projector = ResidualMetricProjection(config)
        with torch.no_grad():
            projector.fc2.weight.zero_()
            projector.fc2.bias.zero_()
        raw = StaticBuildingInstanceMemory(
            torch.eye(3),
            torch.tensor([0, 1, 2]),
            metadata={"unit_test": True},
        )
        memory = ProjectedStaticBuildingInstanceMemory(raw, projector)
        result = memory.retrieve(
            torch.tensor([[0.0, 1.0, 0.0]]),
            RetrievalConfig(
                topk=1,
                alpha=0.1,
                similarity_threshold=0.0,
                purity_threshold=0.0,
                margin_threshold=0.0,
            ),
        )
        self.assertEqual(int(result["cache_prediction"][0].item()), 1)
        self.assertAlmostEqual(float(result["best_similarity"][0].item()), 1.0, places=6)

    def test_projection_checkpoint_roundtrip(self):
        config = ProjectionConfig(input_dim=8, hidden_dim=4, dropout=0.0)
        projector = ResidualMetricProjection(config)
        classifier = ProjectionClassifier(8)
        features = F.normalize(torch.randn(3, 8), dim=1)
        expected = projector(features).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.pt"
            save_projection_checkpoint(
                path,
                projector,
                classifier,
                metadata={"source_cache_sha256": "abc"},
            )
            loaded, metadata = load_projection_checkpoint(path, torch.device("cpu"))
            actual = loaded(features)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertEqual(metadata["source_cache_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
