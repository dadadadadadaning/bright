"""CPU synthetic same-pass and frozen-front contracts for routed readouts."""
import unittest

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    SuperpixelEmbedding, project_region_probabilities,
)

from .model import BuildingRouteHead, FrozenFrontCapture


class TinyAdapter(nn.Module):
    def forward(self, raw, building, boundary, entropy):
        return raw, raw


class TinyFront(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = TinyAdapter()
        self.embedding = SuperpixelEmbedding(content_dim=4, adapted_dim=4, dim=4)
        self.region_classifier = nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, pre, post):
        batch, _, height, width = pre.shape
        n = height * width
        raw = pre * self.scale
        building = torch.sigmoid(pre[:, :1])
        q = raw.new_zeros(batch, 9, n)
        q[:, 4] = 1.
        ids = torch.zeros(9, n, dtype=torch.long, device=pre.device)
        adapted, _ = self.adapter(raw, building, building, building)
        tokens, mass, geometry, valid = self.embedding(adapted, raw, q, ids, (height, width), 1)
        logits = self.region_classifier(tokens)
        prob = project_region_probabilities(logits, q, ids, (height, width), (height, width))
        return dict(q=q, ids=ids, count=1, shape=(height, width),
                    adapted_content=adapted, tokens=tokens, mass=mass,
                    geometry=geometry, valid_regions=valid, prob=prob)


class ModelTests(unittest.TestCase):
    def test_frozen_capture_and_unsplit_readout_match_original(self):
        front = TinyFront()
        head = BuildingRouteHead(front, "unsplit")
        image = torch.randn(2, 4, 2, 3)
        capture = FrozenFrontCapture(front)
        try:
            source = capture(image, image)
        finally:
            capture.close()
        torch.testing.assert_close(source["fused_roi"], image)
        torch.testing.assert_close(source["building"], torch.sigmoid(image[:, :1]))
        routed = head.from_frozen(**source)
        torch.testing.assert_close(routed["prob"], source["result"]["prob"], atol=1e-6, rtol=1e-6)

    def test_one_frozen_pass_can_feed_three_trainable_heads(self):
        front = TinyFront()
        image = torch.randn(2, 4, 2, 3)
        capture = FrozenFrontCapture(front)
        try:
            source = capture(image, image)
        finally:
            capture.close()
        heads = {mode: BuildingRouteHead(front, mode) for mode in ("unsplit", "hard", "soft")}
        for mode, head in heads.items():
            head.train()
            self.assertFalse(front.training)
            self.assertTrue(head.embedding.training)
            output = head.from_frozen(**source)
            self.assertEqual(output["region_logits"].shape[1], 1 if mode == "unsplit" else 2)
            torch.testing.assert_close(output["prob"].sum(1),
                                       torch.ones_like(output["prob"][:, 0]), atol=1e-6, rtol=0)
            output["prob"][:, 0].sum().backward()
            self.assertTrue(any(p.grad is not None for p in head.trainable_parameters()))
        self.assertIsNone(front.scale.grad)

    def test_readout_checkpoint_excludes_source_front(self):
        front = TinyFront()
        head = BuildingRouteHead(front, "hard")
        saved = head.head_state_dict()
        self.assertEqual(set(saved), {"embedding", "region_classifier", "mode", "threshold", "soft_floor"})
        self.assertFalse(any(key.startswith("front.") for key in saved))
        other = BuildingRouteHead(front, "hard")
        other.load_head_state_dict(saved)
        with self.assertRaises(ValueError):
            BuildingRouteHead(front, "soft").load_head_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
