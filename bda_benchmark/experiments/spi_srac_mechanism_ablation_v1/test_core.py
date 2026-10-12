from __future__ import annotations

import torch

from .core import (
    EvidencePooler, RetrievalAugmentedClassifier, select_region_positions, tokens_at,
)


def test_region_sampler_keeps_strong_anomaly_and_bound() -> None:
    score = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    mask = torch.ones(5, 5, dtype=torch.bool)
    positions, valid = select_region_positions(score, mask, maximum=8)
    assert int(valid.sum()) == 8
    assert 24 in positions[valid].tolist()


def test_enhanced_tokens_follow_frozen_b2_gamma() -> None:
    payload = {
        "tokens": {
            "rgb_structure": torch.zeros(2, 3, 4),
            "base_change": torch.ones(2, 3, 4),
            "explicit_diff": torch.full((2, 3, 4), 2.0),
        },
        "token_mask": torch.ones(2, 3, dtype=torch.bool),
        "metadata": {"diff_gamma_effective": 0.25},
    }
    values, mask = tokens_at(payload, torch.tensor([0, 1]), torch.device("cpu"))
    assert torch.allclose(values["enhanced_change"], torch.full((2, 3, 4), 1.5))
    assert bool(mask.all())


def test_pooler_variants_have_stable_shapes() -> None:
    tokens = {
        "rgb_structure": torch.randn(4, 6, 8),
        "base_change": torch.randn(4, 6, 8),
        "explicit_diff": torch.randn(4, 6, 8),
        "enhanced_change": torch.randn(4, 6, 8),
    }
    mask = torch.ones(4, 6, dtype=torch.bool)
    for pool in ("p0_mean", "p1_stats", "p2_attn_skip", "p3_attn_tail"):
        model = EvidencePooler(
            {name: 8 for name in tokens}, "c4_structural", pool,
            hidden_dim=32, key_dim=16,
        )
        key, logits, diagnostics = model(tokens, mask)
        assert key.shape == (4, 16)
        assert logits.shape == (4, 3)
        assert diagnostics["attention_entropy"].shape == (4,)
        assert torch.allclose(key.norm(dim=1), torch.ones(4), atol=1e-5)


def test_rag_variants_have_stable_shapes() -> None:
    query = torch.randn(3, 16)
    neighbours = torch.randn(3, 5, 16)
    labels = torch.randint(1, 4, (3, 5))
    geometry = torch.randn(3, 5, 6)
    events = torch.randint(0, 4, (3, 5))
    for variant in ("x0_keys", "x1_label_geometry", "x2_label_geometry_event"):
        model = RetrievalAugmentedClassifier(16, variant, 6, 4, hidden_dim=24, heads=4)
        assert model(query, neighbours, labels, geometry, events).shape == (3, 3)

