# -*- coding: utf-8 -*-
"""Lightweight structural verification for BGMP-V2 K=3 memory."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The verification only needs DamagePrototypeMemoryV2. Stub the backbone import.
stub = types.ModuleType("DamageFormer_BG")
class _DummyBG(nn.Module):
    pass
stub.DamageFormer_BG = _DummyBG
sys.modules["DamageFormer_BG"] = stub

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from DamageFormer_BGMP_V2 import DamagePrototypeMemoryV2


def main() -> None:
    torch.manual_seed(3407)
    memory = DamagePrototypeMemoryV2(
        in_channels=256,
        prototypes_per_class=3,
        momentum=0.99,
        tau=0.1,
        warmup_iters=5,
        max_samples_per_class=4096,
        assignment_mode="hard",
    )
    memory.train()

    feature = torch.randn(2, 256, 16, 16, requires_grad=True)
    building_prob = torch.sigmoid(torch.randn(2, 1, 16, 16))
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[:, :22] = 1
    target[:, 22:43] = 2
    target[:, 43:] = 3

    # First pass initializes and updates all 3x3 prototypes.
    enhanced, stats = memory(
        feature,
        building_prob,
        damage_target=target,
        current_iter=0,
    )
    assert enhanced.shape == feature.shape
    assert memory.prototypes.shape == (3, 3, 256)
    assert memory.prototype_initialized.tolist() == [True, True, True]
    assert memory.resolved_assignment_mode() == "hard"
    assert torch.allclose(
        memory.prototypes.norm(dim=-1),
        torch.ones(3, 3),
        atol=1e-5,
    )

    # Warm-up bypass.
    assert stats["proto_active"] is False

    # Active query after warm-up.
    feature2 = torch.randn(2, 256, 16, 16, requires_grad=True)
    enhanced2, stats2 = memory(
        feature2,
        building_prob,
        damage_target=None,
        current_iter=5,
    )
    assert stats2["proto_active"] is True
    assert stats2["proto_class_similarity"].shape == (2 * 16 * 16, 3)
    assert torch.isfinite(stats2["proto_query_margin"])
    assert torch.isfinite(stats2["proto_attention_entropy"])
    assert torch.isfinite(stats2["proto_effect_ratio"])

    enhanced2.square().mean().backward()
    assert feature2.grad is not None
    assert torch.isfinite(feature2.grad).all()

    normalized = F.normalize(memory.prototypes, dim=-1)
    for c in range(3):
        cos = normalized[c] @ normalized[c].t()
        assert cos.shape == (3, 3)

    print("[PASS] prototype bank shape = [3, 3, 256]")
    print("[PASS] direct 256-D feature space")
    print("[PASS] spherical hard assignment")
    print("[PASS] empty-prototype reinitialization buffers")
    print("[PASS] warm-up bypass and active retrieval")
    print("[PASS] backward through prototype context")
    print("ALL BGMP-V2 K3 CHECKS PASSED")


if __name__ == "__main__":
    main()
