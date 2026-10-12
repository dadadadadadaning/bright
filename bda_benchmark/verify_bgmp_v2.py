# -*- coding: utf-8 -*-
"""Fast structural checks for DamagePrototypeMemoryV2."""
from __future__ import annotations

import sys
from pathlib import Path

import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# The structural test only needs DamagePrototypeMemoryV2. Provide a tiny stub
# for the parent DamageFormer class so the V2 module can be imported without
# copying the whole BRIGHT repository into this bundle.
stub = types.ModuleType("DamageFormer_BG")
class _DamageFormerBGStub(torch.nn.Module):
    pass
stub.DamageFormer_BG = _DamageFormerBGStub
sys.modules["DamageFormer_BG"] = stub

from DamageFormer_BGMP_V2 import DamagePrototypeMemoryV2


def main() -> None:
    torch.manual_seed(17)

    # K=1: direct 256-D space, mean update, warm-up behavior.
    mem1 = DamagePrototypeMemoryV2(
        in_channels=256,
        prototypes_per_class=1,
        momentum=0.9,
        tau=0.1,
        warmup_iters=5,
    )
    assert isinstance(mem1.feature_proj, torch.nn.Identity)
    assert mem1.proto_dim == 256
    assert mem1.resolved_assignment_mode() == "mean"

    feature = torch.randn(2, 256, 4, 4, requires_grad=True)
    building = torch.rand(2, 1, 4, 4)
    target = torch.randint(1, 4, (2, 16, 16))

    out0, stats0 = mem1(feature, building, target, current_iter=0)
    assert not stats0["proto_active"]
    assert tuple(mem1.prototypes.shape) == (3, 1, 256)
    assert bool(mem1.prototype_initialized.all().item())

    out1, stats1 = mem1(feature, building, target, current_iter=5)
    assert stats1["proto_active"]
    loss = out1.square().mean()
    loss.backward()
    assert feature.grad is not None and float(feature.grad.abs().sum()) > 0
    print("[PASS] K1 direct-256 space, memory init, warm-up, backward")

    # K=3: hard assignment and natural usage counts.
    mem3 = DamagePrototypeMemoryV2(
        in_channels=256,
        prototypes_per_class=3,
        momentum=0.9,
        tau=0.1,
        warmup_iters=0,
        assignment_mode="auto",
    )
    assert mem3.resolved_assignment_mode() == "hard"
    feat3 = F.normalize(torch.randn(2, 256, 8, 8), dim=1)
    target3 = torch.randint(1, 4, (2, 32, 32))
    mem3.update_memory(feat3, target3)
    assert tuple(mem3.prototypes.shape) == (3, 3, 256)
    assert bool(mem3.prototype_initialized.all().item())
    assert torch.allclose(mem3.prototypes.norm(dim=-1), torch.ones(3, 3), atol=1e-5)
    usage = mem3.prototype_usage
    assert float(usage.sum()) > 0
    print("[PASS] K3 spherical hard assignment, normalized prototypes, natural usage")

    print("ALL BGMP-V2 STRUCTURAL CHECKS PASSED")


if __name__ == "__main__":
    main()
