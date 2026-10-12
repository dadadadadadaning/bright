# -*- coding: utf-8 -*-
"""Fast structural checks for S1/S2 and the paper-equivalent schedule."""
from __future__ import annotations

import math

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

import model.DamageFormer_BG as bg_module
from model.DamageFormer_BGMP import DamagePrototypeMemory


class DummyFeatureExtractor(nn.Module):
    @staticmethod
    def _make(x: torch.Tensor, channels: int, divisor: int) -> torch.Tensor:
        h = max(1, x.shape[-2] // divisor)
        w = max(1, x.shape[-1] // divisor)
        base = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (h, w))
        feat = base.repeat(1, channels, 1, 1)
        return feat.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor):
        return [
            self._make(x, 96, 4),
            self._make(x, 96, 4),
            self._make(x, 192, 8),
            self._make(x, 384, 16),
            self._make(x, 768, 32),
        ]


bg_module.swin_t = lambda weights=None: nn.Identity()
bg_module.SwinTransformerFeatureExtractor = lambda *args, **kwargs: DummyFeatureExtractor()
DamageFormer_BG = bg_module.DamageFormer_BG


def check_schedule() -> None:
    configs = [
        ("S1 2x4090 DDP", 4, 2, 2),
        ("S2 1x5090", 4, 1, 4),
    ]
    opt_steps = 50_000
    for name, local_bs, world_size, accum in configs:
        global_micro_bs = local_bs * world_size
        effective_bs = global_micro_bs * accum
        micro_steps = opt_steps * accum
        total_crops = opt_steps * effective_bs
        print(
            f"[SCHEDULE] {name}: local_bs={local_bs}, world={world_size}, "
            f"global_micro_bs={global_micro_bs}, acc={accum}, effective_bs={effective_bs}, "
            f"micro_steps={micro_steps}, optimizer_steps={opt_steps}, total_crops={total_crops}"
        )
        assert effective_bs == 16
        assert total_crops == 800_000


def check_s1() -> None:
    torch.manual_seed(0)
    pre = torch.randn(2, 3, 8, 8)
    post = torch.randn(2, 3, 8, 8)

    model = DamageFormer_BG(roi_alpha=1.0, use_pretrained_swin=False).eval()
    with torch.no_grad():
        _, dam, aux = model(pre, post, return_aux=True)
        expected = F.interpolate(
            model.clf_2(aux["final_feat_fused"]),
            size=post.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    feature_diff = (aux["final_feat_fused"] - aux["final_feat_fused_roi"]).abs().max().item()
    damage_diff = (dam - expected).abs().max().item()
    assert feature_diff < 1e-7 and damage_diff < 1e-7
    print(f"[S1] alpha=1 identity bypass: PASS feature_diff={feature_diff:.3e}, damage_diff={damage_diff:.3e}")

    model.roi_alpha = 0.1
    model.train()
    _, dam, aux = model(pre, post, return_aux=True)
    assert float(aux["roi_weight"].min()) >= 0.1 - 1e-6
    assert float(aux["roi_weight"].max()) <= 1.0 + 1e-6
    model.zero_grad(set_to_none=True)
    F.cross_entropy(dam, torch.randint(0, 4, (2, 8, 8))).backward()
    grad = model.clf_1.weight.grad
    assert grad is None or torch.count_nonzero(grad).item() == 0
    print("[S1] ROI range and damage-loss -> clf_1 detach isolation: PASS")


def make_targets(batch: int, h: int, w: int) -> torch.Tensor:
    target = torch.ones(batch, h, w, dtype=torch.long)
    target[:, :, w // 3 : 2 * w // 3] = 2
    target[:, :, 2 * w // 3 :] = 3
    return target


def check_s2() -> None:
    torch.manual_seed(1)
    acc = 4
    m_step = 0.99
    m_micro = m_step ** (1.0 / acc)
    assert math.isclose(m_micro**acc, m_step, rel_tol=1e-7, abs_tol=1e-7)
    print(f"[S2] EMA conversion: m_step={m_step}, m_micro={m_micro:.8f}, m_micro^ACC={m_micro**acc:.8f}")

    memory = DamagePrototypeMemory(
        in_channels=16,
        proto_dim=8,
        prototypes_per_class=1,
        momentum=m_micro,
        tau=0.1,
        warmup_iters=5000,
        max_samples_per_class=32,
    ).train()
    feature = torch.randn(2, 16, 4, 4, requires_grad=True)
    building_prob = torch.ones(2, 1, 4, 4)
    target = make_targets(2, 4, 4)

    before = memory.prototypes.detach().clone()
    _, stats0 = memory(feature, building_prob, damage_target=target, current_iter=0)
    after = memory.prototypes.detach().clone()
    assert bool(stats0["proto_initialized"].all().item())
    assert stats0["proto_active"] is False
    assert not torch.allclose(before, after)
    norms = after.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("[S2] K=1 memories initialize/update/normalize during warm-up: PASS")

    feature.grad = None
    enhanced, stats1 = memory(feature, building_prob, damage_target=target, current_iter=5000)
    assert stats1["proto_active"] is True
    assert stats1["proto_class_similarity"] is not None
    enhanced.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    print("[S2] optimizer-step warm-up activation, retrieval, and autograd: PASS")

    usage = stats1["proto_usage_ratio"]
    assert usage.shape == (3, 1)
    assert torch.allclose(usage, torch.ones_like(usage), atol=1e-6)
    print("[S2] K=1 usage=[1.0] per class is expected and is NOT a collapse diagnostic: PASS")


if __name__ == "__main__":
    check_schedule()
    check_s1()
    check_s2()
    print("ALL S1/S2 SCHEDULE AND MECHANISM CHECKS PASSED")
