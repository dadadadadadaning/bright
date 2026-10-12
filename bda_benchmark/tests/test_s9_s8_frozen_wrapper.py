#!/usr/bin/env python3
"""Synthetic strict-freeze semantics test for the S8 wrapper."""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.s8_backbone import FrozenS8Backbone


class FakeS8(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 1)
        self.fuse = nn.Conv2d(16, 16, 1)

    def forward(self, pre, post, return_aux=False):
        rgb = self.encoder(pre)
        sar = self.encoder(post)
        fused = self.fuse(torch.cat([rgb, sar], dim=1))
        loc = torch.cat([rgb[:, :1], sar[:, :1]], dim=1)
        damage = torch.cat([fused[:, :4]], dim=1)
        if return_aux:
            return loc, damage, {"fused_roi": fused}
        return loc, damage


def main():
    wrapper = FrozenS8Backbone(
        FakeS8(),
        feature_key="fused_roi",
        expected_channels=16,
    )
    wrapper.train()
    wrapper.assert_frozen()
    output = wrapper(
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 3, 16, 16),
    )
    assert not wrapper.model.training
    assert not output.feature.requires_grad
    assert not any(parameter.requires_grad for parameter in wrapper.parameters())
    print("S8 frozen wrapper: PASS")


if __name__ == "__main__":
    main()
