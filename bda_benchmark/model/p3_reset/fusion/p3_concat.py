import torch
import torch.nn as nn


class P3ConcatControl(nn.Module):
    """Exact P3 concatenation/fusion control (C0)."""

    def __init__(self, p3_fusion: nn.Module) -> None:
        super().__init__()
        self.fusion = p3_fusion

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-2:] != sar.shape[-2:]:
            raise ValueError("C0 requires spatially aligned P3 feature grids")
        return self.fusion(torch.cat((rgb, sar), dim=1))
