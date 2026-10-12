import torch
import torch.nn as nn


class P3FourClassControl(nn.Module):
    def forward(self, p3_logits: torch.Tensor, **_) -> torch.Tensor:
        return p3_logits
