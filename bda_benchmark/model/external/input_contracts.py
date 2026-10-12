"""Photometric adapters kept outside all third-party repositories."""
from __future__ import annotations

import torch

from .contracts import ExternalModelContractError


def display_sar_to_imagenet(post_sar: torch.Tensor) -> torch.Tensor:
    """Convert BRIGHT display SAR to the three-channel ImageNet contract.

    The dataset may expose either display SAR alone or ``[native, display]``.
    No percentile normalization is performed here: this path deliberately
    preserves the display-domain contract used by RGB-pretrained encoders.
    """
    if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
        raise ExternalModelContractError(
            "display SAR adapter expects [B,1,H,W] or [B,2,H,W]"
        )
    display = post_sar[:, -1:].float()
    # Accept both explicit 0..255 display values and already scaled 0..1 data.
    if float(display.detach().amax()) <= 1.5:
        display = display * 255.0
    mean = display.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = display.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    return ((display.repeat(1, 3, 1, 1) - mean) / std).to(post_sar.dtype)
