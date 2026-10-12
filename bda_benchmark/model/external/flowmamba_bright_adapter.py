"""Strict BRIGHT interface around an unmodified official FlowMamba core."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .contracts import normalize_external_output
from .input_contracts import display_sar_to_imagenet


class FlowMambaBRIGHTAdapter(nn.Module):
    """Adapt I/O only; the supplied official core must already have a 4C head."""

    variant = "r1"

    def __init__(self, official_core: nn.Module) -> None:
        super().__init__()
        self.official_core = official_core

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ) -> Any:
        sar_view = display_sar_to_imagenet(post_sar)
        value = self.official_core(pre_rgb, sar_view)
        output = normalize_external_output(
            value,
            pre_rgb.shape[-2:],
            auxiliary={"source": "official_flowmamba", "sar_adapter": "display_3ch"},
        )
        return output if return_dict else (output["output_loc"], output["output_dam"])
