"""Strict BRIGHT interface around an unmodified official M2CD core."""
from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from .contracts import ExternalModelContractError, normalize_external_output
from .input_contracts import display_sar_to_imagenet


class M2CDBRIGHTAdapter(nn.Module):
    """Call an official OpenCD model with an explicitly declared input style.

    ``separate`` is for cores accepting ``core(optical, sar)``. ``stacked`` is
    for OpenCD segmentors accepting a six-channel pair.  The mode is mandatory
    so a repository API change can never be silently guessed.
    """

    variant = "r2"

    def __init__(
        self,
        official_core: nn.Module,
        *,
        call_style: Literal["separate", "stacked"],
    ) -> None:
        super().__init__()
        if call_style not in {"separate", "stacked"}:
            raise ExternalModelContractError("M2CD call_style must be explicit")
        self.official_core = official_core
        self.call_style = call_style

    def _call(self, rgb: torch.Tensor, sar: torch.Tensor) -> Any:
        if self.call_style == "separate":
            return self.official_core(rgb, sar)
        pair = torch.cat([rgb, sar], dim=1)
        try:
            return self.official_core(pair, data_samples=None, mode="tensor")
        except TypeError as error:
            raise ExternalModelContractError(
                "official M2CD core rejected the declared stacked OpenCD contract"
            ) from error

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ) -> Any:
        sar_view = display_sar_to_imagenet(post_sar)
        value = self._call(pre_rgb, sar_view)
        output = normalize_external_output(
            value,
            pre_rgb.shape[-2:],
            auxiliary={
                "source": "official_m2cd",
                "call_style": self.call_style,
                "sar_adapter": "display_3ch",
            },
        )
        return output if return_dict else (output["output_loc"], output["output_dam"])
