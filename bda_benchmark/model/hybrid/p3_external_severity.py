"""R3: frozen P3 localization plus an external I/D/X severity model.

The external model remains responsible only for conditional severity.  P3's
RGB-anchored localization probability is combined with the conditional
severity distribution by the probability product rule.  No pixel router or
residual overwrite is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.model.external.contracts import ExternalModelContractError


def _extract_pair(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, Mapping):
        try:
            return value["output_loc"], value["output_dam"]
        except KeyError as error:
            raise ExternalModelContractError(
                "P3 mapping must expose output_loc and output_dam"
            ) from error
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    raise ExternalModelContractError("unsupported P3 output contract")


def _extract_severity(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        if "output_severity" in value:
            value = value["output_severity"]
        elif "output_dam" in value:
            value = value["output_dam"]
        elif "damage_logits" in value:
            value = value["damage_logits"]
    if not torch.is_tensor(value) or value.ndim != 4:
        raise ExternalModelContractError("severity model must return a BCHW tensor")
    if value.shape[1] == 4:
        value = value[:, 1:4]
    if value.shape[1] != 3:
        raise ExternalModelContractError(
            "severity logits must have 3 conditional classes or 4 BRIGHT classes"
        )
    return value


class P3ExternalSeverityHybrid(nn.Module):
    """Combine frozen P3 localization with trainable external severity logits."""

    variant = "r3"

    def __init__(self, p3: nn.Module, severity_model: nn.Module) -> None:
        super().__init__()
        self.p3_locator = p3
        self.severity_model = severity_model
        self.freeze_locator()

    def freeze_locator(self) -> None:
        self.p3_locator.eval()
        for parameter in self.p3_locator.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.p3_locator.eval()
        return self

    @staticmethod
    def p3_sar_view(post_sar: torch.Tensor) -> torch.Tensor:
        """Recover the legacy three-channel display-SAR contract used by P3."""
        if post_sar.ndim != 4 or post_sar.shape[1] not in (1, 2):
            raise ExternalModelContractError(
                "R3 expects native SAR or [native,display] SAR channels"
            )
        display = post_sar[:, -1:].float()
        # Display SAR is stored in the 0..255 photometric domain.
        mean = display.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        std = display.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        return ((display.repeat(1, 3, 1, 1) - mean) / std).to(post_sar.dtype)

    @staticmethod
    def compose(
        localization_logits: torch.Tensor,
        severity_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized BG/I/D/X log-probabilities."""
        if localization_logits.ndim != 4 or localization_logits.shape[1] != 2:
            raise ExternalModelContractError("localization logits must be [B,2,H,W]")
        if severity_logits.ndim != 4 or severity_logits.shape[1] != 3:
            raise ExternalModelContractError("severity logits must be [B,3,H,W]")
        size = localization_logits.shape[-2:]
        if severity_logits.shape[-2:] != size:
            severity_logits = F.interpolate(
                severity_logits, size, mode="bilinear", align_corners=False
            )
        log_location = F.log_softmax(localization_logits.float(), dim=1)
        log_severity = F.log_softmax(severity_logits.float(), dim=1)
        output = torch.cat(
            [log_location[:, :1], log_location[:, 1:2] + log_severity], dim=1
        )
        if not torch.isfinite(output).all():
            raise ExternalModelContractError("R3 composition produced non-finite logits")
        return output.to(localization_logits.dtype)

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        with torch.no_grad():
            p3_value = self.p3_locator(pre_rgb, self.p3_sar_view(post_sar))
            localization_logits, _ = _extract_pair(p3_value)
        severity_value = self.severity_model(pre_rgb, post_sar)
        severity_logits = _extract_severity(severity_value)
        output_damage = self.compose(localization_logits, severity_logits)
        outputs = {
            "output_loc": localization_logits,
            "output_dam": output_damage,
            "output_severity": severity_logits,
            "aux": {"p3_locator_frozen": True},
        }
        return outputs if return_dict else (localization_logits, output_damage)
