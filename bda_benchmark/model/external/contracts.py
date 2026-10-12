"""Shared BRIGHT output contract for external multimodal models."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


class ExternalModelContractError(RuntimeError):
    """Raised when an official model cannot satisfy the BRIGHT interface."""


def damage_to_localization_logits(damage_logits: torch.Tensor) -> torch.Tensor:
    """Convert BG/I/D/X logits to an exactly compatible BG/building pair."""
    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ExternalModelContractError(
            "damage logits must have shape [B,4,H,W], got "
            f"{tuple(damage_logits.shape)}"
        )
    building = torch.logsumexp(damage_logits[:, 1:4].float(), 1, keepdim=True)
    return torch.cat([damage_logits[:, :1].float(), building], 1).to(
        damage_logits.dtype
    )


def _find_four_class_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.ndim == 4 and value.shape[1] == 4:
            return value
        raise ExternalModelContractError(
            "external tensor output is not [B,4,H,W]: " f"{tuple(value.shape)}"
        )
    if isinstance(value, Mapping):
        for key in ("output_dam", "damage_logits", "logits", "seg_logits"):
            if key in value:
                return _find_four_class_tensor(value[key])
        raise ExternalModelContractError(
            "external mapping has no declared four-class logit key"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = [item for item in value if torch.is_tensor(item)]
        valid = [item for item in candidates if item.ndim == 4 and item.shape[1] == 4]
        if len(valid) == 1:
            return valid[0]
        raise ExternalModelContractError(
            f"external sequence contains {len(valid)} four-class tensors; expected one"
        )
    raise ExternalModelContractError(
        f"unsupported external output type: {type(value).__name__}"
    )


def normalize_external_output(
    value: Any,
    output_size: tuple[int, int],
    *,
    auxiliary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize an official model output without guessing binary task heads."""
    logits = _find_four_class_tensor(value)
    if logits.shape[-2:] != output_size:
        logits = F.interpolate(logits, output_size, mode="bilinear", align_corners=False)
    if not torch.isfinite(logits).all():
        raise ExternalModelContractError("external model produced non-finite logits")
    return {
        "output_dam": logits,
        "output_loc": damage_to_localization_logits(logits),
        "aux": dict(auxiliary or {}),
    }
