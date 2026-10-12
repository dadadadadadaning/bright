"""Adapters for unmodified third-party multimodal models."""

from .contracts import (
    ExternalModelContractError,
    damage_to_localization_logits,
    normalize_external_output,
)
from .flowmamba_bright_adapter import FlowMambaBRIGHTAdapter
from .m2cd_bright_adapter import M2CDBRIGHTAdapter

__all__ = [
    "ExternalModelContractError",
    "damage_to_localization_logits",
    "normalize_external_output",
    "FlowMambaBRIGHTAdapter",
    "M2CDBRIGHTAdapter",
]
