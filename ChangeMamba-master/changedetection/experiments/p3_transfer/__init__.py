"""Controlled transfer of P3 task structure into ChangeMamba."""

from .model import (
    ChangeMambaP3Transfer,
    VALID_P3_TRANSFER_VARIANTS,
    factorized_damage_log_probs,
    freeze_for_p3_transfer,
    load_parent_checkpoint,
)

__all__ = [
    "ChangeMambaP3Transfer",
    "VALID_P3_TRANSFER_VARIANTS",
    "factorized_damage_log_probs",
    "freeze_for_p3_transfer",
    "load_parent_checkpoint",
]
