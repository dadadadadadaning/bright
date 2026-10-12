"""Reusable building blocks for BRIGHT teacher/student distillation."""

from .adapter import (
    DistillationOutput,
    collect_feature_maps,
    forward_for_distillation,
    normalize_model_output,
)
from .checkpoint import CheckpointLoadReport, extract_state_dict, load_checkpoint
from .pair import DistillationPair

__all__ = [
    "CheckpointLoadReport",
    "DistillationOutput",
    "DistillationPair",
    "collect_feature_maps",
    "extract_state_dict",
    "forward_for_distillation",
    "load_checkpoint",
    "normalize_model_output",
]

