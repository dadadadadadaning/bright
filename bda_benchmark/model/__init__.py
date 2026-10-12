"""Unified model access for BRIGHT.

Model implementations remain lazily imported so listing the registry does not
load PyTorch, torchvision, timm, or pretrained weights.
"""

from .registry import (
    MODEL_SPECS,
    ModelSpec,
    create_model,
    get_model_spec,
    list_model_specs,
    resolve_model_name,
)

__all__ = [
    "MODEL_SPECS",
    "ModelSpec",
    "create_model",
    "get_model_spec",
    "list_model_specs",
    "resolve_model_name",
]

