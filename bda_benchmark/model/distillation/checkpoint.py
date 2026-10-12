"""Checkpoint loading shared by teacher and student models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch import nn


_STATE_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "net",
    "deep_model",
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: Path
    state_key: str
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]

    @property
    def exact_match(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def extract_state_dict(checkpoint: Any) -> Tuple[MutableMapping[str, Any], str]:
    """Extract a state dict from the checkpoint formats used in this project."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must be a mapping, got {type(checkpoint).__name__}"
        )

    for key in _STATE_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return dict(value), key

    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        tensor_values = sum(torch.is_tensor(value) for value in checkpoint.values())
        if tensor_values:
            return dict(checkpoint), "<root>"

    raise KeyError(
        "No model state dict found. Expected one of: "
        + ", ".join(_STATE_KEYS)
    )


def _strip_prefixes(
    state_dict: Mapping[str, Any], prefixes: Sequence[str]
) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        if new_key in cleaned:
            raise KeyError(
                f"Checkpoint prefix cleanup produced duplicate key {new_key!r}"
            )
        cleaned[new_key] = value
    return cleaned


def _safe_torch_load(path: Path, map_location: str | torch.device, allow_unsafe: bool):
    if allow_unsafe:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support safe weights-only loading. "
            "Upgrade PyTorch or pass allow_unsafe=True for a trusted checkpoint."
        ) from exc


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    strip_prefixes: Sequence[str] = ("module.", "deep_model."),
    allow_unsafe: bool = False,
) -> CheckpointLoadReport:
    """Load one checkpoint and return an explicit compatibility report.

    The safe weights-only loader is used by default. Set ``allow_unsafe=True``
    only for a trusted checkpoint that contains pickled non-tensor objects.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = _safe_torch_load(path, map_location, allow_unsafe)
    state_dict, state_key = extract_state_dict(checkpoint)
    state_dict = _strip_prefixes(state_dict, strip_prefixes)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return CheckpointLoadReport(
        path=path,
        state_key=state_key,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
