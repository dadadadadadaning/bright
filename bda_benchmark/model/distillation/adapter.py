"""Normalize heterogeneous BRIGHT model inputs and outputs for distillation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from ..registry import ModelSpec, get_model_spec


_LOCALIZATION_KEYS = (
    "output_loc",
    "loc_logits",
    "localization_logits",
    "building_logits",
)
_DAMAGE_KEYS = (
    "output_dam",
    "dam_logits",
    "damage_logits",
    "main_dam_logits",
    "output_damage",
)
_FEATURE_MARKERS = ("feat", "feature", "embedding", "projection", "align")


@dataclass
class DistillationOutput:
    """Canonical output contract used by distillation losses."""

    damage_logits: Tensor
    localization_logits: Optional[Tensor]
    feature_maps: Dict[str, Tensor]
    auxiliary: Dict[str, Any]
    raw: Any

    def feature(self, *preferred_keys: str) -> Tensor:
        """Select a feature by preference, or the first available feature."""

        for key in preferred_keys:
            if key in self.feature_maps:
                return self.feature_maps[key]
        if self.feature_maps:
            return next(iter(self.feature_maps.values()))
        raise KeyError(
            "This model output contains no feature map. Enable return_aux on the "
            "model or distill logits only."
        )


def _first_tensor(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[Tensor]:
    for key in keys:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return value
    return None


def collect_feature_maps(mapping: Mapping[str, Any]) -> Dict[str, Tensor]:
    """Collect tensor features from a model output/auxiliary mapping."""

    features: Dict[str, Tensor] = {}
    for key, value in mapping.items():
        key_lower = str(key).lower()
        if torch.is_tensor(value) and any(
            marker in key_lower for marker in _FEATURE_MARKERS
        ):
            features[str(key)] = value
        elif isinstance(value, Mapping):
            for nested_key, nested_value in collect_feature_maps(value).items():
                features[f"{key}.{nested_key}"] = nested_value
    return features


def normalize_model_output(output: Any) -> DistillationOutput:
    """Convert Tensor, tuple, or mapping outputs to one stable contract."""

    localization_logits: Optional[Tensor] = None
    damage_logits: Optional[Tensor] = None
    auxiliary: Dict[str, Any] = {}

    if torch.is_tensor(output):
        damage_logits = output
    elif isinstance(output, Mapping):
        auxiliary = dict(output)
        localization_logits = _first_tensor(output, _LOCALIZATION_KEYS)
        damage_logits = _first_tensor(output, _DAMAGE_KEYS)
    elif isinstance(output, (tuple, list)):
        values = list(output)
        if len(values) >= 2 and torch.is_tensor(values[0]) and torch.is_tensor(values[1]):
            localization_logits = values[0]
            damage_logits = values[1]
            remaining = values[2:]
        elif values and torch.is_tensor(values[0]):
            damage_logits = values[0]
            remaining = values[1:]
        else:
            remaining = values

        for index, value in enumerate(remaining):
            if isinstance(value, Mapping):
                auxiliary.update(value)
            else:
                auxiliary[f"extra_{index}"] = value
    else:
        raise TypeError(
            "Unsupported model output type for distillation: "
            f"{type(output).__name__}"
        )

    if damage_logits is None:
        available = ", ".join(sorted(auxiliary)) if auxiliary else "<none>"
        raise KeyError(
            "Could not locate damage logits in model output. "
            f"Available output keys: {available}"
        )

    feature_source = dict(auxiliary)
    if isinstance(output, Mapping):
        feature_source.update(output)
    features = collect_feature_maps(feature_source)
    return DistillationOutput(
        damage_logits=damage_logits,
        localization_logits=localization_logits,
        feature_maps=features,
        auxiliary=auxiliary,
        raw=output,
    )


def _accepts_return_aux(model: nn.Module) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "return_aux" in parameters


def forward_for_distillation(
    model: nn.Module,
    model_spec: ModelSpec | str,
    pre_image: Tensor,
    post_image: Tensor,
    *,
    request_aux: bool = True,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> DistillationOutput:
    """Run one registered model and normalize its output.

    Concatenated-input baselines receive ``cat([pre, post], dim=1)``. Paired
    models receive the two images separately.
    """

    spec = get_model_spec(model_spec) if isinstance(model_spec, str) else model_spec
    kwargs = dict(forward_kwargs or {})
    if request_aux and _accepts_return_aux(model):
        kwargs.setdefault("return_aux", True)

    if spec.input_mode == "concatenated":
        raw = model(torch.cat([pre_image, post_image], dim=1), **kwargs)
    elif spec.input_mode == "paired":
        raw = model(pre_image, post_image, **kwargs)
    else:
        raise ValueError(
            f"Unsupported input_mode {spec.input_mode!r} for model {spec.name!r}"
        )
    return normalize_model_output(raw)

