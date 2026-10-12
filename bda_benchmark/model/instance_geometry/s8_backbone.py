"""Strict S8 checkpoint loading and permanently frozen feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import Tensor, nn

from ..DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from ..DamageFormer_S9Ic_ReliableGeometry import select_geometry_feature


@dataclass
class FrozenS8Output:
    localization_logits: Tensor
    damage_logits: Tensor
    feature_key: str
    feature: Tensor
    auxiliary: Mapping[str, Any]


def _extract_state_dict(payload: Any) -> MutableMapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"S8 checkpoint must be a mapping, got {type(payload)!r}")
    for key in ("model_state_dict", "state_dict", "model", "deep_model", "net"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    if payload and all(isinstance(key, str) for key in payload):
        if any(torch.is_tensor(value) for value in payload.values()):
            return dict(payload)
    raise KeyError("S8 checkpoint contains no recognized model state dict")


def _clean_state_dict(state: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    cleaned: Dict[str, Tensor] = {}
    for key, value in state.items():
        name = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "deep_model."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        if name in cleaned:
            raise KeyError(f"duplicate S8 key after prefix cleanup: {name}")
        cleaned[name] = value
    return cleaned


def _load_checkpoint_payload(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
    allow_unsafe: bool,
) -> Any:
    if allow_unsafe:
        try:
            return torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Safe S8 loading requires a PyTorch version with weights_only. "
            "Use allow_unsafe=True only for a trusted legacy checkpoint."
        ) from exc


class FrozenS8Backbone(nn.Module):
    """Own an S8/P3 model that cannot accidentally return to training mode."""

    def __init__(
        self,
        model: DamageFormer_S8_MICB_DSMIL,
        *,
        feature_key: str = "fused_roi",
        expected_channels: int = 256,
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_key = feature_key
        self.expected_channels = int(expected_channels)
        self.freeze()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        feature_key: str = "fused_roi",
        expected_channels: int = 256,
        map_location: str | torch.device = "cpu",
        allow_unsafe: bool = False,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "FrozenS8Backbone":
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"S8 checkpoint not found: {path}")
        model = DamageFormer_S8_MICB_DSMIL(**dict(model_kwargs or {}))
        payload = _load_checkpoint_payload(
            path,
            map_location=map_location,
            allow_unsafe=allow_unsafe,
        )
        state = _clean_state_dict(_extract_state_dict(payload))
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Strict S8 load unexpectedly returned incompatible keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return cls(
            model,
            feature_key=feature_key,
            expected_channels=expected_channels,
        )

    def freeze(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenS8Backbone":
        super().train(False)
        self.model.eval()
        return self

    def assert_frozen(self) -> None:
        trainable = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"S8 contains trainable parameters: {trainable[:5]}")
        if self.model.training:
            raise RuntimeError("S8 backbone must remain in eval mode")

    def forward(self, pre_image: Tensor, post_image: Tensor) -> FrozenS8Output:
        self.assert_frozen()
        with torch.no_grad():
            output = self.model(
                pre_image,
                post_image,
                return_aux=True,
            )
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise RuntimeError(
                "S8 return_aux=True must return (loc, damage, aux)"
            )
        localization_logits, damage_logits, auxiliary = output
        if not isinstance(auxiliary, Mapping):
            raise TypeError("S8 auxiliary output must be a mapping")
        key, feature = select_geometry_feature(
            auxiliary,
            requested_key=self.feature_key,
        )
        if feature.shape[1] != self.expected_channels:
            raise RuntimeError(
                f"S8 feature {key!r} has {feature.shape[1]} channels; "
                f"expected {self.expected_channels}"
            )
        if feature.requires_grad:
            raise RuntimeError("Frozen S8 feature unexpectedly requires gradients")
        return FrozenS8Output(
            localization_logits=localization_logits,
            damage_logits=damage_logits,
            feature_key=key,
            feature=feature,
            auxiliary=auxiliary,
        )

