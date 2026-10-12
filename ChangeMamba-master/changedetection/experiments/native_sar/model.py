"""One-channel SAR input adapter isolated from the ChangeMamba baseline.

The accepted ChangeMamba-BRIGHT baseline repeats a raw SAR image three times
and applies ImageNet channel-wise normalisation.  This experiment receives a
single train-stat-normalised SAR channel and reconstructs that exact tensor
with a 3x3 convolution whose non-central weights are initially zero.  The
larger kernel permits a controlled local-speckle adaptation while preserving
the baseline function at step zero.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Tuple

import torch
import torch.nn as nn

from changedetection.checkpoints import extract_model_state_dict
from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA

IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


def native_adapter_parameters(
    sar_mean: float,
    sar_std: float,
    *,
    imagenet_mean: Tuple[float, float, float] = IMAGENET_MEAN,
    imagenet_std: Tuple[float, float, float] = IMAGENET_STD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a function-preserving 1ch-to-3ch 3x3 affine adapter."""
    if not float(sar_std) > 0.0:
        raise ValueError("sar_std must be positive")
    weight = torch.zeros(3, 1, 3, 3, dtype=torch.float32)
    bias = torch.empty(3, dtype=torch.float32)
    for channel, (rgb_mean, rgb_std) in enumerate(
        zip(imagenet_mean, imagenet_std)
    ):
        if not float(rgb_std) > 0.0:
            raise ValueError("ImageNet standard deviations must be positive")
        weight[channel, 0, 1, 1] = float(sar_std) / float(rgb_std)
        bias[channel] = (float(sar_mean) - float(rgb_mean)) / float(rgb_std)
    return weight, bias


class ChangeMambaMMBDANativeSAR(ChangeMambaMMBDA):
    """ChangeMamba-BRIGHT with an external one-channel SAR contract."""

    def __init__(
        self,
        output_building: int,
        output_damage: int,
        pretrained,
        *,
        sar_mean: float,
        sar_std: float,
        **kwargs,
    ) -> None:
        # Both VMamba encoders deliberately retain their original 3-channel
        # structure.  Only the isolated pre-encoder adapter is new.
        super().__init__(
            output_building=output_building,
            output_damage=output_damage,
            pretrained=pretrained,
            **kwargs,
        )
        self.sar_adapter = nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.reset_native_adapter(sar_mean=sar_mean, sar_std=sar_std)
        self.sar_mean = float(sar_mean)
        self.sar_std = float(sar_std)

    @torch.no_grad()
    def reset_native_adapter(self, *, sar_mean: float, sar_std: float) -> None:
        weight, bias = native_adapter_parameters(sar_mean, sar_std)
        self.sar_adapter.weight.copy_(weight.to(self.sar_adapter.weight))
        self.sar_adapter.bias.copy_(bias.to(self.sar_adapter.bias))

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor):
        if post_data.ndim != 4 or int(post_data.shape[1]) != 1:
            raise ValueError(
                "native SAR input must have shape [B,1,H,W], got "
                f"{tuple(post_data.shape)}"
            )
        return super().forward(pre_data, self.sar_adapter(post_data))

    def experiment_metadata(self) -> Dict[str, object]:
        return {
            "family": "ChangeMamba-BRIGHT NativeSAR input audit",
            "input_contract": "RGB3 + SAR1",
            "sar_adapter": "Conv2d(1,3,k3,p1), exact affine centre init",
            "sar_normalization": "train_mean_std_no_clip",
            "baseline_body_changed": False,
        }


def load_baseline_as_native(
    model: ChangeMambaMMBDANativeSAR,
    checkpoint_path: str,
) -> Dict[str, object]:
    """Strictly migrate a baseline checkpoint; only the adapter may be absent."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = extract_model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"sar_adapter.weight", "sar_adapter.bias"}
    missing = set(incompatible.missing_keys)
    if missing != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "baseline migration changed undeclared namespaces: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model.reset_native_adapter(sar_mean=model.sar_mean, sar_std=model.sar_std)
    return {
        "checkpoint": checkpoint_path,
        "loaded_keys": len(state),
        "missing_keys": sorted(missing),
        "unexpected_keys": [],
        "function_preserving_adapter": True,
    }


def freeze_except_native_adapter(model: nn.Module) -> Dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = getattr(model, "sar_adapter")
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }

