# -*- coding: utf-8 -*-
"""S8-P4 NS-MICB: controlled native-SAR input ablations on the accepted P3.

Only the SAR representation and SAR input stem differ between A0...A4.
The P3 RGB branch, multi-scale fusion, ROI prior, pixel heads and MICB/DSMIL
head are inherited without adding R1, R4 or any reliability gate.

``a2`` is retained as the historical train-stat-normalised one-channel
ablation.  Its channel-sum migration is exact only for a repeated *identical*
input tensor and is therefore not a function-preserving conversion of P3's
ImageNet-normalised pseudo-RGB contract.  ``a4`` is the corrected control: it
uses an unclipped affine one-channel view and folds both the three input
weights and their mean/std-dependent constant term into Conv2d weight+bias.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Tuple

import torch
import torch.nn as nn

from configs.experiment_registry import P4_NATIVE_SAR_VARIANTS
from model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


@dataclass(frozen=True)
class P4VariantSpec:
    name: str
    sar_channels: int
    sar_normalization: str
    sar_input_head: str


P4_VARIANTS: Dict[str, P4VariantSpec] = {
    "a0": P4VariantSpec(
        name="S8-P4-A0-PseudoRGB-Control",
        sar_channels=3,
        sar_normalization="imagenet_after_repeat3",
        sar_input_head="p3_patch_embed",
    ),
    "a1": P4VariantSpec(
        name="S8-P4-A1-TrainStats-Repeat3",
        sar_channels=3,
        sar_normalization="train_stats_after_clip_repeat3",
        sar_input_head="p3_patch_embed",
    ),
    "a2": P4VariantSpec(
        name="S8-P4-A2-Native1ch-PatchEmbed",
        sar_channels=1,
        sar_normalization="train_stats_after_clip",
        sar_input_head="native_1x96_k4s4",
    ),
    "a3": P4VariantSpec(
        name="S8-P4-A3-Native1ch-ConvStem",
        sar_channels=1,
        sar_normalization="train_stats_after_clip",
        sar_input_head="native_1x32x64x96",
    ),
    "a4": P4VariantSpec(
        name="S8-P4-A4-FunctionPreserving-Native1ch",
        sar_channels=1,
        sar_normalization="train_stats_affine_no_clip",
        sar_input_head="native_1x96_k4s4_affine_fold",
    ),
}

if tuple(P4_VARIANTS) != P4_NATIVE_SAR_VARIANTS:
    raise RuntimeError(
        "P4 implementation registry diverged from configs.experiment_registry"
    )


def get_p4_variant_spec(variant: str) -> P4VariantSpec:
    key = str(variant).lower()
    if key not in P4_VARIANTS:
        raise ValueError(f"unknown P4 variant {variant!r}; choose {sorted(P4_VARIANTS)}")
    return P4_VARIANTS[key]


class NativeSARConvStem(nn.Sequential):
    """A3 SAR stem: 1->32->64->96 at total stride four, returning NHWC."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 32)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 64)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(64, 96, kernel_size=1, bias=True)
        self.final_norm = nn.LayerNorm(96)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.conv3(x).permute(0, 2, 3, 1).contiguous()
        return self.final_norm(x)


class ToNHWC(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()


def _native_patch_embed() -> nn.Sequential:
    # The module indices intentionally match torchvision Swin's patch embed:
    # Conv2d -> Permute(NCHW,NHWC) -> LayerNorm. nn.Identity cannot permute,
    # therefore the stateless permutation module occupies index 1.
    return nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=4, stride=4, bias=True),
        ToNHWC(),
        nn.LayerNorm(96),
    )


class DamageFormer_S8_P4_NativeSAR_MICB(DamageFormer_S8_MICB_DSMIL):
    """Accepted P3 with one controlled SAR-input variant."""

    def __init__(self, variant: str = "a0", **kwargs) -> None:
        self.variant = str(variant).lower()
        self.variant_spec = get_p4_variant_spec(self.variant)
        super().__init__(**kwargs)
        if self.variant in ("a2", "a4"):
            self.encoder_2.swin_transformer.features[0] = _native_patch_embed()
        elif self.variant == "a3":
            self.encoder_2.swin_transformer.features[0] = NativeSARConvStem()

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        *args,
        **kwargs,
    ):
        expected = self.variant_spec.sar_channels
        if post_data.ndim != 4 or int(post_data.shape[1]) != expected:
            raise ValueError(
                f"{self.variant_spec.name} expects SAR [B,{expected},H,W], "
                f"got {tuple(post_data.shape)}"
            )
        return super().forward(pre_data, post_data, *args, **kwargs)

    def experiment_metadata(self) -> Dict[str, object]:
        return {
            "family": "S8-P4 NS-MICB",
            "variant": self.variant,
            "variant_spec": asdict(self.variant_spec),
            "r1_enabled": False,
            "r4_enabled": False,
            "reliability_gate_enabled": False,
            "mil_residual_scale": self.mil_residual_scale,
        }


def _extract_state_dict(checkpoint: object) -> MutableMapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state: object = checkpoint
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if not isinstance(state, Mapping):
        raise TypeError("cannot find a model state dict in checkpoint")
    cleaned: MutableMapping[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[7:]
        cleaned[name] = value
    if not cleaned:
        raise RuntimeError("checkpoint model state is empty")
    return cleaned


def collapse_pseudo_rgb_patch_weight(weight: torch.Tensor) -> torch.Tensor:
    """Legacy A2 fold for three *identical, already normalised* channels.

    This helper deliberately remains available so historical A2 checkpoints
    keep their declared semantics.  It must not be used when the old and new
    radiometric normalisations differ; use
    :func:`fold_p3_imagenet_patch_embed_to_native` for that conversion.
    """

    if weight.ndim != 4 or int(weight.shape[1]) != 3:
        raise ValueError(f"expected [out,3,k,k] patch weight, got {tuple(weight.shape)}")
    return weight.sum(dim=1, keepdim=True)


def fold_p3_imagenet_patch_embed_to_native(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    sar_mean: float,
    sar_std: float,
    imagenet_mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
    imagenet_std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold P3's repeated pseudo-RGB patch embed into one affine SAR channel.

    P3 receives ``(x - mean[c]) / std[c]`` in each of three channels.  A4
    receives ``u = (x - sar_mean) / sar_std`` without clipping.  For a
    no-padding patch convolution the returned parameters satisfy, up to the
    floating-point rounding of the parameter dtype::

        conv3(repeat/imagenet_norm(x), W, b)
        == conv1(native_affine(x), W_native, b_native)

    Clipping or a nonlinear transform would destroy this equivalence and is
    intentionally forbidden by the A4 data contract.
    """

    if weight.ndim != 4 or int(weight.shape[1]) != 3:
        raise ValueError(f"expected [out,3,k,k] patch weight, got {tuple(weight.shape)}")
    if not float(sar_std) > 0.0:
        raise ValueError("sar_std must be positive")
    if len(imagenet_mean) != 3 or len(imagenet_std) != 3:
        raise ValueError("ImageNet mean/std must each contain three values")
    if any(float(value) <= 0.0 for value in imagenet_std):
        raise ValueError("ImageNet standard deviations must be positive")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError(
            f"expected bias [{weight.shape[0]}], got {tuple(bias.shape)}"
        )

    # Perform the algebra in float64, then return the checkpoint dtype.  This
    # minimises migration rounding without silently changing model precision.
    work = weight.detach().to(dtype=torch.float64)
    means = torch.as_tensor(imagenet_mean, dtype=torch.float64, device=work.device)
    stds = torch.as_tensor(imagenet_std, dtype=torch.float64, device=work.device)
    native_weight = (
        work * (float(sar_std) / stds).view(1, 3, 1, 1)
    ).sum(dim=1, keepdim=True)
    constant = (
        work
        * ((float(sar_mean) - means) / stds).view(1, 3, 1, 1)
    ).sum(dim=(1, 2, 3))
    if bias is None:
        native_bias = constant
    else:
        native_bias = bias.detach().to(dtype=torch.float64) + constant
    return (
        native_weight.to(dtype=weight.dtype),
        native_bias.to(dtype=weight.dtype),
    )


def load_p3_checkpoint(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
    checkpoint_path: str | Path,
    *,
    sar_stats: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Load the accepted P3 with explicit, auditable A2/A3/A4 migrations."""

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = _extract_state_dict(checkpoint)
    variant = model.variant
    patch_prefix = "encoder_2.swin_transformer.features.0."
    migrated: Dict[str, str] = {}

    if variant in ("a0", "a1"):
        incompatible = model.load_state_dict(state, strict=True)
    elif variant == "a2":
        weight_key = patch_prefix + "0.weight"
        if weight_key not in state:
            raise KeyError(f"P3 checkpoint is missing {weight_key}")
        state[weight_key] = collapse_pseudo_rgb_patch_weight(state[weight_key])
        migrated[weight_key] = "sum RGB input-channel weights: [96,3,4,4] -> [96,1,4,4]"
        incompatible = model.load_state_dict(state, strict=True)
    elif variant == "a4":
        weight_key = patch_prefix + "0.weight"
        bias_key = patch_prefix + "0.bias"
        if weight_key not in state or bias_key not in state:
            raise KeyError(f"P3 checkpoint is missing {weight_key} or {bias_key}")
        if sar_stats is None:
            raise ValueError("A4 migration requires train-only SAR mean/std")
        missing_stats = [key for key in ("mean", "std") if key not in sar_stats]
        if missing_stats:
            raise ValueError(f"A4 SAR statistics are missing {missing_stats}")
        native_weight, native_bias = fold_p3_imagenet_patch_embed_to_native(
            state[weight_key],
            state[bias_key],
            sar_mean=float(sar_stats["mean"]),
            sar_std=float(sar_stats["std"]),
        )
        state[weight_key] = native_weight
        state[bias_key] = native_bias
        migrated[weight_key] = (
            "affine fold of ImageNet-normalised repeated SAR into one "
            "unclipped train-stat-normalised channel"
        )
        migrated[bias_key] = "mean/std constant folded into patch-embed bias"
        incompatible = model.load_state_dict(state, strict=True)
    else:
        # A3 deliberately replaces the complete P3 SAR patch embedding.
        # The old LayerNorm is shape-compatible and is transferred to the
        # final stem norm; all convolutional stem parameters remain new.
        norm_map = {
            patch_prefix + "2.weight": patch_prefix + "final_norm.weight",
            patch_prefix + "2.bias": patch_prefix + "final_norm.bias",
        }
        filtered = {
            key: value for key, value in state.items() if not key.startswith(patch_prefix)
        }
        for old_key, new_key in norm_map.items():
            if old_key in state:
                filtered[new_key] = state[old_key]
                migrated[new_key] = f"copied from {old_key}"
        incompatible = model.load_state_dict(filtered, strict=False)
        illegal_missing = [
            key for key in incompatible.missing_keys if not key.startswith(patch_prefix)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "A3 P3 migration changed parameters outside the declared SAR stem: "
                f"missing={illegal_missing}, unexpected={incompatible.unexpected_keys}"
            )

    return {
        "checkpoint": str(checkpoint_path),
        "variant": variant,
        "strict_common_parameters": True,
        "migrations": migrated,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


SAR_INPUT_PREFIX = "encoder_2.swin_transformer.features.0."
SAR_FROZEN_BODY_PREFIXES: Tuple[str, ...] = (
    "encoder_2.swin_transformer.features.1.",
    "encoder_2.swin_transformer.features.2.",
    "encoder_2.swin_transformer.features.3.",
)
P4_LATE_TRAINABLE_PREFIXES: Tuple[str, ...] = (
    "fusion_layer_3.",
    "clf_2.",
)


def configure_p4_trainable(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
    input_only: bool,
) -> Dict[str, int]:
    """Apply the locked P4 freeze policy and return parameter counts."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = (SAR_INPUT_PREFIX,) if input_only else (
        SAR_INPUT_PREFIX,
        *P4_LATE_TRAINABLE_PREFIXES,
    )
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)

    # Frozen BatchNorm statistics must never drift after model.train().
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def lock_frozen_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


__all__ = [
    "DamageFormer_S8_P4_NativeSAR_MICB",
    "NativeSARConvStem",
    "P4_VARIANTS",
    "P4VariantSpec",
    "SAR_FROZEN_BODY_PREFIXES",
    "SAR_INPUT_PREFIX",
    "collapse_pseudo_rgb_patch_weight",
    "fold_p3_imagenet_patch_embed_to_native",
    "configure_p4_trainable",
    "get_p4_variant_spec",
    "load_p3_checkpoint",
    "lock_frozen_batch_norm",
]
