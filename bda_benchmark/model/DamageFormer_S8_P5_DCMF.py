# -*- coding: utf-8 -*-
"""S8-P5 DCMF: evidence-based native-SAR cross-modal damage fusion.

This is a new research branch rather than another P4 input-stem ablation.
The implementation combines four independently testable ideas:

* MSFA/SARDet-style fixed filter augmentation for SAR domain transition;
* AnySat-style modality-specific projectors/encoders;
* CROMA-style local radar-optical cross-attention;
* DeCUR-style common/unique representation decoupling.

The accepted P3 RGB localization branch and MICB-DSMIL instance objective are
retained because they already have positive BRIGHT evidence.  The complete SAR
encoder and cross-modal fusion are new and never share the RGB input head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DamageFormer_S8_MICB_DSMIL import (
    BuildingBagSpec,
    DamageFormer_S8_MICB_DSMIL,
    build_predicted_bag_specs,
)


@dataclass(frozen=True)
class P5VariantSpec:
    name: str
    filter_augmentation: bool
    cross_attention: bool
    common_unique_decoupling: bool


P5_VARIANTS: Dict[str, P5VariantSpec] = {
    "c0": P5VariantSpec(
        name="S8-P5-C0-NativeSAR-Control",
        filter_augmentation=False,
        cross_attention=False,
        common_unique_decoupling=False,
    ),
    "c1": P5VariantSpec(
        name="S8-P5-C1-MSFA-FilterSAR",
        filter_augmentation=True,
        cross_attention=False,
        common_unique_decoupling=False,
    ),
    "c2": P5VariantSpec(
        name="S8-P5-C2-MSFA-CrossAttention",
        filter_augmentation=True,
        cross_attention=True,
        common_unique_decoupling=False,
    ),
    "c2_raw": P5VariantSpec(
        name="S8-P5-C2Raw-NativeSAR-CrossAttention",
        filter_augmentation=False,
        cross_attention=True,
        common_unique_decoupling=False,
    ),
    "c3": P5VariantSpec(
        name="S8-P5-C3-DCMF-Full",
        filter_augmentation=True,
        cross_attention=True,
        common_unique_decoupling=True,
    ),
}


def get_p5_variant_spec(variant: str) -> P5VariantSpec:
    key = str(variant).lower()
    if key not in P5_VARIANTS:
        raise ValueError(f"unknown P5 variant {variant!r}; choose {sorted(P5_VARIANTS)}")
    return P5_VARIANTS[key]


class FrozenSARFilterBank(nn.Module):
    """Deterministic raw/low/high/gradient SAR views.

    SARDet/MSFA demonstrates that fixed image filters can reduce the
    ImageNet-to-SAR domain gap.  This compact implementation avoids external
    wavelet dependencies and keeps the filter path fully auditable.
    """

    output_channels = 4

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer("sobel_y", sobel_y, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != 1:
            raise ValueError(f"native SAR must be [B,1,H,W], got {tuple(x.shape)}")
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        gradient = torch.sqrt(gx.square() + gy.square() + 1e-6)
        return torch.cat((x, low, high, gradient), dim=1)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock2d(nn.Module):
    """Dependency-free ConvNeXt block for the modality-specific SAR encoder."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * expansion, 1)
        self.project = nn.Conv2d(channels * expansion, channels, 1)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        return x + self.gamma[None, :, None, None] * residual


class NativeSARHierarchicalEncoder(nn.Module):
    """A genuinely independent one-channel SAR hierarchy.

    It follows the four-level spatial contract of the RGB Swin branch but has
    no shared input projection or weights.
    """

    dims: Tuple[int, ...] = (96, 192, 384, 768)

    def __init__(
        self,
        filter_augmentation: bool,
        depths: Tuple[int, ...] = (2, 2, 3, 2),
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError("SAR encoder requires four stage depths")
        self.filter_augmentation = bool(filter_augmentation)
        self.filter_bank = FrozenSARFilterBank()
        # C0 and C1 deliberately share the exact same four-channel stem and
        # parameter count.  C0 repeats raw SAR four times; C1 replaces only
        # those four views with the declared fixed filters.
        in_channels = self.filter_bank.output_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.dims[0], kernel_size=4, stride=4),
            LayerNorm2d(self.dims[0]),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (channels, depth) in enumerate(zip(self.dims, depths)):
            self.stages.append(
                nn.Sequential(*[ConvNeXtBlock2d(channels) for _ in range(depth)])
            )
            if index < 3:
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(channels),
                        nn.Conv2d(channels, self.dims[index + 1], 2, stride=2),
                    )
                )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.ndim != 4 or int(x.shape[1]) != 1:
            raise ValueError(f"P5 SAR encoder expects [B,1,H,W], got {tuple(x.shape)}")
        x = self.filter_bank(x) if self.filter_augmentation else x.repeat(1, 4, 1, 1)
        x = self.stem(x)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return tuple(outputs)


class SARMultiScaleProjector(nn.Module):
    """AnySat-style modality-specific projection into a 256-D dense space."""

    def __init__(self, input_dims: Tuple[int, ...] = (96, 192, 384, 768)) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, 64, 1, bias=False),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                )
                for channels in input_dims
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            ConvNeXtBlock2d(256, expansion=2),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("SAR multi-scale projector requires four features")
        size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False
                )
            projected.append(value)
        return self.fuse(torch.cat(projected, dim=1))


def _window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
    batch, channels, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    x = F.pad(x, (0, pad_w, 0, pad_h))
    padded_h, padded_w = x.shape[-2:]
    windows = (
        x.view(
            batch,
            channels,
            padded_h // window_size,
            window_size,
            padded_w // window_size,
            window_size,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window_size * window_size, channels)
    )
    return windows, (batch, height, width, padded_h, padded_w)


def _window_reverse(
    windows: torch.Tensor,
    metadata: Tuple[int, int, int, int, int],
    window_size: int,
) -> torch.Tensor:
    batch, height, width, padded_h, padded_w = metadata
    channels = windows.shape[-1]
    x = (
        windows.view(
            batch,
            padded_h // window_size,
            padded_w // window_size,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, padded_h, padded_w)
    )
    return x[:, :, :height, :width].contiguous()


class BidirectionalWindowCrossAttention(nn.Module):
    """CROMA-style radar-optical cross attention at bounded local cost."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("cross-attention channels must be divisible by heads")
        self.window_size = int(window_size)
        self.rgb_norm = nn.LayerNorm(channels)
        self.sar_norm = nn.LayerNorm(channels)
        self.rgb_from_sar = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.sar_from_rgb = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.rgb_scale = nn.Parameter(torch.tensor(0.1))
        self.sar_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb.shape != sar.shape:
            raise ValueError("cross-attention inputs must share [B,C,H,W]")
        rgb_windows, metadata = _window_partition(rgb, self.window_size)
        sar_windows, sar_metadata = _window_partition(sar, self.window_size)
        if metadata != sar_metadata:
            raise RuntimeError("RGB/SAR window metadata mismatch")
        rgb_norm = self.rgb_norm(rgb_windows)
        sar_norm = self.sar_norm(sar_windows)
        rgb_delta, _ = self.rgb_from_sar(
            query=rgb_norm, key=sar_norm, value=sar_norm, need_weights=False
        )
        sar_delta, _ = self.sar_from_rgb(
            query=sar_norm, key=rgb_norm, value=rgb_norm, need_weights=False
        )
        rgb_windows = rgb_windows + torch.tanh(self.rgb_scale) * rgb_delta
        sar_windows = sar_windows + torch.tanh(self.sar_scale) * sar_delta
        return (
            _window_reverse(rgb_windows, metadata, self.window_size),
            _window_reverse(sar_windows, metadata, self.window_size),
        )


class DecoupledCrossModalFusion(nn.Module):
    """Nested concat -> cross-attention -> common/unique fusion ablations."""

    def __init__(
        self,
        cross_attention: bool,
        common_unique_decoupling: bool,
        common_dim: int = 96,
        unique_dim: int = 32,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        if common_unique_decoupling and not cross_attention:
            raise ValueError("common/unique decoupling requires cross-attention")
        self.cross_attention_enabled = bool(cross_attention)
        self.decoupling_enabled = bool(common_unique_decoupling)
        self.common_dim = int(common_dim)
        self.unique_dim = int(unique_dim)

        if not self.cross_attention_enabled:
            self.concat_fusion = nn.Sequential(
                nn.Conv2d(512, 256, 1, bias=False),
                nn.GroupNorm(16, 256),
                nn.GELU(),
                ConvNeXtBlock2d(256, expansion=2),
            )
            return

        if not self.decoupling_enabled:
            self.rgb_projector = nn.Sequential(
                nn.Conv2d(256, 128, 1, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
            )
            self.sar_projector = nn.Sequential(
                nn.Conv2d(256, 128, 1, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
            )
            self.cross_attention = BidirectionalWindowCrossAttention(
                128, num_heads=4, window_size=window_size
            )
            fusion_channels = 128 * 3
        else:
            self.rgb_common = nn.Conv2d(256, common_dim, 1, bias=False)
            self.sar_common = nn.Conv2d(256, common_dim, 1, bias=False)
            self.rgb_unique = nn.Conv2d(256, unique_dim, 1, bias=False)
            self.sar_unique = nn.Conv2d(256, unique_dim, 1, bias=False)
            self.common_norm_rgb = nn.GroupNorm(8, common_dim)
            self.common_norm_sar = nn.GroupNorm(8, common_dim)
            self.unique_norm_rgb = nn.GroupNorm(8, unique_dim)
            self.unique_norm_sar = nn.GroupNorm(8, unique_dim)
            self.cross_attention = BidirectionalWindowCrossAttention(
                common_dim, num_heads=4, window_size=window_size
            )
            fusion_channels = common_dim * 2 + unique_dim * 3

        self.output = nn.Sequential(
            nn.Conv2d(fusion_channels, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            ConvNeXtBlock2d(256, expansion=2),
        )

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb.shape != sar.shape:
            raise ValueError("fusion inputs must share [B,C,H,W]")
        if not self.cross_attention_enabled:
            fused = self.concat_fusion(torch.cat((rgb, sar), dim=1))
            return fused, {}

        if not self.decoupling_enabled:
            rgb_token = self.rgb_projector(rgb)
            sar_token = self.sar_projector(sar)
            rgb_joint, sar_joint = self.cross_attention(rgb_token, sar_token)
            evidence = torch.cat(
                (rgb_joint, sar_joint, torch.abs(rgb_joint - sar_joint)), dim=1
            )
            return self.output(evidence), {
                "joint_rgb": rgb_joint,
                "joint_sar": sar_joint,
            }

        rgb_common = F.gelu(self.common_norm_rgb(self.rgb_common(rgb)))
        sar_common = F.gelu(self.common_norm_sar(self.sar_common(sar)))
        rgb_unique = F.gelu(self.unique_norm_rgb(self.rgb_unique(rgb)))
        sar_unique = F.gelu(self.unique_norm_sar(self.sar_unique(sar)))
        rgb_common, sar_common = self.cross_attention(rgb_common, sar_common)
        evidence = torch.cat(
            (
                0.5 * (rgb_common + sar_common),
                torch.abs(rgb_common - sar_common),
                rgb_unique,
                sar_unique,
                torch.abs(rgb_unique - sar_unique),
            ),
            dim=1,
        )
        return self.output(evidence), {
            "common_rgb": rgb_common,
            "common_sar": sar_common,
            "unique_rgb": rgb_unique,
            "unique_sar": sar_unique,
        }


def _standardize_tokens(feature: torch.Tensor) -> torch.Tensor:
    tokens = feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1]).float()
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    return tokens / tokens.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)


def decur_redundancy_losses(
    auxiliary: Mapping[str, object],
    off_diagonal_weight: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """Dense DeCUR objective: common correlation -> I, unique -> 0."""

    reference = auxiliary.get("fused_roi")
    if not isinstance(reference, torch.Tensor):
        raise TypeError("auxiliary['fused_roi'] must be a tensor")
    zero = reference.float().sum() * 0.0
    keys = ("common_rgb", "common_sar", "unique_rgb", "unique_sar")
    if not all(isinstance(auxiliary.get(key), torch.Tensor) for key in keys):
        return {
            "loss_common": zero,
            "loss_unique": zero,
            "common_diag": zero.detach(),
            "unique_correlation": zero.detach(),
        }

    common_rgb = _standardize_tokens(auxiliary["common_rgb"])
    common_sar = _standardize_tokens(auxiliary["common_sar"])
    unique_rgb = _standardize_tokens(auxiliary["unique_rgb"])
    unique_sar = _standardize_tokens(auxiliary["unique_sar"])
    common_corr = common_rgb.T @ common_sar / max(common_rgb.shape[0], 1)
    unique_corr = unique_rgb.T @ unique_sar / max(unique_rgb.shape[0], 1)
    diagonal = torch.diagonal(common_corr)
    off_diagonal = common_corr - torch.diag_embed(diagonal)
    common_loss = (diagonal - 1.0).square().mean()
    common_loss = common_loss + float(off_diagonal_weight) * off_diagonal.square().mean()
    unique_loss = unique_corr.square().mean()
    return {
        "loss_common": common_loss,
        "loss_unique": unique_loss,
        "common_diag": diagonal.mean().detach(),
        "unique_correlation": unique_corr.abs().mean().detach(),
    }


class DamageFormer_S8_P5_DCMF(DamageFormer_S8_MICB_DSMIL):
    """P3 RGB/instance knowledge plus a new native-SAR DCMF student."""

    def __init__(self, variant: str = "c3", window_size: int = 8, **kwargs) -> None:
        self.variant = str(variant).lower()
        self.variant_spec = get_p5_variant_spec(self.variant)
        super().__init__(**kwargs)
        self.encoder_2 = NativeSARHierarchicalEncoder(
            filter_augmentation=self.variant_spec.filter_augmentation
        )
        self.fusion_layer_2 = SARMultiScaleProjector()
        self.fusion_layer_3 = DecoupledCrossModalFusion(
            cross_attention=self.variant_spec.cross_attention,
            common_unique_decoupling=self.variant_spec.common_unique_decoupling,
            window_size=window_size,
        )

    def _encode_and_fuse(
        self, pre_data: torch.Tensor, post_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if post_data.ndim != 4 or int(post_data.shape[1]) != 1:
            raise ValueError(
                f"{self.variant_spec.name} expects native SAR [B,1,H,W], "
                f"got {tuple(post_data.shape)}"
            )
        _, pre1, pre2, pre3, pre4 = self.encoder_1(pre_data)
        pre1, pre2, pre3, pre4 = map(
            self._to_nchw, (pre1, pre2, pre3, pre4)
        )
        rgb_feature = self.fusion_layer_1(
            self._multi_scale_fuse(pre1, pre2, pre3, pre4)
        )
        sar_feature = self.fusion_layer_2(self.encoder_2(post_data))
        return rgb_feature, sar_feature

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        bag_specs: Optional[Sequence[BuildingBagSpec]] = None,
        return_aux: bool = False,
    ):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)
        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        fused_feature, fusion_aux = self.fusion_layer_3(rgb_feature, sar_feature)
        fused_roi = fused_feature * roi_weight
        damage_logits_feature = self.clf_2(fused_roi)

        active_specs: Sequence[BuildingBagSpec] = bag_specs or []
        if not active_specs and self.mil_residual_scale > 0:
            active_specs = build_predicted_bag_specs(
                loc_logits_feature,
                min_area_feature=self.predicted_min_area_feature,
            )
        bag_result = None
        if active_specs:
            bag_result = self.micb_dsmil(
                rgb_feature=rgb_feature,
                sar_feature=sar_feature,
                fused_feature=fused_roi,
                bag_specs=active_specs,
            )
            if self.mil_residual_scale > 0 and int(bag_result["bag_count"]) > 0:
                residual = self._scatter_bag_residual(
                    bag_result,
                    active_specs,
                    output_size=damage_logits_feature.shape[-2:],
                    batch_size=damage_logits_feature.shape[0],
                    dtype=damage_logits_feature.dtype,
                    device=damage_logits_feature.device,
                )
                damage_logits_feature = damage_logits_feature.clone()
                damage_logits_feature[:, 1:4] += self.mil_residual_scale * residual

        output_loc = F.interpolate(
            loc_logits_feature,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_damage = F.interpolate(
            damage_logits_feature,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if not return_aux:
            return output_loc, output_damage
        auxiliary: Dict[str, object] = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "rgb_feature": rgb_feature,
            "sar_feature": sar_feature,
            "fused_roi": fused_roi,
            "bag_result": bag_result,
            "bag_count": 0 if bag_result is None else int(bag_result["bag_count"]),
            "mil_residual_scale": self.mil_residual_scale,
            "prototype_bypassed": True,
        }
        auxiliary.update(fusion_aux)
        return output_loc, output_damage, auxiliary

    def experiment_metadata(self) -> Dict[str, object]:
        return {
            "family": "S8-P5 DCMF",
            "variant": self.variant,
            "variant_spec": asdict(self.variant_spec),
            "sar_encoder": "native_1ch_hierarchical_convnext",
            "sar_rgb_input_head_shared": False,
            "retained_positive_modules": ("P3_RGB_localization", "MICB_DSMIL"),
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
        if isinstance(value, torch.Tensor):
            name = str(key)
            cleaned[name[7:] if name.startswith("module.") else name] = value
    return cleaned


P5_NEW_PREFIXES: Tuple[str, ...] = (
    "encoder_2.",
    "fusion_layer_2.",
    "fusion_layer_3.",
)
P5_LATE_PREFIXES: Tuple[str, ...] = ("clf_2.", "micb_dsmil.")


def load_p3_checkpoint_for_p5(
    model: DamageFormer_S8_P5_DCMF,
    checkpoint_path: str | Path,
) -> Dict[str, object]:
    """Strictly preserve P3 RGB/localization/instance parameters.

    Only the declared native-SAR encoder and DCMF modules are allowed to remain
    newly initialized.
    """

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = _extract_state_dict(checkpoint)
    transferable = {
        key: value for key, value in state.items() if not key.startswith(P5_NEW_PREFIXES)
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    illegal_missing = [
        key for key in incompatible.missing_keys if not key.startswith(P5_NEW_PREFIXES)
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "P5 migration changed parameters outside declared new modules: "
            f"missing={illegal_missing}, unexpected={incompatible.unexpected_keys}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "strict_retained_parameters": True,
        "new_prefixes": list(P5_NEW_PREFIXES),
        "missing_new_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def configure_p5_trainable(
    model: DamageFormer_S8_P5_DCMF,
    warmup: bool,
) -> Dict[str, int]:
    """Freeze the proven RGB/localization path; train only declared P5 parts."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = P5_NEW_PREFIXES if warmup else (*P5_NEW_PREFIXES, *P5_LATE_PREFIXES)
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def lock_p5_frozen_modules(model: DamageFormer_S8_P5_DCMF, warmup: bool) -> None:
    model.eval()
    model.encoder_2.train()
    model.fusion_layer_2.train()
    model.fusion_layer_3.train()
    if not warmup:
        model.clf_2.train()
        model.micb_dsmil.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


__all__ = [
    "BidirectionalWindowCrossAttention",
    "DamageFormer_S8_P5_DCMF",
    "DecoupledCrossModalFusion",
    "FrozenSARFilterBank",
    "NativeSARHierarchicalEncoder",
    "P5_VARIANTS",
    "P5VariantSpec",
    "configure_p5_trainable",
    "decur_redundancy_losses",
    "get_p5_variant_spec",
    "load_p3_checkpoint_for_p5",
    "lock_p5_frozen_modules",
]
