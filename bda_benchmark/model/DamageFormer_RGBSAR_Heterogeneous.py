# -*- coding: utf-8 -*-
"""DamageFormer rebuilt for heterogeneous pre-RGB / post-SAR observations.

This is a new branch from the original DamageFormer design.  It deliberately
does not import the P4/P5/recovery models and it does not use a teacher model.

The implementation combines three ideas at architecture level:

* modality-specific encoders and a joint encoder (CROMA, NeurIPS 2023);
* common/unique representation separation (DeCUR, ECCV 2024);
* local sparse cross-modal exchange instead of global attention (KtB, CVPR
  2025), with the stage-wise exchange pattern used by CMX.

The RGB path alone predicts building localization.  Damage classification uses
the joint RGB/SAR representation.  This prevents noisy post-event SAR from
moving the geometric building prior while still allowing SAR-unique evidence
to explain damage severity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


VALID_VARIANTS = ("h0", "h1", "h2", "h3")
STAGE_CHANNELS = (96, 192, 384, 768)


@dataclass(frozen=True)
class HeterogeneousVariant:
    name: str
    native_sar_encoder: bool
    common_unique: bool
    local_correspondence: bool
    sparse_correspondence: bool
    building_prior: bool


VARIANTS: Mapping[str, HeterogeneousVariant] = {
    "h0": HeterogeneousVariant(
        "H0-Original-PseudoRGB-Control", False, False, False, False, False
    ),
    "h1": HeterogeneousVariant(
        "H1-Modality-Specific-Encoders", True, False, False, False, False
    ),
    "h2": HeterogeneousVariant(
        "H2-CommonUnique-LocalFusion", True, True, True, False, False
    ),
    "h3": HeterogeneousVariant(
        "H3-SparseHeterogeneous-DamageFormer", True, True, True, True, True
    ),
}


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ConvNeXtSARBlock(nn.Module):
    """Small-batch-safe residual block for the SAR-only hierarchy."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.act = nn.GELU()
        self.reduce = nn.Conv2d(4 * channels, channels, 1)
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.reduce(self.act(self.expand(x)))
        return residual + self.layer_scale * x


class NativeSARPyramid(nn.Module):
    """A genuinely one-channel SAR hierarchy with no RGB/ImageNet stem."""

    def __init__(self, depths: Sequence[int] = (2, 2, 4, 2)) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError("SAR depths must contain four stages")
        self.stem = nn.Sequential(
            ConvGNAct(1, 48, kernel_size=7, stride=2),
            ConvGNAct(48, STAGE_CHANNELS[0], kernel_size=3, stride=2),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (channels, depth) in enumerate(zip(STAGE_CHANNELS, depths)):
            blocks = [
                ConvNeXtSARBlock(channels, dilation=2 if index == 0 and i % 2 else 1)
                for i in range(depth)
            ]
            self.stages.append(nn.Sequential(*blocks))
            if index < 3:
                self.downsamples.append(
                    ConvGNAct(channels, STAGE_CHANNELS[index + 1], 3, stride=2)
                )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, sar: torch.Tensor) -> List[torch.Tensor]:
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError(f"native SAR encoder expects [B,1,H,W], got {tuple(sar.shape)}")
        x = self.stem(sar)
        outputs: List[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return outputs


class SwinRGBPyramid(nn.Module):
    """Robustly extract the four semantic Swin-T stages in NCHW layout."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        network = swin_t(weights=weights)
        network.norm = nn.Identity()
        network.head = nn.Identity()
        self.features = network.features

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        stage_outputs: List[torch.Tensor] = []
        x = image
        sequential_count = 0
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                # torchvision's first Sequential is patch embedding.  The next
                # four are the semantic stages used by original DamageFormer.
                if sequential_count > 0:
                    stage_outputs.append(x.permute(0, 3, 1, 2).contiguous())
                sequential_count += 1
        if len(stage_outputs) != 4:
            raise RuntimeError(
                f"expected four Swin stages, received {len(stage_outputs)}; "
                "the installed torchvision Swin layout has changed"
            )
        return stage_outputs


class PseudoRGBSwinPyramid(SwinRGBPyramid):
    def forward(self, sar: torch.Tensor) -> List[torch.Tensor]:
        if sar.ndim != 4 or sar.shape[1] != 3:
            raise ValueError(f"H0 expects pseudo-RGB SAR [B,3,H,W], got {tuple(sar.shape)}")
        return super().forward(sar)


class LocalSparseCorrespondence(nn.Module):
    """Cross-attend to a bounded local SAR neighbourhood.

    Unlike global attention, a pixel can only import SAR evidence from nearby
    positions.  H3 additionally retains only the top-k correspondences.  This
    both limits optical/SAR registration error and blocks unrelated global SAR
    clutter from contaminating RGB features.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        window_size: int = 3,
        topk: int | None = None,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        neighbours = window_size * window_size
        if topk is not None and not 1 <= topk <= neighbours:
            raise ValueError("topk must be within the local neighbourhood")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = window_size
        self.topk = topk
        self.scale = self.head_dim**-0.5
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, rgb_common: torch.Tensor, sar_common: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb_common.shape != sar_common.shape:
            raise ValueError("local correspondence requires equal RGB/SAR feature shapes")
        batch, channels, height, width = rgb_common.shape
        locations = height * width
        neighbours = self.window_size * self.window_size

        q = self.query(rgb_common).reshape(
            batch, self.heads, self.head_dim, locations
        )
        key_patches = F.unfold(
            self.key(sar_common),
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).reshape(batch, self.heads, self.head_dim, neighbours, locations)
        value_patches = F.unfold(
            self.value(sar_common),
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).reshape(batch, self.heads, self.head_dim, neighbours, locations)

        scores = torch.einsum("bhdl,bhdkl->bhkl", q, key_patches) * self.scale
        if self.topk is not None and self.topk < neighbours:
            selected_values, selected_indices = scores.topk(self.topk, dim=2)
            sparse_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
            scores = sparse_scores.scatter(2, selected_indices, selected_values)
        attention = scores.softmax(dim=2)
        aligned = torch.einsum("bhkl,bhdkl->bhdl", attention, value_patches)
        aligned = aligned.reshape(batch, channels, height, width)
        aligned = self.output(aligned)

        # Normalized entropy is a deterministic reliability diagnostic: 1 means
        # no clear local match; 0 means a concentrated match.
        entropy = -(attention.float().clamp_min(1e-8).log() * attention.float()).sum(2)
        normalizer = math.log(float(self.topk or neighbours))
        if normalizer > 0:
            entropy = entropy / normalizer
        reliability = (1.0 - entropy.mean(1, keepdim=True)).clamp(0.0, 1.0)
        # Entropy is evaluated on flattened spatial locations [B, 1, H*W].
        # Restore the NCHW map before it is concatenated with stage features.
        reliability = reliability.reshape(batch, 1, height, width)
        scale = torch.tanh(self.residual_scale)
        return self.norm(sar_common + scale * aligned), reliability


class CommonUniqueStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        use_correspondence: bool,
        sparse: bool,
    ) -> None:
        super().__init__()
        self.rgb_common = ConvGNAct(in_channels, latent_channels, 1)
        self.sar_common = ConvGNAct(in_channels, latent_channels, 1)
        self.rgb_unique = ConvGNAct(in_channels, latent_channels, 1)
        self.sar_unique = ConvGNAct(in_channels, latent_channels, 1)
        self.correspondence = (
            LocalSparseCorrespondence(
                latent_channels,
                heads=4,
                window_size=3,
                topk=3 if sparse else None,
            )
            if use_correspondence
            else None
        )
        descriptor_channels = 5 * latent_channels + 1
        self.fuse = nn.Sequential(
            ConvGNAct(descriptor_channels, 2 * latent_channels, 1),
            ConvGNAct(2 * latent_channels, latent_channels, 3),
        )

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_common = self.rgb_common(rgb)
        sar_common = self.sar_common(sar)
        rgb_unique = self.rgb_unique(rgb)
        sar_unique = self.sar_unique(sar)
        if self.correspondence is None:
            aligned_sar = sar_common
            reliability = torch.ones_like(rgb_common[:, :1])
        else:
            aligned_sar, reliability = self.correspondence(rgb_common, sar_common)
        descriptor = torch.cat(
            [
                rgb_common,
                aligned_sar,
                (rgb_common - aligned_sar).abs(),
                rgb_unique,
                sar_unique,
                reliability,
            ],
            dim=1,
        )
        diagnostics = {
            "rgb_common": rgb_common,
            "sar_common": aligned_sar,
            "rgb_unique": rgb_unique,
            "sar_unique": sar_unique,
            "reliability": reliability,
        }
        return self.fuse(descriptor), diagnostics


class NaiveHeterogeneousStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.rgb = ConvGNAct(in_channels, out_channels, 1)
        self.sar = ConvGNAct(in_channels, out_channels, 1)
        self.fuse = nn.Sequential(
            ConvGNAct(3 * out_channels, 2 * out_channels, 1),
            ConvGNAct(2 * out_channels, out_channels, 3),
        )

    def forward(
        self, rgb: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_latent = self.rgb(rgb)
        sar_latent = self.sar(sar)
        fused = self.fuse(
            torch.cat([rgb_latent, sar_latent, (rgb_latent - sar_latent).abs()], 1)
        )
        return fused, {"reliability": torch.ones_like(fused[:, :1])}


class MultiScaleDecoder(nn.Module):
    def __init__(self, stage_channels: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvGNAct(channels, 96, 1) for channels in stage_channels]
        )
        self.fuse = nn.Sequential(
            ConvGNAct(96 * len(stage_channels), out_channels, 3),
            ConvGNAct(out_channels, out_channels, 3),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("decoder received an unexpected number of stages")
        target_size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(value)
        return self.fuse(torch.cat(projected, dim=1))


class OriginalResBlock(nn.Module):
    """The residual block used by the public original DamageFormer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        value = self.relu(self.bn1(self.conv1(x)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + identity)


class DamageFormerRGBSARHeterogeneous(nn.Module):
    """Native RGB/SAR DamageFormer variants H1-H3."""

    def __init__(
        self,
        variant: str = "h3",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in ("h1", "h2", "h3"):
            raise ValueError("heterogeneous model variant must be h1, h2, or h3")
        if len(latent_channels) != 4:
            raise ValueError("latent_channels must define four stages")
        self.variant = variant
        self.variant_config = VARIANTS[variant]
        self.rgb_encoder = SwinRGBPyramid(pretrained=pretrained_rgb)
        self.sar_encoder = NativeSARPyramid()

        stages: List[nn.Module] = []
        for source_channels, latent in zip(STAGE_CHANNELS, latent_channels):
            if self.variant_config.common_unique:
                stages.append(
                    CommonUniqueStage(
                        source_channels,
                        latent,
                        use_correspondence=self.variant_config.local_correspondence,
                        sparse=self.variant_config.sparse_correspondence,
                    )
                )
            else:
                stages.append(NaiveHeterogeneousStage(source_channels, latent))
        self.joint_stages = nn.ModuleList(stages)
        self.localization_decoder = MultiScaleDecoder(STAGE_CHANNELS, 256)
        self.damage_decoder = MultiScaleDecoder(latent_channels, 256)
        self.localization_head = nn.Conv2d(256, 2, 1)
        self.damage_head = nn.Conv2d(256, 4, 1)
        self.sar_aux_head = nn.Sequential(
            ConvGNAct(STAGE_CHANNELS[-1], 256, 1),
            nn.Conv2d(256, 4, 1),
        )

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)

        joint_stages: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for index, (module, rgb, sar) in enumerate(
            zip(self.joint_stages, rgb_stages, sar_stages), start=1
        ):
            joint, stage_diagnostics = module(rgb, sar)
            joint_stages.append(joint)
            for key, value in stage_diagnostics.items():
                diagnostics[f"stage{index}_{key}"] = value

        localization_feature = self.localization_decoder(rgb_stages)
        localization_logits_low = self.localization_head(localization_feature)
        localization_probability = localization_logits_low.softmax(1)[:, 1:2]

        damage_feature = self.damage_decoder(joint_stages)
        if self.variant_config.building_prior:
            # Soft, non-suppressive prior: background is attenuated but never
            # erased, so localization mistakes cannot zero the damage gradient.
            prior = localization_probability.detach()
            damage_feature = damage_feature * (0.5 + 0.5 * prior)
        damage_logits_low = self.damage_head(damage_feature)
        sar_aux_low = self.sar_aux_head(sar_stages[-1])

        output_loc = F.interpolate(
            localization_logits_low,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        output_dam = F.interpolate(
            damage_logits_low, size=input_size, mode="bilinear", align_corners=False
        )
        sar_aux = F.interpolate(
            sar_aux_low, size=input_size, mode="bilinear", align_corners=False
        )
        outputs: Dict[str, torch.Tensor] = {
            "output_loc": output_loc,
            "output_dam": output_dam,
            "sar_aux_logits": sar_aux,
            "localization_probability_low": localization_probability,
            **diagnostics,
        }
        if return_dict:
            return outputs
        return output_loc, output_dam


class OriginalDamageFormerControl(nn.Module):
    """H0 reproduction of the public original DamageFormer architecture."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.variant = "h0"
        self.variant_config = VARIANTS["h0"]
        self.rgb_encoder = SwinRGBPyramid(pretrained=pretrained)
        self.sar_encoder = PseudoRGBSwinPyramid(pretrained=pretrained)
        self.fusion_layer_1 = OriginalResBlock(1440, 256)
        self.fusion_layer_2 = OriginalResBlock(1440, 256)
        self.fusion_layer_3 = OriginalResBlock(512, 256)
        self.localization_head = nn.Conv2d(256, 2, 1)
        self.damage_head = nn.Conv2d(256, 4, 1)

    @staticmethod
    def _original_multiscale(stages: Sequence[torch.Tensor]) -> torch.Tensor:
        target_size = stages[0].shape[-2:]
        values = []
        for stage in reversed(stages[1:]):
            values.append(
                F.interpolate(
                    stage, size=target_size, mode="bilinear", align_corners=False
                )
            )
        values.append(stages[0])
        return torch.cat(values, dim=1)

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar)
        rgb = self.fusion_layer_1(self._original_multiscale(rgb_stages))
        sar = self.fusion_layer_2(self._original_multiscale(sar_stages))
        damage = self.fusion_layer_3(torch.cat([rgb, sar], dim=1))
        output_loc = F.interpolate(
            self.localization_head(rgb),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        output_dam = F.interpolate(
            self.damage_head(damage),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        outputs = {"output_loc": output_loc, "output_dam": output_dam}
        if return_dict:
            return outputs
        return output_loc, output_dam


def build_heterogeneous_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> nn.Module:
    variant = variant.lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose {VALID_VARIANTS}")
    if variant == "h0":
        return OriginalDamageFormerControl(pretrained=pretrained_rgb)
    return DamageFormerRGBSARHeterogeneous(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


def common_unique_regularization(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    unchanged_weight: float = 1.0,
    orthogonality_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Supervised DeCUR-style regularization without invalid global alignment.

    Pre-event RGB and post-event SAR are not two views of an unchanged scene.
    Common features are therefore aligned only over background/intact pixels.
    Common/unique decorrelation is computed within each modality over all valid
    pixels.  This avoids aligning away real damaged/destroyed evidence.
    """

    reference = next(iter(outputs.values()))
    zero = reference.float().sum() * 0.0
    unchanged_terms: List[torch.Tensor] = []
    orthogonal_terms: List[torch.Tensor] = []
    for stage in range(1, 5):
        required = [
            f"stage{stage}_rgb_common",
            f"stage{stage}_sar_common",
            f"stage{stage}_rgb_unique",
            f"stage{stage}_sar_unique",
        ]
        if not all(key in outputs for key in required):
            continue
        rgb_common, sar_common, rgb_unique, sar_unique = [
            outputs[key].float() for key in required
        ]
        target = F.interpolate(
            labels.unsqueeze(1).float(),
            size=rgb_common.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        cosine_distance = 1.0 - F.cosine_similarity(
            rgb_common, sar_common, dim=1
        )
        # Background vastly outnumbers buildings.  Average each unchanged
        # class separately before averaging the two, so intact buildings are
        # not numerically erased by background.
        class_terms = []
        for class_index in (0, 1):
            mask = target == class_index
            if mask.any():
                class_terms.append(cosine_distance[mask].mean())
        if class_terms:
            unchanged_terms.append(torch.stack(class_terms).mean())

        for common, unique in (
            (rgb_common, rgb_unique),
            (sar_common, sar_unique),
        ):
            common = F.normalize(common.flatten(2), dim=1)
            unique = F.normalize(unique.flatten(2), dim=1)
            orthogonal_terms.append((common * unique).sum(1).square().mean())

    unchanged_loss = torch.stack(unchanged_terms).mean() if unchanged_terms else zero
    orthogonality_loss = (
        torch.stack(orthogonal_terms).mean() if orthogonal_terms else zero
    )
    total = (
        float(unchanged_weight) * unchanged_loss
        + float(orthogonality_weight) * orthogonality_loss
    )
    return {
        "representation_loss": total,
        "common_unchanged_loss": unchanged_loss,
        "common_unique_orthogonality_loss": orthogonality_loss,
    }


__all__ = [
    "VALID_VARIANTS",
    "VARIANTS",
    "DamageFormerRGBSARHeterogeneous",
    "OriginalDamageFormerControl",
    "NativeSARPyramid",
    "LocalSparseCorrespondence",
    "build_heterogeneous_damageformer",
    "common_unique_regularization",
]
