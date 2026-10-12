# -*- coding: utf-8 -*-
"""H11-SATR: safe SAR representation transfer for RGB/SAR damage assessment.

H11 is intentionally a small descendant of the validated H6-A2 model.  The
native one-channel SAR pyramid remains the inference path.  T1/T2 add an
external MAE/ViT SAR encoder only at stages 3 and 4 through two exactly-zero
residual transitions.  Consequently a newly constructed H11 model is exactly
H6-A2 until optimisation demonstrates that the transferred representation is
useful.

T1E/T2E are the controlled input-contract correction.  They preserve the
official SARMAE three-channel PatchEmbed and reproduce its public preprocessing
from the un-clipped display-domain SAR channel.  T1/T2 remain available as the
already-completed naive one-channel ablation.

The module contains a minimal MAE-compatible ViT encoder instead of importing
an external repository at runtime.  Its parameter names follow the common MAE
layout (patch_embed.proj, blocks.*.attn.qkv, blocks.*.mlp.fc*) so official
encoder weights can be migrated while decoder/head parameters are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DamageFormer_RGBSAR_H10_H3D import factorize_building_logits
from .DamageFormer_RGBSAR_H6_SDCR import DamageFormerRGBSARH6SDCR
from .DamageFormer_RGBSAR_Heterogeneous import NativeSARPyramid


VALID_H11_VARIANTS = ("t0", "t1", "t2", "t1e", "t2e")


@dataclass(frozen=True)
class H11Variant:
    name: str
    transfer: bool
    trainable_sarmae_blocks: int
    exact_official_input: bool


H11_VARIANTS: Mapping[str, H11Variant] = {
    "t0": H11Variant("H11-T0-Exact-H6A2-Control", False, 0, False),
    "t1": H11Variant("H11-T1-Naive-1Ch-Frozen-SARMAE", True, 0, False),
    "t2": H11Variant("H11-T2-Naive-1Ch-Late-Block-Transfer", True, 2, False),
    "t1e": H11Variant("H11-T1E-Exact-Official-Input-Frozen-SARMAE", True, 0, True),
    "t2e": H11Variant("H11-T2E-Exact-Official-Input-Late-Block-Transfer", True, 2, True),
}


class MAEMlp(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(value)))


class MAEAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("MAE channels must be divisible by the head count")
        self.heads = int(heads)
        self.head_channels = channels // heads
        self.scale = self.head_channels ** -0.5
        self.qkv = nn.Linear(channels, 3 * channels, bias=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch, tokens, 3, self.heads, self.head_channels
        ).permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                query, key, val, dropout_p=0.0, is_causal=False
            )
        else:
            attention = (query * self.scale) @ key.transpose(-2, -1)
            attention = attention.softmax(dim=-1)
            output = attention @ val
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(output)


class MAEBlock(nn.Module):
    def __init__(self, channels: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = MAEAttention(channels, heads)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = MAEMlp(channels, int(channels * mlp_ratio))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class MAEPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, channels: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(image)


class SARMAEEncoder(nn.Module):
    """MAE-compatible ViT encoder returning spatial intermediate features."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        channels: int = 768,
        depth: int = 12,
        heads: int = 12,
        tap_blocks: Sequence[int] = (8, 11),
        trainable_tail_blocks: int = 0,
    ) -> None:
        super().__init__()
        if not tap_blocks or min(tap_blocks) < 0 or max(tap_blocks) >= depth:
            raise ValueError("SARMAE tap_blocks must be non-empty valid block indices")
        self.patch_embed = MAEPatchEmbed(in_channels, channels, patch_size)
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        base_grid = image_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + base_grid * base_grid, channels),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [MAEBlock(channels, heads=heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.tap_blocks = tuple(int(value) for value in tap_blocks)
        self.trainable_tail_blocks = int(trainable_tail_blocks)
        self._initialise()

    def _initialise(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _position_embedding(
        self, height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        cls_position = self.pos_embed[:, :1]
        patch_position = self.pos_embed[:, 1:]
        source_side = int(round(patch_position.shape[1] ** 0.5))
        if source_side * source_side != patch_position.shape[1]:
            raise RuntimeError("SARMAE positional embedding is not a square grid")
        patch_position = patch_position.reshape(
            1, source_side, source_side, patch_position.shape[-1]
        ).permute(0, 3, 1, 2)
        if (source_side, source_side) != (height, width):
            patch_position = F.interpolate(
                patch_position.float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        patch_position = patch_position.flatten(2).transpose(1, 2)
        return torch.cat([cls_position, patch_position], dim=1).to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _tokens_to_map(
        tokens: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        patches = tokens[:, 1:]
        return patches.transpose(1, 2).reshape(
            patches.shape[0], patches.shape[2], height, width
        ).contiguous()

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(
                "SARMAE transfer input channel mismatch: "
                f"expected [B,{self.in_channels},H,W], received {tuple(image.shape)}"
            )
        patches = self.patch_embed(image)
        height, width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1).to(tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._position_embedding(
            height, width, tokens.dtype, tokens.device
        )

        frozen_prefix = len(self.blocks) - self.trainable_tail_blocks
        taps: Dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            # A frozen prefix is evaluated without autograd.  T2 therefore
            # backpropagates only through the last two transformer blocks.
            if index < frozen_prefix:
                with torch.no_grad():
                    tokens = block(tokens)
                tokens = tokens.detach()
            else:
                tokens = block(tokens)
            if index in self.tap_blocks:
                taps[index] = tokens
        tokens = self.norm(tokens)
        if self.tap_blocks[-1] == len(self.blocks) - 1:
            taps[self.tap_blocks[-1]] = tokens
        if any(index not in taps for index in self.tap_blocks):
            raise RuntimeError("SARMAE did not produce the requested feature taps")
        return tuple(
            self._tokens_to_map(taps[index], height, width)
            for index in self.tap_blocks
        )  # type: ignore[return-value]


class TransferProjection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )


class SARMAEResidualPyramid(NativeSARPyramid):
    """Preserve native A2 features and add safe deep SARMAE residuals."""

    def __init__(
        self,
        trainable_tail_blocks: int = 0,
        exact_official_input: bool = False,
        encoder_kwargs: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        options = dict(encoder_kwargs or {})
        options["trainable_tail_blocks"] = trainable_tail_blocks
        options["in_channels"] = 3 if exact_official_input else 1
        self.exact_official_input = bool(exact_official_input)
        self.sarmae_encoder = SARMAEEncoder(**options)
        transfer_channels = self.sarmae_encoder.channels
        self.transfer_adapters = nn.ModuleDict(
            {
                "stage3": TransferProjection(transfer_channels, 384),
                "stage4": TransferProjection(transfer_channels, 768),
            }
        )
        self.transfer_transition = nn.ParameterDict(
            {
                "stage3": nn.Parameter(torch.tensor(0.0)),
                "stage4": nn.Parameter(torch.tensor(0.0)),
            }
        )
        self.last_transfer_diagnostics: Dict[str, torch.Tensor] = {}
        self._display_sar_context: Optional[torch.Tensor] = None
        self.register_buffer(
            "sarmae_imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sarmae_imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def set_display_sar_context(self, display_sar: torch.Tensor) -> None:
        if not self.exact_official_input:
            return
        if display_sar.ndim != 4 or display_sar.shape[1] != 1:
            raise ValueError("official SARMAE input requires display SAR [B,1,H,W]")
        self._display_sar_context = display_sar

    def clear_display_sar_context(self) -> None:
        self._display_sar_context = None

    def official_sarmae_input(self, display_sar: torch.Tensor) -> torch.Tensor:
        """Reproduce the official grayscale-repeat and ImageNet normalization."""

        if display_sar.ndim != 4 or display_sar.shape[1] != 1:
            raise ValueError("official SARMAE preprocessing expects [B,1,H,W]")
        unit = display_sar.float().clamp(0.0, 255.0) / 255.0
        repeated = unit.repeat(1, 3, 1, 1)
        return (repeated - self.sarmae_imagenet_mean) / self.sarmae_imagenet_std

    def _sarmae_input(self, native_sar: torch.Tensor) -> torch.Tensor:
        if not self.exact_official_input:
            return native_sar
        if self._display_sar_context is None:
            raise RuntimeError(
                "T1E/T2E require the H4 two-channel dataset contract: "
                "channel 0 native normalized SAR, channel 1 raw display SAR"
            )
        display = self._display_sar_context
        if display.shape[0] != native_sar.shape[0] or display.shape[-2:] != native_sar.shape[-2:]:
            raise RuntimeError("display SAR context does not match native SAR shape")
        return self.official_sarmae_input(display).to(device=native_sar.device)

    def forward(self, sar: torch.Tensor) -> List[torch.Tensor]:
        native = super().forward(sar)
        mae_stage3, mae_stage4 = self.sarmae_encoder(self._sarmae_input(sar))
        transferred = {"stage3": mae_stage3, "stage4": mae_stage4}
        diagnostics: Dict[str, torch.Tensor] = {}
        outputs = list(native)
        for stage_name, output_index in (("stage3", 2), ("stage4", 3)):
            candidate = self.transfer_adapters[stage_name](transferred[stage_name])
            candidate = F.interpolate(
                candidate,
                size=native[output_index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            transition = torch.tanh(self.transfer_transition[stage_name])
            residual = transition * candidate
            outputs[output_index] = native[output_index] + residual
            diagnostics[f"h11_{stage_name}_transfer_transition"] = transition.reshape(
                1, 1, 1, 1
            )
            diagnostics[f"h11_{stage_name}_transfer_candidate_abs"] = (
                candidate.abs().mean(1, keepdim=True)
            )
            diagnostics[f"h11_{stage_name}_transfer_residual_abs"] = (
                residual.abs().mean(1, keepdim=True)
            )
        self.last_transfer_diagnostics = diagnostics
        return outputs


class DamageFormerRGBSARH11SATR(DamageFormerRGBSARH6SDCR):
    """H6-A2 with a controlled external SAR representation transfer path."""

    def __init__(
        self,
        variant: str = "t1",
        pretrained_rgb: bool = True,
        latent_channels: Sequence[int] = (64, 96, 128, 160),
    ) -> None:
        variant = str(variant).lower()
        if variant not in VALID_H11_VARIANTS:
            raise ValueError(f"unknown H11 variant {variant!r}")
        super().__init__(
            variant="a2",
            pretrained_rgb=pretrained_rgb,
            latent_channels=latent_channels,
        )
        self.variant = variant
        self.h11_config = H11_VARIANTS[variant]
        if self.h11_config.transfer:
            self.sar_encoder = SARMAEResidualPyramid(
                trainable_tail_blocks=self.h11_config.trainable_sarmae_blocks,
                exact_official_input=self.h11_config.exact_official_input,
            )

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        if (
            isinstance(self.sar_encoder, SARMAEResidualPyramid)
            and self.h11_config.exact_official_input
        ):
            if post_sar.ndim != 4 or post_sar.shape[1] < 2:
                raise RuntimeError(
                    "H11-T1E/T2E require two SAR channels from dataset variant h4"
                )
            self.sar_encoder.set_display_sar_context(post_sar[:, 1:2])
        try:
            outputs = super().forward(pre_rgb, post_sar, return_dict=True)
        finally:
            if isinstance(self.sar_encoder, SARMAEResidualPyramid):
                self.sar_encoder.clear_display_sar_context()
        affected, severity = factorize_building_logits(outputs["output_dam"])
        # These are diagnostics only.  H11 does not add a severity head or an
        # auxiliary loss; it keeps the original A2 four-class objective.
        outputs["h10_affected_logits"] = affected
        outputs["h10_severity_logits"] = severity
        if isinstance(self.sar_encoder, SARMAEResidualPyramid):
            outputs.update(self.sar_encoder.last_transfer_diagnostics)
        if return_dict:
            return outputs
        return outputs["output_loc"], outputs["output_dam"]


def build_h11_satr_damageformer(
    variant: str,
    pretrained_rgb: bool = True,
) -> DamageFormerRGBSARH11SATR:
    return DamageFormerRGBSARH11SATR(
        variant=variant,
        pretrained_rgb=pretrained_rgb,
    )


__all__ = [
    "VALID_H11_VARIANTS",
    "H11_VARIANTS",
    "SARMAEEncoder",
    "SARMAEResidualPyramid",
    "DamageFormerRGBSARH11SATR",
    "build_h11_satr_damageformer",
]
