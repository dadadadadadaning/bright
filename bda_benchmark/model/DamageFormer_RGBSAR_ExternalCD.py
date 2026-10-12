# -*- coding: utf-8 -*-
"""Fair BRIGHT adapters for whole-model optical--SAR CD baselines.

``x1`` is the repository-native plain non-shared dual-encoder control.
``x2`` wraps the authors' HeteCD implementation.  It is deliberately not
silently reimplemented: the official research-only source must exist under
``third_party/HeteCD`` before x2 can be constructed.

Both variants expose the same BRIGHT contract: native 3-channel pre-event
RGB, native 1-channel post-event SAR and four semantic damage logits.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_RGBSAR_Heterogeneous import (
    DamageFormerRGBSARHeterogeneous,
)


VALID_VARIANTS = ("x1", "x2")


@dataclass(frozen=True)
class ExternalVariant:
    name: str
    source: str
    faithful_official_core: bool


VARIANTS: Mapping[str, ExternalVariant] = {
    "x1": ExternalVariant(
        "X1-Plain-NonShared-DualEncoder-4C",
        "BRIGHT controlled baseline",
        False,
    ),
    "x2": ExternalVariant(
        "X2-HeteCD-BRIGHT-4C",
        "weiAI1996/HeteCD (research-only license)",
        True,
    ),
}


def damage_to_localization_logits(damage_logits: torch.Tensor) -> torch.Tensor:
    """Derive BG/building logits without adding a second supervised head."""
    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ValueError("damage logits must be [B,4,H,W]")
    building = torch.logsumexp(damage_logits[:, 1:4].float(), dim=1, keepdim=True)
    return torch.cat([damage_logits[:, :1].float(), building], dim=1).to(
        damage_logits.dtype
    )


class PlainDualBRIGHT(nn.Module):
    """Minimal control: independent encoders + concat/absolute-difference."""

    variant = "x1"

    def __init__(self, pretrained_rgb: bool = True) -> None:
        super().__init__()
        core = DamageFormerRGBSARHeterogeneous(
            variant="h1", pretrained_rgb=pretrained_rgb
        )
        # Preserve optimizer group names used by the common trainer.
        self.rgb_encoder = core.rgb_encoder
        self.sar_encoder = core.sar_encoder
        self.joint_stages = core.joint_stages
        self.damage_decoder = core.damage_decoder
        self.damage_head = core.damage_head

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        if post_sar.ndim != 4 or post_sar.shape[1] < 1:
            raise ValueError("X1 expects native SAR [B,1,H,W]")
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar[:, :1])
        joint = [
            module(rgb, sar)[0]
            for module, rgb, sar in zip(
                self.joint_stages, rgb_stages, sar_stages
            )
        ]
        logits = self.damage_head(self.damage_decoder(joint))
        logits = F.interpolate(
            logits, input_size, mode="bilinear", align_corners=False
        )
        outputs = {
            "output_dam": logits,
            "output_loc": damage_to_localization_logits(logits),
        }
        return outputs if return_dict else (outputs["output_loc"], logits)


def _official_hetecd_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "third_party" / "HeteCD"


def _import_official_hetecd():
    root = _official_hetecd_root()
    required = (root / "models" / "hetecd.py", root / "models" / "base_block.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "X2 requires the official HeteCD repository. Missing: "
            + ", ".join(missing)
            + ". Clone https://github.com/weiAI1996/HeteCD into third_party/HeteCD."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("models.hetecd")
    return module


class OfficialHeteCDBRIGHT(nn.Module):
    """Thin BRIGHT adapter around the authors' official HeteCD core.

    Architectural changes are restricted to the declared task interface:
    the SAR patch embedding accepts one channel and the decoder predicts four
    BRIGHT classes.  The two Transformer encoders and 3D difference decoder
    remain the official implementation.
    """

    variant = "x2"

    def __init__(self, pretrained_rgb: bool = True) -> None:
        super().__init__()
        official = _import_official_hetecd()
        core = official.hetecd(input_nc=3, output_nc=4, decoder_softmax=False)
        self.rgb_encoder = core.backbone1
        self.sar_encoder = core.backbone2
        self.fusion_decoder = core.CD_Decoder

        # The official CD_3D_Decoder accepts ``output_nc`` but hard-codes
        # ``make_pred_bcd`` to two output channels because HeteCD is a binary
        # change detector.  Replace only that task head for BRIGHT's four
        # semantic classes; all official encoders and difference blocks stay
        # untouched.
        self.fusion_decoder.make_pred_bcd = official.make_prediction(
            in_channels=self.fusion_decoder.embedding_dim,
            out_channels=4,
            sigmoid=False,
        )

        old = self.sar_encoder.patch_embed1.proj
        replacement = nn.Conv2d(
            1,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
        )
        # Channel averaging is only an initialization.  Unlike the failed
        # frozen SARMAE experiment, this SAR encoder is trained end-to-end.
        with torch.no_grad():
            replacement.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            if old.bias is not None:
                replacement.bias.copy_(old.bias)
        self.sar_encoder.patch_embed1.proj = replacement

    def forward(
        self,
        pre_rgb: torch.Tensor,
        post_sar: torch.Tensor,
        return_dict: bool = False,
    ):
        input_size = pre_rgb.shape[-2:]
        rgb_stages = self.rgb_encoder(pre_rgb)
        sar_stages = self.sar_encoder(post_sar[:, :1])
        decoder_outputs = self.fusion_decoder(rgb_stages, sar_stages)
        logits = (
            decoder_outputs[-1]
            if isinstance(decoder_outputs, (list, tuple))
            else decoder_outputs
        )
        if not torch.is_tensor(logits) or logits.ndim != 4:
            shape = getattr(logits, "shape", None)
            raise RuntimeError(
                "official HeteCD decoder must return a tensor [B,C,H,W]; "
                f"received {type(logits).__name__} with shape={shape}"
            )
        if logits.shape[1] != 4:
            raise RuntimeError(
                "BRIGHT HeteCD task head must emit four channels; "
                f"received shape={tuple(logits.shape)}"
            )
        logits = F.interpolate(
            logits, input_size, mode="bilinear", align_corners=False
        )
        outputs: Dict[str, torch.Tensor] = {
            "output_dam": logits,
            "output_loc": damage_to_localization_logits(logits),
            # The common trainer's BRIGHT-safe consistency term aligns only
            # BG/Intact cells. Zeros make its unrelated orthogonality part null.
            "stage3_rgb_common": rgb_stages[2],
            "stage3_sar_common": sar_stages[2],
            "stage3_rgb_unique": torch.zeros_like(rgb_stages[2]),
            "stage3_sar_unique": torch.zeros_like(sar_stages[2]),
            "stage4_rgb_common": rgb_stages[3],
            "stage4_sar_common": sar_stages[3],
            "stage4_rgb_unique": torch.zeros_like(rgb_stages[3]),
            "stage4_sar_unique": torch.zeros_like(sar_stages[3]),
        }
        return outputs if return_dict else (outputs["output_loc"], logits)


def build_external_cd_bright(
    variant: str,
    pretrained_rgb: bool = True,
) -> nn.Module:
    variant = variant.lower()
    if variant == "x1":
        return PlainDualBRIGHT(pretrained_rgb=pretrained_rgb)
    if variant == "x2":
        return OfficialHeteCDBRIGHT(pretrained_rgb=pretrained_rgb)
    raise ValueError(f"unknown external CD variant {variant!r}")


__all__ = [
    "VALID_VARIANTS",
    "VARIANTS",
    "damage_to_localization_logits",
    "PlainDualBRIGHT",
    "OfficialHeteCDBRIGHT",
    "build_external_cd_bright",
]
