from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FrozenFeatureSource, freeze_module, raw_display_channel


def _normalise_checkpoint(payload) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("SAR-DINO checkpoint must contain a mapping")
    for key in ("teacher", "model", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    cleaned = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        for prefix in ("module.", "backbone.", "dino_sarmodel."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def _load_official_vit(source_root: Path, patch_size: int) -> nn.Module:
    path = source_root / "mmov_seg" / "vision_transformer.py"
    if not path.is_file():
        raise FileNotFoundError(f"official MM-OVSeg ViT source is missing: {path}")
    spec = importlib.util.spec_from_file_location("p3_reset_mmov_vit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official MM-OVSeg ViT from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vit_base(patch_size=patch_size, num_classes=0)


class MMOVSARDINOSource(FrozenFeatureSource):
    """Frozen MM-OVSeg SAR-DINO blocks 4/8/12 with its official input contract."""

    output_channels = {"block4": 768, "block8": 768, "block12": 768}

    def __init__(
        self,
        checkpoint_path: str,
        source_root: str,
        input_size: int = 384,
        pixel_mean: Tuple[float, float, float] = (122.7709383, 116.7460125, 104.09373615),
        pixel_std: Tuple[float, float, float] = (68.5005327, 66.6321579, 70.3231630),
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MM-OVSeg SAR-DINO checkpoint is missing: {checkpoint}")
        state = _normalise_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False))
        patch = state.get("patch_embed.proj.weight")
        if patch is None or patch.ndim != 4 or patch.shape[1] != 3:
            raise RuntimeError("checkpoint lacks a compatible three-channel patch_embed.proj.weight")
        patch_size = int(patch.shape[-1])
        if patch.shape[-2] != patch_size:
            raise RuntimeError("non-square SAR-DINO patch embedding is unsupported")
        self.encoder = _load_official_vit(Path(source_root), patch_size)
        incompatible = self.encoder.load_state_dict(state, strict=False)
        required = {"cls_token", "pos_embed", "patch_embed.proj.weight", "norm.weight", "norm.bias"}
        missing_required = sorted(required.intersection(incompatible.missing_keys))
        present_blocks = {int(key.split(".")[1]) for key in state if key.startswith("blocks.") and key.split(".")[1].isdigit()}
        if missing_required or len(present_blocks) < 12:
            raise RuntimeError(
                f"incomplete SAR-DINO encoder: missing={missing_required}, blocks={sorted(present_blocks)}"
            )
        self.encoder = freeze_module(self.encoder.float())
        self.input_size = int(input_size)
        self.patch_size = patch_size
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, post_sar: torch.Tensor) -> torch.Tensor:
        raw = raw_display_channel(post_sar).repeat(1, 3, 1, 1)
        raw = (raw - self.pixel_mean.to(raw)) / self.pixel_std.to(raw)
        return F.interpolate(raw, (self.input_size, self.input_size), mode="bilinear", align_corners=False)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, 1:]
        side = int(math.isqrt(int(patches.shape[1])))
        if side * side != patches.shape[1]:
            raise RuntimeError(f"SAR-DINO token count {patches.shape[1]} is not a square grid")
        return patches.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    @torch.no_grad()
    def forward(self, post_sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encoder.get_intermediate_layers(self.preprocess(post_sar), n=12)
        if len(tokens) < 12:
            raise RuntimeError(f"SAR-DINO returned only {len(tokens)} blocks")
        return {
            "block4": self._tokens_to_map(tokens[3]),
            "block8": self._tokens_to_map(tokens[7]),
            "block12": self._tokens_to_map(tokens[11]),
        }
