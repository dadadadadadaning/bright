# -*- coding: utf-8 -*-
"""Minimal building-guided DamageFormer based on the official BRIGHT baseline."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, swin_transformer: nn.Module, depths=None):
        super().__init__()
        # DamageFormer only uses Swin features.
        # The ImageNet classification norm/head are never used by forward.
        # Remove their trainable parameters to avoid permanent unused
        # parameters under DistributedDataParallel.
        swin_transformer.norm = nn.Identity()
        swin_transformer.head = nn.Identity()

        self.swin_transformer = swin_transformer
        self.depths = depths

    def forward(self, x: torch.Tensor):
        feature_maps = []
        for layer in self.swin_transformer.features:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                feature_maps.append(x)
        return feature_maps


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class DamageFormer_BG(nn.Module):
    """Official DamageFormer + minimal building-prior soft ROI guidance."""

    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
    ):
        super().__init__()
        if not 0.0 <= roi_alpha <= 1.0:
            raise ValueError(f"roi_alpha must be in [0, 1], got {roi_alpha}")

        weights = Swin_T_Weights.DEFAULT if use_pretrained_swin else None
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_t(weights=weights), depths=[2, 2, 6, 2])
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_t(weights=weights), depths=[2, 2, 6, 2])

        self.fusion_layer_1 = ResBlock(1440, 256, downsample=nn.Conv2d(1440, 256, 1))
        self.fusion_layer_2 = ResBlock(1440, 256, downsample=nn.Conv2d(1440, 256, 1))
        self.fusion_layer_3 = ResBlock(512, 256, downsample=nn.Conv2d(512, 256, 1))

        self.clf_1 = nn.Conv2d(256, 2, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)

        self.roi_alpha = float(roi_alpha)
        self.detach_building_prior = bool(detach_building_prior)

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _multi_scale_fuse(f1, f2, f3, f4) -> torch.Tensor:
        size = f1.shape[-2:]
        p4 = F.interpolate(f4, size=size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(f3, size=size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(f2, size=size, mode="bilinear", align_corners=False)
        return torch.cat([p4, p3, p2, f1], dim=1)

    def _encode_and_fuse(self, pre_data: torch.Tensor, post_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, pre1, pre2, pre3, pre4 = self.encoder_1(pre_data)
        _, post1, post2, post3, post4 = self.encoder_2(post_data)

        pre1, pre2, pre3, pre4 = map(self._to_nchw, (pre1, pre2, pre3, pre4))
        post1, post2, post3, post4 = map(self._to_nchw, (post1, post2, post3, post4))

        pre_ms = self._multi_scale_fuse(pre1, pre2, pre3, pre4)
        post_ms = self._multi_scale_fuse(post1, post2, post3, post4)
        final_feat_1 = self.fusion_layer_1(pre_ms)
        feat_2 = self.fusion_layer_2(post_ms)
        return final_feat_1, feat_2

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        final_feat_1, feat_2 = self._encode_and_fuse(pre_data, post_data)

        # Localization branch produces the building prior at feature scale.
        loc_logits_feat = self.clf_1(final_feat_1)
        p_building = torch.softmax(loc_logits_feat, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        # Original RGB-SAR fusion, then minimal soft ROI modulation.
        final_feat_2 = self.fusion_layer_3(torch.cat([final_feat_1, feat_2], dim=1))
        final_feat_2_roi = final_feat_2 * roi_weight

        output_loc = F.interpolate(loc_logits_feat, size=pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_dam = F.interpolate(self.clf_2(final_feat_2_roi), size=post_data.shape[-2:], mode="bilinear", align_corners=False)

        if not return_aux:
            return output_loc, output_dam

        aux: Dict[str, torch.Tensor] = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "final_feat_rgb": final_feat_1,
            "final_feat_sar": feat_2,
            "final_feat_fused": final_feat_2,
            "final_feat_fused_roi": final_feat_2_roi,
        }
        return output_loc, output_dam, aux
