# -*- coding: utf-8 -*-
"""S1-C: Pre-fusion dual-branch residual soft building guidance.

The online soft building prior P_B is predicted from the pre-event optical
representation and detached.  The same spatial prior enhances both modality
representations before RGB-SAR fusion:

    F_pre'  = F_pre  * (1 + beta * stopgrad(P_B))
    F_post' = F_post * (1 + beta * stopgrad(P_B))
    F_dam   = Fusion([F_pre', F_post'])

For beta=1, the guidance weight lies in [1, 2].
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, swin_transformer: nn.Module, depths=None):
        super().__init__()
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


class DamageFormer_S1C(nn.Module):
    """DamageFormer + pre-fusion dual-branch residual soft building guidance."""

    def __init__(
        self,
        guidance_beta: float = 1.0,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
    ):
        super().__init__()
        if guidance_beta < 0.0:
            raise ValueError(f"guidance_beta must be >= 0, got {guidance_beta}")

        weights = Swin_T_Weights.DEFAULT if use_pretrained_swin else None
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_t(weights=weights), depths=[2, 2, 6, 2])
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_t(weights=weights), depths=[2, 2, 6, 2])

        self.fusion_layer_1 = ResBlock(1440, 256, downsample=nn.Conv2d(1440, 256, 1))
        self.fusion_layer_2 = ResBlock(1440, 256, downsample=nn.Conv2d(1440, 256, 1))
        self.fusion_layer_3 = ResBlock(512, 256, downsample=nn.Conv2d(512, 256, 1))

        self.clf_1 = nn.Conv2d(256, 2, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)

        self.guidance_beta = float(guidance_beta)
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

        # Online internal building prior from the pre-event optical branch.
        loc_logits_feat = self.clf_1(final_feat_1)
        p_building = torch.softmax(loc_logits_feat, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        guidance_weight = 1.0 + self.guidance_beta * building_prior

        # S1-C key: enhance both modality representations before fusion.
        final_feat_1_guided = final_feat_1 * guidance_weight
        feat_2_guided = feat_2 * guidance_weight
        final_feat_2 = self.fusion_layer_3(torch.cat([final_feat_1_guided, feat_2_guided], dim=1))

        output_loc = F.interpolate(loc_logits_feat, size=pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_dam = F.interpolate(self.clf_2(final_feat_2), size=post_data.shape[-2:], mode="bilinear", align_corners=False)

        if not return_aux:
            return output_loc, output_dam

        aux: Dict[str, torch.Tensor] = {
            "p_building": p_building,
            "guidance_weight": guidance_weight,
            "roi_weight": guidance_weight,  # compatibility alias
            "final_feat_rgb": final_feat_1,
            "final_feat_sar": feat_2,
            "final_feat_rgb_guided": final_feat_1_guided,
            "final_feat_sar_guided": feat_2_guided,
            "final_feat_fused": final_feat_2,
        }
        return output_loc, output_dam, aux
