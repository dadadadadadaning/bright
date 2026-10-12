# -*- coding: utf-8 -*-
"""S8: BG-AS-HCDM + Intact-Anchored Multi-Level Alignment (IA-MLA).

Experiment definition
---------------------
S8 contains the cumulative structural path defined in the experiment plan:
  S5 HCDM
  + building-prior guided difference mining
  + adaptive scale-aware difference fusion
  + intact-only Stage3/Stage4 cross-modal alignment.

The alignment loss itself is intentionally implemented in the training script,
because it needs GT damage labels and a warmup/ramp schedule.  The model returns
Stage3/Stage4 projected features for this purpose.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, swin_transformer: nn.Module):
        super().__init__()
        self.swin_transformer = swin_transformer

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_maps: List[torch.Tensor] = []
        for layer in self.swin_transformer.features:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                feature_maps.append(x)
        if len(feature_maps) != 5:
            raise RuntimeError(
                f"Expected 5 Swin sequential feature maps, got {len(feature_maps)}."
            )
        return feature_maps


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class StageDifferenceBlock(nn.Module):
    def __init__(self, in_channels: int, diff_channels: int = 64):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, diff_channels, 1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )
        self.sar_proj = nn.Sequential(
            nn.Conv2d(in_channels, diff_channels, 1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )
        self.relation_fuse = nn.Sequential(
            nn.Conv2d(diff_channels * 2, diff_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_proj(rgb_feat)
        sar = self.sar_proj(sar_feat)
        return self.relation_fuse(
            torch.cat([torch.abs(rgb - sar), rgb * sar], dim=1)
        )


class BuildingGuidedAdaptiveHCDM(nn.Module):
    STAGE_CHANNELS = (96, 192, 384, 768)

    def __init__(
        self,
        diff_channels: int = 64,
        out_channels: int = 256,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.roi_alpha = float(roi_alpha)
        self.detach_building_prior = bool(detach_building_prior)
        self.stage_blocks = nn.ModuleList(
            [StageDifferenceBlock(c, diff_channels) for c in self.STAGE_CHANNELS]
        )
        hidden = max(diff_channels, 32)
        self.scale_gate = nn.Sequential(
            nn.Conv2d(diff_channels * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(diff_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma_hcdm = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        rgb_stages: List[torch.Tensor],
        sar_stages: List[torch.Tensor],
        building_prob: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(rgb_stages) != 4 or len(sar_stages) != 4:
            raise ValueError("BG-AS-HCDM expects four RGB and four SAR stage features.")
        prior = building_prob.detach() if self.detach_building_prior else building_prob

        guided_diffs: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for idx, (block, rgb, sar) in enumerate(
            zip(self.stage_blocks, rgb_stages, sar_stages), start=1
        ):
            d = block(rgb, sar)
            p = F.interpolate(
                prior, size=d.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            soft_roi = self.roi_alpha + (1.0 - self.roi_alpha) * p
            d = d * soft_roi
            diagnostics[f"hcdm_stage{idx}_abs_mean"] = d.detach().abs().mean()
            diagnostics[f"building_prior_stage{idx}_mean"] = p.detach().mean()
            if d.shape[-2:] != target_size:
                d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
            guided_diffs.append(d)

        diff_cat = torch.cat(guided_diffs, dim=1)
        scale_weights = torch.softmax(self.scale_gate(diff_cat), dim=1)
        weighted = sum(
            guided_diffs[i] * scale_weights[:, i : i + 1]
            for i in range(4)
        )
        hierarchical_diff = self.out_proj(weighted)

        diagnostics["scale_weights"] = scale_weights
        for i in range(4):
            diagnostics[f"scale_w{i+1}_mean"] = scale_weights[:, i : i + 1].detach().mean()
        diagnostics["hcdm_fused_abs_mean"] = hierarchical_diff.detach().abs().mean()
        diagnostics["gamma_hcdm"] = self.gamma_hcdm.detach()
        return hierarchical_diff, diagnostics


class AlignmentProjection(nn.Module):
    """Independent modality projections for high-level semantic alignment."""

    def __init__(self, in_channels: int, align_dim: int = 128):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, align_dim, 1, bias=False),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True),
        )
        self.sar_proj = nn.Sequential(
            nn.Conv2d(in_channels, align_dim, 1, bias=False),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_proj(rgb), self.sar_proj(sar)


class DamageFormer_S8_IA_MLA(nn.Module):
    def __init__(
        self,
        diff_channels: int = 64,
        align_dim: int = 128,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        gamma_init: float = 0.0,
        use_pretrained_swin: bool = True,
        return_dict: bool = False,
    ):
        super().__init__()
        self.return_dict = bool(return_dict)
        weights = Swin_T_Weights.DEFAULT if use_pretrained_swin else None
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_t(weights=weights))
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_t(weights=weights))

        self.fusion_layer_1 = ResBlock(1440, 256)
        self.fusion_layer_2 = ResBlock(1440, 256)
        self.fusion_layer_3 = ResBlock(512, 256)
        self.clf_1 = nn.Conv2d(256, 2, kernel_size=1)
        self.clf_2 = nn.Conv2d(256, 4, kernel_size=1)

        self.hcdm = BuildingGuidedAdaptiveHCDM(
            diff_channels=diff_channels,
            out_channels=256,
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            gamma_init=gamma_init,
        )
        # Stage3=384 channels, Stage4=768 channels.
        self.ia_stage3 = AlignmentProjection(384, align_dim)
        self.ia_stage4 = AlignmentProjection(768, align_dim)

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _aggregate_swin_stages(stages: List[torch.Tensor], fusion: nn.Module) -> torch.Tensor:
        target_size = stages[0].shape[-2:]
        aligned = [stages[0]]
        for feat in stages[1:]:
            aligned.append(F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False))
        return fusion(torch.cat([aligned[3], aligned[2], aligned[1], aligned[0]], dim=1))

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        return_dict: bool | None = None,
    ):
        pre_maps = self.encoder_1(pre_data)
        post_maps = self.encoder_2(post_data)
        rgb_stages = [self._to_nchw(x) for x in pre_maps[1:5]]
        sar_stages = [self._to_nchw(x) for x in post_maps[1:5]]

        final_feat_1 = self._aggregate_swin_stages(rgb_stages, self.fusion_layer_1)
        final_feat_sar = self._aggregate_swin_stages(sar_stages, self.fusion_layer_2)

        loc_logits_feat = self.clf_1(final_feat_1)
        building_prob = torch.softmax(loc_logits_feat, dim=1)[:, 1:2]

        final_feat_2_base = self.fusion_layer_3(
            torch.cat([final_feat_1, final_feat_sar], dim=1)
        )
        hcdm_feat, diagnostics = self.hcdm(
            rgb_stages,
            sar_stages,
            building_prob=building_prob,
            target_size=final_feat_2_base.shape[-2:],
        )
        final_feat_2 = final_feat_2_base + self.hcdm.gamma_hcdm * hcdm_feat

        ia_rgb_s3, ia_sar_s3 = self.ia_stage3(rgb_stages[2], sar_stages[2])
        ia_rgb_s4, ia_sar_s4 = self.ia_stage4(rgb_stages[3], sar_stages[3])

        output_dam_feat = self.clf_2(final_feat_2)
        output_loc = F.interpolate(
            loc_logits_feat, size=pre_data.shape[-2:], mode="bilinear", align_corners=False
        )
        output_dam = F.interpolate(
            output_dam_feat, size=post_data.shape[-2:], mode="bilinear", align_corners=False
        )

        use_dict = self.return_dict if return_dict is None else bool(return_dict)
        if not use_dict:
            return output_loc, output_dam
        return {
            "output_loc": output_loc,
            "output_dam": output_dam,
            "main_dam_logits": output_dam,
            "building_prob": building_prob,
            "final_feat_rgb": final_feat_1,
            "final_feat_sar": final_feat_sar,
            "final_feat_damage_base": final_feat_2_base,
            "final_feat_damage": final_feat_2,
            "hcdm_feat": hcdm_feat,
            "ia_rgb_s3": ia_rgb_s3,
            "ia_sar_s3": ia_sar_s3,
            "ia_rgb_s4": ia_rgb_s4,
            "ia_sar_s4": ia_sar_s4,
            **diagnostics,
        }


DamageFormer = DamageFormer_S8_IA_MLA


if __name__ == "__main__":
    model = DamageFormer_S8_IA_MLA(use_pretrained_swin=False, return_dict=True).eval()
    with torch.no_grad():
        x1 = torch.randn(1, 3, 128, 128)
        x2 = torch.randn(1, 3, 128, 128)
        out = model(x1, x2)
    for key in ["output_loc", "output_dam", "ia_rgb_s3", "ia_rgb_s4", "scale_weights"]:
        print(key, tuple(out[key].shape))
    print("gamma_hcdm:", float(out["gamma_hcdm"].item()))
