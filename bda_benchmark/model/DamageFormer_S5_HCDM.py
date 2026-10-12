# -*- coding: utf-8 -*-
"""S5: DamageFormer + Hierarchical Cross-modal Damage Difference Mining (HCDM).

Experiment definition
---------------------
S5 = official DamageFormer + HCDM only.

No building ROI, no prototype memory, no auxiliary damage head, and no
cross-modal alignment loss are included in this model.  The purpose is to test
one hypothesis cleanly: whether explicit RGB-SAR interaction at all four Swin
stages is better than waiting until the final 256-d feature to concatenate the
modalities.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class SwinTransformerFeatureExtractor(nn.Module):
    """Return the five Sequential outputs from torchvision Swin-T features.

    The outputs are:
      [patch_embed, stage1, stage2, stage3, stage4]
    and use channels-last layout [B, H, W, C].
    """

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
                f"Expected 5 Swin sequential feature maps, got {len(feature_maps)}. "
                "Check the installed torchvision Swin-T implementation."
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
    """Project paired RGB/SAR features and encode explicit cross-modal relations."""

    def __init__(self, in_channels: int, diff_channels: int = 64):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, diff_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )
        self.sar_proj = nn.Sequential(
            nn.Conv2d(in_channels, diff_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )
        # S5 deliberately uses only |RGB-SAR| and RGB*SAR.  This keeps the first
        # hierarchical-difference ablation compact and easier to interpret.
        self.relation_fuse = nn.Sequential(
            nn.Conv2d(diff_channels * 2, diff_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_proj(rgb_feat)
        sar = self.sar_proj(sar_feat)
        diff_abs = torch.abs(rgb - sar)
        interaction = rgb * sar
        return self.relation_fuse(torch.cat([diff_abs, interaction], dim=1))


class HierarchicalCrossModalDifferenceMining(nn.Module):
    """Mine RGB-SAR relations independently at four Swin hierarchy levels."""

    STAGE_CHANNELS = (96, 192, 384, 768)

    def __init__(self, diff_channels: int = 64, out_channels: int = 256, gamma_init: float = 0.0):
        super().__init__()
        self.stage_blocks = nn.ModuleList(
            [StageDifferenceBlock(c, diff_channels) for c in self.STAGE_CHANNELS]
        )
        self.hierarchical_fuse = nn.Sequential(
            nn.Conv2d(diff_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Zero-init makes the initial network exactly follow the original
        # DamageFormer fusion path. The model must learn to use HCDM.
        self.gamma_hcdm = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        rgb_stages: List[torch.Tensor],
        sar_stages: List[torch.Tensor],
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(rgb_stages) != 4 or len(sar_stages) != 4:
            raise ValueError("HCDM expects four RGB and four SAR stage features.")

        stage_diffs: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for idx, (block, rgb, sar) in enumerate(
            zip(self.stage_blocks, rgb_stages, sar_stages), start=1
        ):
            d = block(rgb, sar)
            diagnostics[f"hcdm_stage{idx}_abs_mean"] = d.detach().abs().mean()
            if d.shape[-2:] != target_size:
                d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
            stage_diffs.append(d)

        diff_cat = torch.cat(stage_diffs, dim=1)
        hierarchical_diff = self.hierarchical_fuse(diff_cat)
        diagnostics["hcdm_fused_abs_mean"] = hierarchical_diff.detach().abs().mean()
        diagnostics["gamma_hcdm"] = self.gamma_hcdm.detach()
        return hierarchical_diff, diagnostics


class DamageFormer_S5_HCDM(nn.Module):
    """Official DamageFormer backbone with one clean HCDM residual branch."""

    def __init__(
        self,
        diff_channels: int = 64,
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

        self.hcdm = HierarchicalCrossModalDifferenceMining(
            diff_channels=diff_channels,
            out_channels=256,
            gamma_init=gamma_init,
        )

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _aggregate_swin_stages(stages: List[torch.Tensor], fusion: nn.Module) -> torch.Tensor:
        target_size = stages[0].shape[-2:]
        aligned = [stages[0]]
        for feat in stages[1:]:
            aligned.append(F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False))
        # Preserve official DamageFormer channel order: stage4, stage3, stage2, stage1.
        feat_cat = torch.cat([aligned[3], aligned[2], aligned[1], aligned[0]], dim=1)
        return fusion(feat_cat)

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

        final_feat_2_base = self.fusion_layer_3(
            torch.cat([final_feat_1, final_feat_sar], dim=1)
        )
        hcdm_feat, diagnostics = self.hcdm(
            rgb_stages,
            sar_stages,
            target_size=final_feat_2_base.shape[-2:],
        )
        final_feat_2 = final_feat_2_base + self.hcdm.gamma_hcdm * hcdm_feat

        output_loc_feat = self.clf_1(final_feat_1)
        output_dam_feat = self.clf_2(final_feat_2)
        output_loc = F.interpolate(
            output_loc_feat, size=pre_data.shape[-2:], mode="bilinear", align_corners=False
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
            "final_feat_rgb": final_feat_1,
            "final_feat_sar": final_feat_sar,
            "final_feat_damage_base": final_feat_2_base,
            "final_feat_damage": final_feat_2,
            "hcdm_feat": hcdm_feat,
            **diagnostics,
        }


# Compatibility alias for scripts that expect a generic model name.
DamageFormer = DamageFormer_S5_HCDM


if __name__ == "__main__":
    model = DamageFormer_S5_HCDM(use_pretrained_swin=False, return_dict=True).eval()
    with torch.no_grad():
        x1 = torch.randn(1, 3, 128, 128)
        x2 = torch.randn(1, 3, 128, 128)
        out = model(x1, x2)
    print("output_loc:", tuple(out["output_loc"].shape))
    print("output_dam:", tuple(out["output_dam"].shape))
    print("gamma_hcdm:", float(out["gamma_hcdm"].item()))
