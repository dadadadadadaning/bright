# -*- coding: utf-8 -*-
"""Unified RGB-SAR hierarchical DamageFormer for S5/S6/S7/S8 experiments.

Experiment definitions
----------------------
S5: HCDM
    Four Swin stages independently form RGB-SAR relations using |R-S| and R*S.
    Stage differences are resized, concatenated, fused, and residual-injected.

S6: BG-HCDM
    S5 + detached soft building prior at every stage before stage fusion.

S7: AS-BG-HCDM
    S6 + spatial adaptive softmax weights over D1-D4 instead of fixed concat.

S8: IA-MLA
    S7 + independent Stage3/Stage4 projection features for intact-only weak
    cross-modal alignment. The alignment loss is implemented in the trainer.

The original DamageFormer RGB and SAR aggregation/fusion path is preserved.
The hierarchical branch is residual-injected with learnable gamma_hcdm, which
is normally initialized to zero for stable training and checkpoint transfer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


VALID_VARIANTS = ("S5", "S6", "S7", "S8")


class SwinTransformerFeatureExtractor(nn.Module):
    """Return [patch_embed, stage1, stage2, stage3, stage4] Swin-T outputs.

    torchvision Swin feature tensors use channels-last shape [B, H, W, C].
    """

    def __init__(self, swin_transformer: nn.Module):
        super().__init__()
        self.swin_transformer = swin_transformer

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for layer in self.swin_transformer.features:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                outputs.append(x)
        if len(outputs) != 5:
            raise RuntimeError(
                f"Expected 5 Swin sequential outputs, got {len(outputs)}. "
                "Please check the installed torchvision Swin-T implementation."
            )
        return outputs


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class StageRGBSARRelation(nn.Module):
    """Independent modality projection followed by relation encoding."""

    def __init__(self, in_channels: int, diff_channels: int):
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
        self.relation = nn.Sequential(
            nn.Conv2d(diff_channels * 2, diff_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(diff_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_proj(rgb_feat)
        sar = self.sar_proj(sar_feat)
        relation = torch.cat([torch.abs(rgb - sar), rgb * sar], dim=1)
        return self.relation(relation)


class HierarchicalRGBSARDifference(nn.Module):
    """Variant-controlled S5/S6/S7/S8 hierarchical difference branch."""

    STAGE_CHANNELS = (96, 192, 384, 768)

    def __init__(
        self,
        variant: str,
        diff_channels: int = 64,
        out_channels: int = 256,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        variant = variant.upper()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
        if not (0.0 <= roi_alpha <= 1.0):
            raise ValueError(f"roi_alpha must be in [0,1], got {roi_alpha}")

        self.variant = variant
        self.use_building_prior = variant in ("S6", "S7", "S8")
        self.use_adaptive_scale = variant in ("S7", "S8")
        self.roi_alpha = float(roi_alpha)
        self.detach_building_prior = bool(detach_building_prior)

        self.stage_blocks = nn.ModuleList(
            [StageRGBSARRelation(c, diff_channels) for c in self.STAGE_CHANNELS]
        )

        if self.use_adaptive_scale:
            hidden = max(diff_channels, 32)
            self.scale_gate = nn.Sequential(
                nn.Conv2d(diff_channels * 4, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 4, kernel_size=1),
            )
            fuse_in_channels = diff_channels
        else:
            self.scale_gate = None
            fuse_in_channels = diff_channels * 4

        self.out_proj = nn.Sequential(
            nn.Conv2d(fuse_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma_hcdm = nn.Parameter(torch.tensor(float(gamma_init)))

    def _guide_with_building_prior(
        self, diff: torch.Tensor, building_prob: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_building_prior:
            return diff, None
        if building_prob is None:
            raise ValueError(f"{self.variant} requires building_prob")
        prior = building_prob.detach() if self.detach_building_prior else building_prob
        prior = F.interpolate(
            prior, size=diff.shape[-2:], mode="bilinear", align_corners=False
        ).clamp_(0.0, 1.0)
        soft_roi = self.roi_alpha + (1.0 - self.roi_alpha) * prior
        return diff * soft_roi, prior

    def forward(
        self,
        rgb_stages: List[torch.Tensor],
        sar_stages: List[torch.Tensor],
        target_size: Tuple[int, int],
        building_prob: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(rgb_stages) != 4 or len(sar_stages) != 4:
            raise ValueError("Expected four RGB and four SAR stage feature maps")

        stage_diffs: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}
        for stage_idx, (block, rgb_feat, sar_feat) in enumerate(
            zip(self.stage_blocks, rgb_stages, sar_stages), start=1
        ):
            diff = block(rgb_feat, sar_feat)
            diff, prior = self._guide_with_building_prior(diff, building_prob)
            diagnostics[f"hcdm_stage{stage_idx}_abs_mean"] = diff.detach().abs().mean()
            if prior is not None:
                diagnostics[f"building_prior_stage{stage_idx}_mean"] = prior.detach().mean()
            if diff.shape[-2:] != target_size:
                diff = F.interpolate(
                    diff, size=target_size, mode="bilinear", align_corners=False
                )
            stage_diffs.append(diff)

        diff_cat = torch.cat(stage_diffs, dim=1)
        if self.use_adaptive_scale:
            assert self.scale_gate is not None
            scale_weights = torch.softmax(self.scale_gate(diff_cat), dim=1)
            fused = torch.zeros_like(stage_diffs[0])
            for idx, diff in enumerate(stage_diffs):
                fused = fused + scale_weights[:, idx : idx + 1] * diff
        else:
            # Uniform weights are returned only for diagnostics. S5/S6 use fixed
            # concatenation, not the weighted tensor below.
            b, _, h, w = stage_diffs[0].shape
            scale_weights = stage_diffs[0].new_full((b, 4, h, w), 0.25)
            fused = diff_cat

        hierarchical_diff = self.out_proj(fused)
        diagnostics["scale_weights"] = scale_weights
        for idx in range(4):
            diagnostics[f"scale_w{idx + 1}_mean"] = (
                scale_weights[:, idx : idx + 1].detach().mean()
            )
        diagnostics["hcdm_fused_abs_mean"] = hierarchical_diff.detach().abs().mean()
        diagnostics["gamma_hcdm"] = self.gamma_hcdm.detach()
        return hierarchical_diff, diagnostics


class AlignmentProjection(nn.Module):
    """Separate RGB and SAR projections for S8 intact-only alignment."""

    def __init__(self, in_channels: int, align_dim: int):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, align_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True),
        )
        self.sar_proj = nn.Sequential(
            nn.Conv2d(in_channels, align_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_proj(rgb), self.sar_proj(sar)


class DamageFormer_Hierarchical_RGBSAR(nn.Module):
    """One model class for the clean S5 -> S6 -> S7 -> S8 ablation chain."""

    def __init__(
        self,
        variant: str = "S5",
        diff_channels: int = 64,
        gamma_init: float = 0.0,
        roi_alpha: float = 0.1,
        align_dim: int = 128,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        return_dict: bool = False,
    ):
        super().__init__()
        variant = variant.upper()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.return_dict = bool(return_dict)

        weights = Swin_T_Weights.DEFAULT if use_pretrained_swin else None
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_t(weights=weights))
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_t(weights=weights))

        # Original DamageFormer path.
        self.fusion_layer_1 = ResBlock(1440, 256)
        self.fusion_layer_2 = ResBlock(1440, 256)
        self.fusion_layer_3 = ResBlock(512, 256)
        self.clf_1 = nn.Conv2d(256, 2, kernel_size=1)
        self.clf_2 = nn.Conv2d(256, 4, kernel_size=1)

        self.hcdm = HierarchicalRGBSARDifference(
            variant=variant,
            diff_channels=diff_channels,
            out_channels=256,
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            gamma_init=gamma_init,
        )

        if variant == "S8":
            self.ia_stage3: Optional[AlignmentProjection] = AlignmentProjection(384, align_dim)
            self.ia_stage4: Optional[AlignmentProjection] = AlignmentProjection(768, align_dim)
        else:
            self.ia_stage3 = None
            self.ia_stage4 = None

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _aggregate_swin_stages(stages: List[torch.Tensor], fusion: nn.Module) -> torch.Tensor:
        if len(stages) != 4:
            raise ValueError(f"Expected four Swin stages, got {len(stages)}")
        target_size = stages[0].shape[-2:]
        aligned = [stages[0]]
        for feat in stages[1:]:
            aligned.append(
                F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            )
        # Keep official DamageFormer order: stage4, stage3, stage2, stage1.
        return fusion(torch.cat([aligned[3], aligned[2], aligned[1], aligned[0]], dim=1))

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        return_dict: Optional[bool] = None,
    ):
        pre_maps = self.encoder_1(pre_data)
        post_maps = self.encoder_2(post_data)
        rgb_stages = [self._to_nchw(x) for x in pre_maps[1:5]]
        sar_stages = [self._to_nchw(x) for x in post_maps[1:5]]

        final_feat_rgb = self._aggregate_swin_stages(rgb_stages, self.fusion_layer_1)
        final_feat_sar = self._aggregate_swin_stages(sar_stages, self.fusion_layer_2)

        loc_logits_low = self.clf_1(final_feat_rgb)
        building_prob = torch.softmax(loc_logits_low, dim=1)[:, 1:2]

        final_feat_damage_base = self.fusion_layer_3(
            torch.cat([final_feat_rgb, final_feat_sar], dim=1)
        )
        hcdm_feat, diagnostics = self.hcdm(
            rgb_stages=rgb_stages,
            sar_stages=sar_stages,
            target_size=final_feat_damage_base.shape[-2:],
            building_prob=building_prob,
        )
        final_feat_damage = (
            final_feat_damage_base + self.hcdm.gamma_hcdm * hcdm_feat
        )

        dam_logits_low = self.clf_2(final_feat_damage)
        output_loc = F.interpolate(
            loc_logits_low, size=pre_data.shape[-2:], mode="bilinear", align_corners=False
        )
        output_dam = F.interpolate(
            dam_logits_low, size=post_data.shape[-2:], mode="bilinear", align_corners=False
        )

        use_dict = self.return_dict if return_dict is None else bool(return_dict)
        if not use_dict:
            return output_loc, output_dam

        outputs: Dict[str, torch.Tensor] = {
            "output_loc": output_loc,
            "output_dam": output_dam,
            "main_dam_logits": output_dam,
            "building_prob": building_prob,
            "final_feat_rgb": final_feat_rgb,
            "final_feat_sar": final_feat_sar,
            "final_feat_damage_base": final_feat_damage_base,
            "final_feat_damage": final_feat_damage,
            "hcdm_feat": hcdm_feat,
            **diagnostics,
        }

        if self.variant == "S8":
            assert self.ia_stage3 is not None and self.ia_stage4 is not None
            ia_rgb_s3, ia_sar_s3 = self.ia_stage3(rgb_stages[2], sar_stages[2])
            ia_rgb_s4, ia_sar_s4 = self.ia_stage4(rgb_stages[3], sar_stages[3])
            outputs.update(
                {
                    "ia_rgb_s3": ia_rgb_s3,
                    "ia_sar_s3": ia_sar_s3,
                    "ia_rgb_s4": ia_rgb_s4,
                    "ia_sar_s4": ia_sar_s4,
                }
            )
        return outputs


# Compatibility aliases.
DamageFormer = DamageFormer_Hierarchical_RGBSAR
DamageFormer_S5_HCDM = DamageFormer_Hierarchical_RGBSAR
DamageFormer_S6_BG_HCDM = DamageFormer_Hierarchical_RGBSAR
DamageFormer_S7_AS_BG_HCDM = DamageFormer_Hierarchical_RGBSAR
DamageFormer_S8_IA_MLA = DamageFormer_Hierarchical_RGBSAR


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    for variant in VALID_VARIANTS:
        model = DamageFormer_Hierarchical_RGBSAR(
            variant=variant,
            use_pretrained_swin=False,
            return_dict=True,
        ).eval()
        pre = torch.randn(1, 3, 128, 128)
        post = torch.randn(1, 3, 128, 128)
        out = model(pre, post)
        print(
            variant,
            tuple(out["output_loc"].shape),
            tuple(out["output_dam"].shape),
            float(out["gamma_hcdm"].item()),
        )
        if variant == "S8":
            print("  IA:", tuple(out["ia_rgb_s3"].shape), tuple(out["ia_rgb_s4"].shape))
