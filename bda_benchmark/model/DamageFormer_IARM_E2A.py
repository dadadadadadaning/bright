# -*- coding: utf-8 -*-
"""IARM E2-A model for BRIGHT.

E2-A is a representation diagnostic:
- learn Intact vs Non-Intact from the intact-anchored residual;
- do not inject residual features into the four-class damage classifier;
- preserve the E1/main segmentation path exactly;
- keep residual direction and residual magnitude for Damage-Exist.

The parent IARM model normalizes the residual embedding for metric learning.
Pure normalization removes magnitude information. E2-A therefore feeds:

    [normalized residual direction, log(1 + residual norm)]

to the Damage-Exist head.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_IARM import (
        ConvGNAct,
        DamageFormer_IARM,
        Tensor,
    )
except ImportError:  # pragma: no cover
    from DamageFormer_IARM import (
        ConvGNAct,
        DamageFormer_IARM,
        Tensor,
    )


class DamageFormer_IARM_E2A(DamageFormer_IARM):
    """Exist-only IARM ablation with the four-class path frozen externally."""

    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        residual_stages: Sequence[int] = (4,),
        residual_dim: int = 128,
        flow_hidden_dim: int = 64,
        max_offset_pixels: float = 2.0,
        detach_backbone_for_residual: bool = True,
        residual_gamma_max: float = 0.25,
        num_events: int = 14,
        gaussian_step_momentum: float = 0.99,
        gaussian_variance_floor: float = 2.0e-3,
        max_pixels_per_image_class: int = 256,
    ) -> None:
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
            residual_stages=residual_stages,
            residual_dim=residual_dim,
            flow_hidden_dim=flow_hidden_dim,
            max_offset_pixels=max_offset_pixels,
            detach_backbone_for_residual=detach_backbone_for_residual,
            enable_residual_fusion=False,
            residual_gamma_max=residual_gamma_max,
            num_events=num_events,
            gaussian_step_momentum=gaussian_step_momentum,
            gaussian_variance_floor=gaussian_variance_floor,
            max_pixels_per_image_class=max_pixels_per_image_class,
        )

        # The E1 head consumed only residual_dim normalized channels.
        # E2-A adds one log-norm channel and deliberately reinitializes
        # the whole head when warm-starting from E1.
        self.damage_exist_head = nn.Sequential(
            ConvGNAct(
                self.residual_dim + 1,
                self.residual_dim,
                kernel_size=3,
            ),
            nn.Conv2d(
                self.residual_dim,
                2,
                kernel_size=1,
            ),
        )

    def _build_residual(
        self,
        pre_stages: Dict[int, Tensor],
        post_stages: Dict[int, Tensor],
        output_size: Tuple[int, int],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        residuals: List[Tensor] = []
        aux: Dict[str, Tensor] = {}

        for stage in self.residual_stages:
            key = str(stage)
            pre_feature = pre_stages[stage]
            post_feature = post_stages[stage]

            if self.detach_backbone_for_residual:
                pre_feature = pre_feature.detach()
                post_feature = post_feature.detach()

            optical = self.optical_projectors[key](pre_feature)
            sar = self.sar_projectors[key](post_feature)
            aligned_sar, flow = self.aligners[key](optical, sar)
            predicted_normal = self.normal_mappers[key](optical)
            residual = self.residual_encoders[key](
                optical,
                aligned_sar,
                predicted_normal,
            )

            residuals.append(
                F.interpolate(
                    residual,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

            aux[f"optical_metric_s{stage}"] = optical
            aux[f"sar_metric_s{stage}"] = sar
            aux[f"aligned_sar_metric_s{stage}"] = aligned_sar
            aux[f"normal_prediction_s{stage}"] = predicted_normal
            aux[f"flow_s{stage}"] = flow
            aux[f"residual_raw_s{stage}"] = residual

        residual_fused_raw = self.residual_fuser(
            torch.cat(residuals, dim=1)
        )
        metric_embedding = F.normalize(
            residual_fused_raw,
            dim=1,
            eps=1.0e-6,
        )
        residual_norm = torch.linalg.vector_norm(
            residual_fused_raw,
            ord=2,
            dim=1,
            keepdim=True,
        )
        residual_log_norm = torch.log1p(
            residual_norm.clamp_min(0.0)
        )

        aux["residual_fused_raw"] = residual_fused_raw
        aux["metric_embedding"] = metric_embedding
        aux["residual_norm"] = residual_norm
        aux["residual_log_norm"] = residual_log_norm
        return metric_embedding, aux

    def forward(
        self,
        pre_data: Tensor,
        post_data: Tensor,
        damage_target: Optional[Tensor] = None,
        event_ids: Optional[Tensor] = None,
        current_iter: Optional[int] = None,
        return_aux: bool = False,
        force_residual_off: bool = False,
    ):
        del current_iter
        del force_residual_off

        (
            pre_stages,
            post_stages,
            final_feat_rgb,
            final_feat_sar,
        ) = self._encode_all(pre_data, post_data)

        loc_logits_feat = self.clf_1(final_feat_rgb)
        p_building = torch.softmax(
            loc_logits_feat,
            dim=1,
        )[:, 1:2]
        building_prior = (
            p_building.detach()
            if self.detach_building_prior
            else p_building
        )
        roi_weight = (
            self.roi_alpha
            + (1.0 - self.roi_alpha) * building_prior
        )

        final_feat_fused = self.fusion_layer_3(
            torch.cat(
                [final_feat_rgb, final_feat_sar],
                dim=1,
            )
        )
        final_feat_fused_roi = final_feat_fused * roi_weight

        metric_embedding, residual_aux = self._build_residual(
            pre_stages,
            post_stages,
            output_size=final_feat_fused_roi.shape[-2:],
        )

        exist_input = torch.cat(
            [
                metric_embedding,
                residual_aux["residual_log_norm"],
            ],
            dim=1,
        )
        damage_exist_logits_feat = self.damage_exist_head(
            exist_input
        )

        # E2-A never changes the four-class damage path.
        damage_feature = final_feat_fused_roi
        damage_logits_feat = self.clf_2(damage_feature)
        gamma = self.residual_gamma()

        # E2-A does not train or update Gaussian/class-distribution
        # memory. Keeping these buffers fixed avoids unrelated state drift.
        del damage_target
        del event_ids

        anchor_logits_feat = (
            self.class_distribution_memory.query(
                metric_embedding
            )
        )

        output_loc = F.interpolate(
            loc_logits_feat,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_damage = F.interpolate(
            damage_logits_feat,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_exist = F.interpolate(
            damage_exist_logits_feat,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if not return_aux:
            return output_loc, output_damage

        # residual_to_damage is kept for checkpoint compatibility only.
        # It is frozen by the E2-A trainer and is not used in damage_feature.
        residual_feature = self.residual_to_damage(
            metric_embedding
        ) * roi_weight

        aux: Dict[str, Tensor] = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "final_feat_rgb": final_feat_rgb,
            "final_feat_sar": final_feat_sar,
            "final_feat_fused": final_feat_fused,
            "final_feat_fused_roi": final_feat_fused_roi,
            "residual_feature": residual_feature,
            "damage_feature": damage_feature,
            "main_damage_logits_feat": damage_logits_feat,
            "damage_logits_feat": damage_logits_feat,
            "damage_exist_logits_feat": (
                damage_exist_logits_feat
            ),
            "exist_input": exist_input,
            "anchor_logits_feat": anchor_logits_feat,
            "anchor_ready": torch.tensor(
                self.class_distribution_memory.ready(),
                device=damage_logits_feat.device,
                dtype=torch.bool,
            ),
            "residual_gamma": gamma,
            "residual_applied": torch.tensor(
                False,
                device=damage_logits_feat.device,
                dtype=torch.bool,
            ),
        }
        aux.update(residual_aux)
        aux.update(
            self.class_distribution_memory.diagnostics()
        )
        return (
            output_loc,
            output_damage,
            output_exist,
            aux,
        )


__all__ = ["DamageFormer_IARM_E2A"]
