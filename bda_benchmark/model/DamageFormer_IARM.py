# -*- coding: utf-8 -*-
"""IARM-DamageFormer: intact-anchored residual metric learning for BRIGHT.

This module is a clean extension of the official DamageFormer/BG baseline.
It intentionally does *not* use the old prototype-residual logit correction.
Instead it implements four controlled ideas:

1. Lightweight feature-flow alignment at Swin Stage 3/4.
2. Cross-modal unification (paired optical/SAR metric projections).
3. Intact-anchored normal relation modelling and residual embeddings.
4. Event-balanced Gaussian class anchors used for metric regularization and
   retrieval routing, not as a dominant replacement for the main classifier.

The final instance memory is constructed offline from TRAIN features by the
companion scripts in ``memory/``. This separation prevents validation/test
labels from entering the memory bank and keeps the training path auditable.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_BG import DamageFormer_BG
except ImportError:  # pragma: no cover - supports direct script imports
    from DamageFormer_BG import DamageFormer_BG


Tensor = torch.Tensor


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 16,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        if out_channels % groups != 0:
            # Keep GroupNorm valid for small experimental dimensions.
            valid_groups = [g for g in range(min(groups, out_channels), 0, -1) if out_channels % g == 0]
            groups = valid_groups[0]
        modules: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
        ]
        if activation:
            modules.append(nn.GELU())
        super().__init__(*modules)


class FeatureProjector(nn.Module):
    """Two-layer projection head, following supervised contrastive segmentation.

    The output is not normalized here because some auxiliary heads consume the
    dense feature. Callers normalize it only for metric comparisons.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LightweightFlowAlign(nn.Module):
    """Estimate a bounded 2-D feature displacement and warp the SAR feature.

    The last offset convolution is zero-initialized, so the module starts as an
    identity warp. Offsets are expressed in feature pixels and converted to a
    normalized grid for ``grid_sample``.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        max_offset_pixels: float = 2.0,
    ) -> None:
        super().__init__()
        if max_offset_pixels < 0:
            raise ValueError("max_offset_pixels must be non-negative")
        self.max_offset_pixels = float(max_offset_pixels)
        self.encoder = nn.Sequential(
            ConvGNAct(2 * channels, hidden_channels, kernel_size=3),
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, device, dtype) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, optical: Tensor, sar: Tensor) -> Tuple[Tensor, Tensor]:
        if optical.shape != sar.shape:
            raise ValueError(
                f"Alignment inputs must share shape, got {tuple(optical.shape)} and {tuple(sar.shape)}"
            )
        batch, _, height, width = optical.shape
        offset_pixels = torch.tanh(self.encoder(torch.cat([optical, sar], dim=1)))
        offset_pixels = offset_pixels * self.max_offset_pixels

        # grid_sample expects offsets in [-1, 1] coordinates.
        norm_x = 2.0 / max(width - 1, 1)
        norm_y = 2.0 / max(height - 1, 1)
        offset_grid = torch.stack(
            [offset_pixels[:, 0] * norm_x, offset_pixels[:, 1] * norm_y],
            dim=-1,
        )
        grid = self._base_grid(batch, height, width, sar.device, sar.dtype) + offset_grid
        warped = F.grid_sample(
            sar,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped, offset_pixels


class NormalRelationMapper(nn.Module):
    """Predict the normal post-disaster SAR embedding from pre-disaster optical."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(channels, channels, kernel_size=3),
            ConvGNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, optical_embedding: Tensor) -> Tensor:
        return self.net(optical_embedding)


class ResidualEncoder(nn.Module):
    """Encode signed, absolute, and multiplicative normality deviations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(3 * channels, channels, kernel_size=3),
            ConvGNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(
        self,
        optical_embedding: Tensor,
        aligned_sar_embedding: Tensor,
        predicted_normal_sar: Tensor,
    ) -> Tensor:
        signed = aligned_sar_embedding - predicted_normal_sar
        absolute = signed.abs()
        interaction = optical_embedding * aligned_sar_embedding
        return self.net(torch.cat([signed, absolute, interaction], dim=1))


class EventBalancedDiagonalGaussianMemory(nn.Module):
    """One diagonal Gaussian per damage class, updated with equal-event means.

    Class indices in this memory are 0=Intact, 1=Damaged, 2=Destroyed.
    It is deliberately a coarse distributional router. Detailed variation is
    retained by the offline instance memory rather than compressed into this
    module.
    """

    NUM_CLASSES = 3

    def __init__(
        self,
        embedding_dim: int,
        num_events: int = 14,
        momentum: float = 0.99,
        variance_floor: float = 2.0e-3,
        max_pixels_per_image_class: int = 256,
    ) -> None:
        super().__init__()
        if num_events <= 0:
            raise ValueError("num_events must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        self.embedding_dim = int(embedding_dim)
        self.num_events = int(num_events)
        self.momentum = float(momentum)
        self.variance_floor = float(variance_floor)
        self.max_pixels_per_image_class = int(max_pixels_per_image_class)

        self.register_buffer("means", torch.zeros(self.NUM_CLASSES, embedding_dim))
        self.register_buffer("second_moments", torch.ones(self.NUM_CLASSES, embedding_dim))
        self.register_buffer("initialized", torch.zeros(self.NUM_CLASSES, dtype=torch.bool))
        self.register_buffer("event_updates", torch.zeros(self.NUM_CLASSES))
        self.register_buffer("image_updates", torch.zeros(self.NUM_CLASSES))
        self.register_buffer("pixel_observations", torch.zeros(self.NUM_CLASSES))

    @staticmethod
    def _downsample_target(target: Tensor, size: Tuple[int, int]) -> Tensor:
        if target.dim() == 4:
            target = target[:, 0]
        if target.dim() != 3:
            raise ValueError(f"Expected target [B,H,W], got {tuple(target.shape)}")
        return F.interpolate(
            target.unsqueeze(1).float(),
            size=size,
            mode="nearest",
        )[:, 0].long()

    @staticmethod
    def _all_reduce_sum(x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @torch.no_grad()
    def update(self, embedding: Tensor, target: Tensor, event_ids: Tensor) -> None:
        if event_ids is None:
            raise ValueError("event_ids are required for event-balanced Gaussian updates")
        if event_ids.dim() != 1 or event_ids.shape[0] != embedding.shape[0]:
            raise ValueError("event_ids must be [B]")

        z = F.normalize(embedding.detach(), dim=1)
        target_small = self._downsample_target(target, z.shape[-2:])
        batch_size = z.shape[0]

        for class_idx, label_value in enumerate((1, 2, 3)):
            event_sum = z.new_zeros(self.num_events, self.embedding_dim)
            event_second = z.new_zeros(self.num_events, self.embedding_dim)
            event_image_count = z.new_zeros(self.num_events)
            pixel_count = z.new_zeros(())

            for batch_idx in range(batch_size):
                mask = target_small[batch_idx] == label_value
                count = int(mask.sum().item())
                if count <= 0:
                    continue
                features = z[batch_idx].permute(1, 2, 0)[mask]
                if features.shape[0] > self.max_pixels_per_image_class:
                    choice = torch.randperm(features.shape[0], device=features.device)[
                        : self.max_pixels_per_image_class
                    ]
                    features = features[choice]
                image_mean = features.mean(dim=0)
                image_second = (features * features).mean(dim=0)
                event_id = int(event_ids[batch_idx].item())
                if not 0 <= event_id < self.num_events:
                    raise ValueError(
                        f"event id {event_id} is outside [0,{self.num_events})"
                    )
                event_sum[event_id] += image_mean
                event_second[event_id] += image_second
                event_image_count[event_id] += 1.0
                pixel_count += float(count)

            self._all_reduce_sum(event_sum)
            self._all_reduce_sum(event_second)
            self._all_reduce_sum(event_image_count)
            self._all_reduce_sum(pixel_count)

            active = event_image_count > 0
            if not bool(active.any().item()):
                continue
            per_event_mean = event_sum[active] / event_image_count[active, None]
            per_event_second = event_second[active] / event_image_count[active, None]
            candidate_mean = per_event_mean.mean(dim=0)
            candidate_second = per_event_second.mean(dim=0)

            if not bool(self.initialized[class_idx].item()):
                self.means[class_idx].copy_(candidate_mean)
                self.second_moments[class_idx].copy_(candidate_second)
                self.initialized[class_idx] = True
            else:
                self.means[class_idx].mul_(self.momentum).add_(
                    candidate_mean, alpha=1.0 - self.momentum
                )
                self.second_moments[class_idx].mul_(self.momentum).add_(
                    candidate_second, alpha=1.0 - self.momentum
                )

            self.event_updates[class_idx] += float(active.sum().item())
            self.image_updates[class_idx] += float(event_image_count.sum().item())
            self.pixel_observations[class_idx] += float(pixel_count.item())

    def ready(self) -> bool:
        return bool(self.initialized.all().item())

    def variance(self) -> Tensor:
        variance = self.second_moments - self.means.square()
        return variance.clamp_min(self.variance_floor)

    def query(self, embedding: Tensor) -> Tensor:
        """Return dense class log densities [B,3,H,W]."""
        z = F.normalize(embedding, dim=1)
        means = self.means.detach().clone()
        variance = self.variance().detach().clone()
        diff = z[:, None] - means[None, :, :, None, None]
        score = -0.5 * (
            diff.square() / variance[None, :, :, None, None]
            + variance.log()[None, :, :, None, None]
        ).mean(dim=2)
        if not self.ready():
            score = score * 0.0
        return score

    def diagnostics(self) -> Dict[str, Tensor]:
        means = F.normalize(self.means.detach(), dim=1)
        cosine = means @ means.t()
        variance = self.variance().detach()
        return {
            "gaussian_initialized": self.initialized.detach().clone(),
            "gaussian_means": self.means.detach().clone(),
            "gaussian_variance_mean": variance.mean(dim=1),
            "gaussian_variance_min": variance.min(dim=1).values,
            "gaussian_variance_max": variance.max(dim=1).values,
            "gaussian_centre_cosine": cosine,
            "gaussian_event_updates": self.event_updates.detach().clone(),
            "gaussian_image_updates": self.image_updates.detach().clone(),
            "gaussian_pixel_observations": self.pixel_observations.detach().clone(),
        }


class DamageFormer_IARM(DamageFormer_BG):
    """DamageFormer with intact-anchored aligned residual modelling.

    ``enable_residual_fusion=False`` leaves the damage output exactly on the
    BG baseline path while still allowing auxiliary representation experiments.
    This makes E1 a clean diagnostic ablation.
    """

    STAGE_CHANNELS = {3: 384, 4: 768}

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
        enable_residual_fusion: bool = False,
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
        )
        stages = tuple(sorted(set(int(stage) for stage in residual_stages)))
        if not stages or any(stage not in self.STAGE_CHANNELS for stage in stages):
            raise ValueError("residual_stages must be a non-empty subset of {3,4}")
        if residual_dim <= 0:
            raise ValueError("residual_dim must be positive")
        if residual_gamma_max < 0:
            raise ValueError("residual_gamma_max must be non-negative")

        self.residual_stages = stages
        self.residual_dim = int(residual_dim)
        self.detach_backbone_for_residual = bool(detach_backbone_for_residual)
        self.enable_residual_fusion = bool(enable_residual_fusion)
        self.residual_gamma_max = float(residual_gamma_max)

        self.optical_projectors = nn.ModuleDict()
        self.sar_projectors = nn.ModuleDict()
        self.aligners = nn.ModuleDict()
        self.normal_mappers = nn.ModuleDict()
        self.residual_encoders = nn.ModuleDict()
        for stage in self.residual_stages:
            channels = self.STAGE_CHANNELS[stage]
            key = str(stage)
            self.optical_projectors[key] = FeatureProjector(channels, residual_dim)
            self.sar_projectors[key] = FeatureProjector(channels, residual_dim)
            self.aligners[key] = LightweightFlowAlign(
                residual_dim,
                hidden_channels=flow_hidden_dim,
                max_offset_pixels=max_offset_pixels,
            )
            self.normal_mappers[key] = NormalRelationMapper(residual_dim)
            self.residual_encoders[key] = ResidualEncoder(residual_dim)

        self.residual_fuser = nn.Sequential(
            ConvGNAct(len(self.residual_stages) * residual_dim, residual_dim, kernel_size=3),
            nn.Conv2d(residual_dim, residual_dim, kernel_size=1, bias=False),
        )
        self.damage_exist_head = nn.Sequential(
            ConvGNAct(residual_dim, residual_dim, kernel_size=3),
            nn.Conv2d(residual_dim, 2, kernel_size=1),
        )
        self.residual_to_damage = nn.Sequential(
            ConvGNAct(residual_dim, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
        )
        self.residual_gamma_raw = nn.Parameter(torch.zeros(()))

        self.class_distribution_memory = EventBalancedDiagonalGaussianMemory(
            embedding_dim=residual_dim,
            num_events=num_events,
            momentum=gaussian_step_momentum,
            variance_floor=gaussian_variance_floor,
            max_pixels_per_image_class=max_pixels_per_image_class,
        )

    @staticmethod
    def _extract_stages(features: Sequence[Tensor]) -> Dict[int, Tensor]:
        if len(features) != 5:
            raise RuntimeError(
                f"Expected 5 Swin feature maps from the BRIGHT extractor, got {len(features)}"
            )
        _, stage1, stage2, stage3, stage4 = features
        return {
            1: DamageFormer_BG._to_nchw(stage1),
            2: DamageFormer_BG._to_nchw(stage2),
            3: DamageFormer_BG._to_nchw(stage3),
            4: DamageFormer_BG._to_nchw(stage4),
        }

    def _encode_all(
        self, pre_data: Tensor, post_data: Tensor
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Tensor, Tensor]:
        pre_stages = self._extract_stages(self.encoder_1(pre_data))
        post_stages = self._extract_stages(self.encoder_2(post_data))
        pre_ms = self._multi_scale_fuse(
            pre_stages[1], pre_stages[2], pre_stages[3], pre_stages[4]
        )
        post_ms = self._multi_scale_fuse(
            post_stages[1], post_stages[2], post_stages[3], post_stages[4]
        )
        final_feat_rgb = self.fusion_layer_1(pre_ms)
        final_feat_sar = self.fusion_layer_2(post_ms)
        return pre_stages, post_stages, final_feat_rgb, final_feat_sar

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
                optical, aligned_sar, predicted_normal
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

        fused = self.residual_fuser(torch.cat(residuals, dim=1))
        metric_embedding = F.normalize(fused, dim=1)
        aux["metric_embedding"] = metric_embedding
        return metric_embedding, aux

    def residual_gamma(self) -> Tensor:
        return torch.tanh(self.residual_gamma_raw) * self.residual_gamma_max

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
        del current_iter  # reserved for schedule-compatible trainers
        pre_stages, post_stages, final_feat_rgb, final_feat_sar = self._encode_all(
            pre_data, post_data
        )

        loc_logits_feat = self.clf_1(final_feat_rgb)
        p_building = torch.softmax(loc_logits_feat, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        final_feat_fused = self.fusion_layer_3(
            torch.cat([final_feat_rgb, final_feat_sar], dim=1)
        )
        final_feat_fused_roi = final_feat_fused * roi_weight

        metric_embedding, residual_aux = self._build_residual(
            pre_stages,
            post_stages,
            output_size=final_feat_fused_roi.shape[-2:],
        )
        damage_exist_logits_feat = self.damage_exist_head(metric_embedding)

        gamma = self.residual_gamma()
        apply_residual = self.enable_residual_fusion and not force_residual_off
        residual_feature = self.residual_to_damage(metric_embedding) * roi_weight
        # Keep the residual path in the autograd graph even for E0/E1 so DDP
        # does not see unused parameters. A zero scale preserves the exact
        # baseline output while producing zero gradients for the fusion path.
        fusion_scale = gamma if apply_residual else gamma * 0.0
        damage_feature = final_feat_fused_roi + fusion_scale * residual_feature

        damage_logits_feat = self.clf_2(damage_feature)

        if self.training and damage_target is not None:
            self.class_distribution_memory.update(
                metric_embedding.detach(), damage_target, event_ids
            )
        anchor_logits_feat = self.class_distribution_memory.query(metric_embedding)

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

        aux: Dict[str, Tensor] = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "final_feat_rgb": final_feat_rgb,
            "final_feat_sar": final_feat_sar,
            "final_feat_fused": final_feat_fused,
            "final_feat_fused_roi": final_feat_fused_roi,
            "residual_feature": residual_feature,
            "damage_feature": damage_feature,
            "main_damage_logits_feat": self.clf_2(final_feat_fused_roi),
            "damage_logits_feat": damage_logits_feat,
            "damage_exist_logits_feat": damage_exist_logits_feat,
            "anchor_logits_feat": anchor_logits_feat,
            "anchor_ready": torch.tensor(
                self.class_distribution_memory.ready(),
                device=damage_logits_feat.device,
                dtype=torch.bool,
            ),
            "residual_gamma": gamma,
            "residual_applied": torch.tensor(
                bool(apply_residual),
                device=damage_logits_feat.device,
                dtype=torch.bool,
            ),
        }
        aux.update(residual_aux)
        aux.update(self.class_distribution_memory.diagnostics())
        return output_loc, output_damage, output_exist, aux


__all__ = [
    "DamageFormer_IARM",
    "EventBalancedDiagonalGaussianMemory",
    "LightweightFlowAlign",
]
