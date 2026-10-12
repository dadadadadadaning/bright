# -*- coding: utf-8 -*-
"""
S3V5: Role-aware K=3 prototype memory + uncertainty-gated logit correction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_BG import DamageFormer_BG
except ImportError:
    from DamageFormer_BG import DamageFormer_BG


class RoleAwareK3PrototypeMemory(nn.Module):
    NUM_DAMAGE_CLASSES = 3
    K = 3

    def __init__(
        self,
        in_channels: int = 256,
        proto_dim: int = 256,
        momentum: float = 0.99,
        tau: float = 0.1,
        warmup_iters: int = 5000,
        core_confidence: float = 0.70,
        max_samples_per_role: int = 4096,
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0,1), got {momentum}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if not 0.0 <= core_confidence <= 1.0:
            raise ValueError(
                f"core_confidence must be in [0,1], got {core_confidence}"
            )

        self.in_channels = int(in_channels)
        self.proto_dim = int(proto_dim)
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.warmup_iters = int(warmup_iters)
        self.core_confidence = float(core_confidence)
        self.max_samples_per_role = int(max_samples_per_role)

        if self.proto_dim == self.in_channels:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Conv2d(
                self.in_channels, self.proto_dim, kernel_size=1, bias=False
            )

        prototypes = F.normalize(
            torch.randn(self.NUM_DAMAGE_CLASSES, self.K, self.proto_dim),
            dim=-1,
        )
        self.register_buffer("prototypes", prototypes)
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.NUM_DAMAGE_CLASSES, self.K, dtype=torch.bool),
        )
        self.register_buffer(
            "prototype_valid",
            torch.tensor(
                [
                    [True, False, False],
                    [True, True, True],
                    [True, True, True],
                ],
                dtype=torch.bool,
            ),
        )
        self.register_buffer(
            "prototype_usage",
            torch.zeros(self.NUM_DAMAGE_CLASSES, self.K, dtype=torch.float32),
        )

    @staticmethod
    def _prepare_target(target: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target[:, 0]
            else:
                target = target.argmax(dim=1)
        if target.dim() != 3:
            raise ValueError(
                "damage_target must be [B,H,W], [B,1,H,W], or [B,C,H,W], "
                f"got {tuple(target.shape)}"
            )
        target = F.interpolate(
            target.unsqueeze(1).float(), size=size, mode="nearest"
        )
        return target[:, 0].long()

    @torch.no_grad()
    def _update_slot(
        self,
        class_idx: int,
        slot_idx: int,
        features: torch.Tensor,
    ) -> None:
        if not bool(self.prototype_valid[class_idx, slot_idx].item()):
            return
        if features.numel() == 0:
            return
        if features.shape[0] > self.max_samples_per_role:
            perm = torch.randperm(
                features.shape[0], device=features.device
            )[: self.max_samples_per_role]
            features = features[perm]

        features = F.normalize(features, dim=1)
        candidate = F.normalize(features.mean(dim=0), dim=0)

        if not bool(self.prototype_initialized[class_idx, slot_idx].item()):
            self.prototypes[class_idx, slot_idx].copy_(candidate)
            self.prototype_initialized[class_idx, slot_idx] = True
        else:
            old = F.normalize(self.prototypes[class_idx, slot_idx], dim=0)
            updated = self.momentum * old + (1.0 - self.momentum) * candidate
            self.prototypes[class_idx, slot_idx].copy_(
                F.normalize(updated, dim=0)
            )

        self.prototype_usage[class_idx, slot_idx] += float(features.shape[0])

    @torch.no_grad()
    def update_memory(
        self,
        z: torch.Tensor,
        main_logits: torch.Tensor,
        damage_target: torch.Tensor,
    ) -> None:
        target = self._prepare_target(damage_target, z.shape[-2:])
        if main_logits.shape[-2:] != z.shape[-2:]:
            main_logits = F.interpolate(
                main_logits,
                size=z.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probabilities = torch.softmax(main_logits.detach(), dim=1)
        confidence, prediction = probabilities.max(dim=1)

        features = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)
        target_flat = target.reshape(-1)
        pred_flat = prediction.reshape(-1)
        conf_flat = confidence.reshape(-1)

        intact_all = target_flat == 1
        intact_core = (
            intact_all
            & (pred_flat == 1)
            & (conf_flat >= self.core_confidence)
        )
        if int(intact_core.sum().item()) < 8:
            intact_core = intact_all
        self._update_slot(0, 0, features[intact_core])

        for class_idx, label_value in ((1, 2), (2, 3)):
            class_mask = target_flat == label_value
            core_mask = (
                class_mask
                & (pred_flat == label_value)
                & (conf_flat >= self.core_confidence)
            )
            hard_mask = class_mask & (pred_flat == 1)
            diverse_mask = class_mask & (~core_mask) & (~hard_mask)

            self._update_slot(class_idx, 0, features[core_mask])
            self._update_slot(class_idx, 1, features[hard_mask])
            self._update_slot(class_idx, 2, features[diverse_mask])

    def _class_ready(self) -> torch.Tensor:
        return (
            self.prototype_initialized & self.prototype_valid
        ).any(dim=1)

    def is_active(self, current_iter: Optional[int]) -> bool:
        ready = bool(self._class_ready().all().item())
        if current_iter is None:
            return ready
        return ready and int(current_iter) >= self.warmup_iters

    def query(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prototypes = F.normalize(self.prototypes.detach().clone(), dim=-1)
        similarity = torch.einsum("bdhw,ckd->bckhw", z, prototypes)
        slot_mask = self.prototype_valid & self.prototype_initialized
        similarity = similarity.masked_fill(
            ~slot_mask[None, :, :, None, None], -1.0e4
        )
        return similarity.max(dim=2)

    def forward(
        self,
        feature: torch.Tensor,
        main_logits: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
    ) -> Dict[str, object]:
        z = F.normalize(self.feature_proj(feature), dim=1)
        active = self.is_active(current_iter)

        class_scores = None
        winning_slot = None
        if active:
            class_scores, winning_slot = self.query(z)

        if self.training and damage_target is not None:
            self.update_memory(
                z=z.detach(),
                main_logits=main_logits.detach(),
                damage_target=damage_target,
            )

        usage = self.prototype_usage.detach() * self.prototype_valid.float()
        usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        return {
            "proto_active": active,
            "proto_initialized": self.prototype_initialized.detach().clone(),
            "proto_valid": self.prototype_valid.detach().clone(),
            "proto_usage_ratio": usage_ratio.detach().clone(),
            "proto_class_scores": class_scores,
            "proto_winning_slot": (
                winning_slot.detach() if winning_slot is not None else None
            ),
        }


class DamageFormer_BGMP_K3Gated(DamageFormer_BG):
    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        proto_dim: int = 256,
        proto_momentum: float = 0.99,
        proto_tau: float = 0.1,
        proto_warmup_iters: int = 5000,
        core_confidence: float = 0.70,
        lambda_damaged: float = 0.05,
        lambda_destroyed: float = 0.05,
        evidence_clip: float = 4.0,
    ) -> None:
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
        )
        if lambda_damaged < 0 or lambda_destroyed < 0:
            raise ValueError("Prototype correction weights must be non-negative.")
        if evidence_clip <= 0:
            raise ValueError("evidence_clip must be positive.")

        self.lambda_damaged = float(lambda_damaged)
        self.lambda_destroyed = float(lambda_destroyed)
        self.evidence_clip = float(evidence_clip)

        self.damage_proto_memory = RoleAwareK3PrototypeMemory(
            in_channels=256,
            proto_dim=proto_dim,
            momentum=proto_momentum,
            tau=proto_tau,
            warmup_iters=proto_warmup_iters,
            core_confidence=core_confidence,
        )

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
        return_aux: bool = False,
        force_proto_off: bool = False,
    ):
        final_feat_1, feat_2 = self._encode_and_fuse(pre_data, post_data)

        loc_logits_feat = self.clf_1(final_feat_1)
        p_building = torch.softmax(loc_logits_feat, dim=1)[:, 1:2]
        building_prior = (
            p_building.detach()
            if self.detach_building_prior
            else p_building
        )
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        final_feat_2 = self.fusion_layer_3(
            torch.cat([final_feat_1, feat_2], dim=1)
        )
        final_feat_2_roi = final_feat_2 * roi_weight
        main_logits_feat = self.clf_2(final_feat_2_roi)

        memory_stats = self.damage_proto_memory(
            feature=final_feat_2_roi,
            main_logits=main_logits_feat,
            damage_target=damage_target,
            current_iter=current_iter,
        )

        proto_active = bool(memory_stats["proto_active"]) and not force_proto_off
        class_scores = memory_stats["proto_class_scores"]

        final_logits_feat = main_logits_feat
        gate_damaged = gate_destroyed = None
        evidence_damaged = evidence_destroyed = None

        if proto_active and class_scores is not None:
            main_prob = torch.softmax(main_logits_feat.detach(), dim=1)
            p_intact = main_prob[:, 1:2]
            p_damaged = main_prob[:, 2:3]
            p_destroyed = main_prob[:, 3:4]

            building_gate = F.interpolate(
                p_building.detach(),
                size=main_logits_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            gate_damaged = (
                building_gate
                * (1.0 - torch.abs(p_intact - p_damaged))
                * (p_intact + p_damaged)
            ).clamp(0.0, 1.0)
            gate_destroyed = (
                building_gate
                * (1.0 - torch.abs(p_intact - p_destroyed))
                * (p_intact + p_destroyed)
            ).clamp(0.0, 1.0)

            intact_score = class_scores[:, 0:1]
            evidence_damaged = torch.relu(
                (class_scores[:, 1:2] - intact_score)
                / self.damage_proto_memory.tau
            ).clamp(max=self.evidence_clip)
            evidence_destroyed = torch.relu(
                (class_scores[:, 2:3] - intact_score)
                / self.damage_proto_memory.tau
            ).clamp(max=self.evidence_clip)

            final_logits_feat = main_logits_feat.clone()
            final_logits_feat[:, 2:3] += (
                self.lambda_damaged * gate_damaged * evidence_damaged
            )
            final_logits_feat[:, 3:4] += (
                self.lambda_destroyed * gate_destroyed * evidence_destroyed
            )

        output_loc = F.interpolate(
            loc_logits_feat,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_dam = F.interpolate(
            final_logits_feat,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if not return_aux:
            return output_loc, output_dam

        class_similarity_flat = (
            class_scores.permute(0, 2, 3, 1).reshape(-1, 3)
            if class_scores is not None
            else None
        )
        scalar = final_feat_2_roi.new_tensor(
            max(self.lambda_damaged, self.lambda_destroyed)
            if proto_active
            else 0.0
        )

        aux = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "final_feat_fused": final_feat_2,
            "final_feat_fused_roi": final_feat_2_roi,
            "final_feat_fused_proto": final_feat_2_roi,
            "main_damage_logits_feat": main_logits_feat,
            "final_damage_logits_feat": final_logits_feat,
            "proto_class_similarity": class_similarity_flat,
            "proto_class_scores": class_scores,
            "proto_gate_damaged": gate_damaged,
            "proto_gate_destroyed": gate_destroyed,
            "proto_evidence_damaged": evidence_damaged,
            "proto_evidence_destroyed": evidence_destroyed,
            "proto_gamma": scalar,
            **memory_stats,
        }
        return output_loc, output_dam, aux
