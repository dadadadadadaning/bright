# -*- coding: utf-8 -*-
"""
S3V6: Event-balanced role-aware K=[1,3,3] prototype rescue for BRIGHT.

Key changes from S3V5
---------------------
1. Event-balanced memory update:
   pixel -> image candidate -> event candidate -> equal-event prototype EMA.
2. The damage training loss uses the main classifier output only by default.
   Prototype correction is inference-only, so the main classifier cannot rely
   on the prototype branch during optimization.
3. Prototype rescue is restricted to pixels whose main prediction is Intact.
4. A positive prototype margin is required before rescue is activated.
5. DDP-safe: fixed-size per-event sums/counts are all-reduced on every role
   update, keeping prototype buffers identical on all ranks.

The module keeps the original CE + Lovasz objective. Optional event-balanced
loss is implemented in the trainer, not in this model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_BG import DamageFormer_BG
except ImportError:
    from DamageFormer_BG import DamageFormer_BG


class EventBalancedRoleAwareK3Memory(nn.Module):
    """Role-aware prototype memory with equal event contribution.

    Damage-class index in this memory:
        0 -> Intact
        1 -> Damaged
        2 -> Destroyed

    Role slots:
        0 -> core / typical
        1 -> hard-to-intact
        2 -> diverse

    Valid K configuration:
        Intact    [core]
        Damaged   [core, hard, diverse]
        Destroyed [core, hard, diverse]
    """

    NUM_DAMAGE_CLASSES = 3
    K = 3

    def __init__(
        self,
        in_channels: int = 256,
        proto_dim: int = 256,
        num_events: int = 14,
        momentum: float = 0.99,
        tau: float = 0.1,
        warmup_iters: int = 5000,
        core_confidence: float = 0.70,
        max_pixels_per_image_role: int = 2048,
    ) -> None:
        super().__init__()

        if proto_dim != in_channels:
            raise ValueError(
                "S3V6 uses the validated direct256 prototype space; "
                f"got in_channels={in_channels}, proto_dim={proto_dim}."
            )
        if num_events <= 0:
            raise ValueError(f"num_events must be positive, got {num_events}")
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
        self.num_events = int(num_events)
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.warmup_iters = int(warmup_iters)
        self.core_confidence = float(core_confidence)
        self.max_pixels_per_image_role = int(max_pixels_per_image_role)

        # direct256: no trainable projection that could become unused when
        # prototype correction is disabled during training.
        self.feature_proj = nn.Identity()

        prototypes = F.normalize(
            torch.randn(
                self.NUM_DAMAGE_CLASSES,
                self.K,
                self.proto_dim,
            ),
            dim=-1,
        )
        self.register_buffer("prototypes", prototypes)
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(
                self.NUM_DAMAGE_CLASSES,
                self.K,
                dtype=torch.bool,
            ),
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

        # Diagnostics. These counters are globally synchronized because every
        # rank receives the same reduced event/image/pixel statistics.
        self.register_buffer(
            "prototype_event_updates",
            torch.zeros(
                self.NUM_DAMAGE_CLASSES,
                self.K,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "prototype_image_updates",
            torch.zeros(
                self.NUM_DAMAGE_CLASSES,
                self.K,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "prototype_pixel_observations",
            torch.zeros(
                self.NUM_DAMAGE_CLASSES,
                self.K,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _prepare_target(
        target: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
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
            target.unsqueeze(1).float(),
            size=size,
            mode="nearest",
        )
        return target[:, 0].long()

    def _distributed_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @torch.no_grad()
    def _update_slot_event_balanced(
        self,
        class_idx: int,
        slot_idx: int,
        z: torch.Tensor,
        role_mask: torch.Tensor,
        event_ids: torch.Tensor,
    ) -> None:
        """Update one role with image-normalized, event-balanced candidates.

        Stage 1: each image contributes at most one normalized candidate.
        Stage 2: candidates from the same event are averaged.
        Stage 3: all active events are averaged with equal weight.

        Therefore, a large event, a tile with many buildings, or a large
        building cannot dominate the prototype merely through pixel count.
        """
        if not bool(self.prototype_valid[class_idx, slot_idx].item()):
            return

        batch_size = z.shape[0]
        event_sums = torch.zeros(
            self.num_events,
            self.proto_dim,
            device=z.device,
            dtype=z.dtype,
        )
        event_image_counts = torch.zeros(
            self.num_events,
            device=z.device,
            dtype=torch.float32,
        )
        pixel_count = torch.zeros(
            1,
            device=z.device,
            dtype=torch.float32,
        )

        for batch_idx in range(batch_size):
            mask = role_mask[batch_idx]
            number_of_pixels = int(mask.sum().item())
            if number_of_pixels <= 0:
                continue

            features = z[batch_idx].permute(1, 2, 0)[mask]
            if features.shape[0] > self.max_pixels_per_image_role:
                chosen = torch.randperm(
                    features.shape[0],
                    device=features.device,
                )[: self.max_pixels_per_image_role]
                features = features[chosen]

            features = F.normalize(features, dim=1)
            image_candidate = F.normalize(
                features.mean(dim=0),
                dim=0,
            )

            event_id = int(event_ids[batch_idx].item())
            if not 0 <= event_id < self.num_events:
                raise ValueError(
                    f"event id {event_id} is outside [0,{self.num_events})"
                )
            event_sums[event_id] += image_candidate
            event_image_counts[event_id] += 1.0
            pixel_count += float(number_of_pixels)

        # Fixed-size tensors make this safe even when the two ranks contain
        # completely different events in the local micro-batch.
        self._distributed_sum(event_sums)
        self._distributed_sum(event_image_counts)
        self._distributed_sum(pixel_count)

        active_events = event_image_counts > 0
        number_of_active_events = int(active_events.sum().item())
        if number_of_active_events <= 0:
            return

        event_candidates = event_sums[active_events] / event_image_counts[
            active_events
        ].unsqueeze(1)
        event_candidates = F.normalize(event_candidates, dim=1)

        # Equal event contribution, independent of event tile count.
        candidate = F.normalize(event_candidates.mean(dim=0), dim=0)

        if not bool(
            self.prototype_initialized[class_idx, slot_idx].item()
        ):
            self.prototypes[class_idx, slot_idx].copy_(candidate)
            self.prototype_initialized[class_idx, slot_idx] = True
        else:
            old = F.normalize(
                self.prototypes[class_idx, slot_idx],
                dim=0,
            )
            updated = (
                self.momentum * old
                + (1.0 - self.momentum) * candidate
            )
            self.prototypes[class_idx, slot_idx].copy_(
                F.normalize(updated, dim=0)
            )

        self.prototype_event_updates[class_idx, slot_idx] += float(
            number_of_active_events
        )
        self.prototype_image_updates[class_idx, slot_idx] += float(
            event_image_counts.sum().item()
        )
        self.prototype_pixel_observations[class_idx, slot_idx] += float(
            pixel_count.item()
        )

    @torch.no_grad()
    def update_memory(
        self,
        z: torch.Tensor,
        main_logits: torch.Tensor,
        damage_target: torch.Tensor,
        event_ids: torch.Tensor,
    ) -> None:
        if event_ids is None:
            raise ValueError(
                "event_ids are required during S3V6 prototype memory update."
            )
        if event_ids.dim() != 1 or event_ids.shape[0] != z.shape[0]:
            raise ValueError(
                "event_ids must be [B], got "
                f"{tuple(event_ids.shape)} for B={z.shape[0]}"
            )

        target = self._prepare_target(damage_target, z.shape[-2:])
        if main_logits.shape[-2:] != z.shape[-2:]:
            main_logits = F.interpolate(
                main_logits,
                size=z.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probability = torch.softmax(main_logits.detach(), dim=1)
        confidence, prediction = probability.max(dim=1)

        intact_all = target == 1
        intact_core = (
            intact_all
            & (prediction == 1)
            & (confidence >= self.core_confidence)
        )

        # Per-image fallback prevents an event from disappearing merely because
        # an early model is under-confident on all Intact pixels in that tile.
        for batch_idx in range(z.shape[0]):
            if int(intact_core[batch_idx].sum().item()) < 8:
                intact_core[batch_idx] = intact_all[batch_idx]

        self._update_slot_event_balanced(
            class_idx=0,
            slot_idx=0,
            z=z,
            role_mask=intact_core,
            event_ids=event_ids,
        )

        for class_idx, label_value in ((1, 2), (2, 3)):
            class_mask = target == label_value
            core_mask = (
                class_mask
                & (prediction == label_value)
                & (confidence >= self.core_confidence)
            )
            hard_mask = class_mask & (prediction == 1)
            diverse_mask = class_mask & (~core_mask) & (~hard_mask)

            self._update_slot_event_balanced(
                class_idx=class_idx,
                slot_idx=0,
                z=z,
                role_mask=core_mask,
                event_ids=event_ids,
            )
            self._update_slot_event_balanced(
                class_idx=class_idx,
                slot_idx=1,
                z=z,
                role_mask=hard_mask,
                event_ids=event_ids,
            )
            self._update_slot_event_balanced(
                class_idx=class_idx,
                slot_idx=2,
                z=z,
                role_mask=diverse_mask,
                event_ids=event_ids,
            )

    def _class_ready(self) -> torch.Tensor:
        return (
            self.prototype_initialized & self.prototype_valid
        ).any(dim=1)

    def is_active(self, current_iter: Optional[int]) -> bool:
        ready = bool(self._class_ready().all().item())
        if current_iter is None:
            return ready
        return ready and int(current_iter) >= self.warmup_iters

    def query(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prototypes = F.normalize(
            self.prototypes.detach().clone(),
            dim=-1,
        )
        similarity = torch.einsum(
            "bdhw,ckd->bckhw",
            z,
            prototypes,
        )
        slot_mask = (
            self.prototype_valid & self.prototype_initialized
        )
        similarity = similarity.masked_fill(
            ~slot_mask[None, :, :, None, None],
            -1.0e4,
        )
        return similarity.max(dim=2)

    def forward(
        self,
        feature: torch.Tensor,
        main_logits: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        event_ids: Optional[torch.Tensor] = None,
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
                event_ids=event_ids,
            )

        valid_float = self.prototype_valid.float()

        def normalized_counter(counter: torch.Tensor) -> torch.Tensor:
            masked = counter.detach() * valid_float
            return masked / masked.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-6)

        return {
            "proto_active": active,
            "proto_initialized": (
                self.prototype_initialized.detach().clone()
            ),
            "proto_valid": self.prototype_valid.detach().clone(),
            "proto_event_usage_ratio": normalized_counter(
                self.prototype_event_updates
            ),
            "proto_image_usage_ratio": normalized_counter(
                self.prototype_image_updates
            ),
            "proto_pixel_usage_ratio": normalized_counter(
                self.prototype_pixel_observations
            ),
            # Compatibility name used by earlier diagnostics. S3V6 defines
            # usage by equal-event update count rather than raw pixels.
            "proto_usage_ratio": normalized_counter(
                self.prototype_event_updates
            ),
            "proto_class_scores": class_scores,
            "proto_winning_slot": (
                winning_slot.detach()
                if winning_slot is not None
                else None
            ),
        }


class DamageFormer_BGMP_K3EventBalanced_DDP(DamageFormer_BG):
    """S3V6 event-balanced K3 memory with protected main classifier."""

    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        proto_dim: int = 256,
        num_events: int = 14,
        proto_momentum: float = 0.99,
        proto_tau: float = 0.1,
        proto_warmup_iters: int = 5000,
        core_confidence: float = 0.70,
        lambda_damaged: float = 0.05,
        lambda_destroyed: float = 0.05,
        evidence_clip: float = 4.0,
        evidence_margin: float = 0.03,
        apply_proto_during_training: bool = False,
        rescue_only_main_intact: bool = True,
        max_pixels_per_image_role: int = 2048,
    ) -> None:
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
        )

        if lambda_damaged < 0 or lambda_destroyed < 0:
            raise ValueError(
                "Prototype correction weights must be non-negative."
            )
        if evidence_clip <= 0:
            raise ValueError("evidence_clip must be positive.")
        if evidence_margin < 0:
            raise ValueError("evidence_margin must be non-negative.")

        self.lambda_damaged = float(lambda_damaged)
        self.lambda_destroyed = float(lambda_destroyed)
        self.evidence_clip = float(evidence_clip)
        self.evidence_margin = float(evidence_margin)
        self.apply_proto_during_training = bool(
            apply_proto_during_training
        )
        self.rescue_only_main_intact = bool(
            rescue_only_main_intact
        )

        self.damage_proto_memory = EventBalancedRoleAwareK3Memory(
            in_channels=256,
            proto_dim=proto_dim,
            num_events=num_events,
            momentum=proto_momentum,
            tau=proto_tau,
            warmup_iters=proto_warmup_iters,
            core_confidence=core_confidence,
            max_pixels_per_image_role=max_pixels_per_image_role,
        )

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        event_ids: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
        return_aux: bool = False,
        force_proto_off: bool = False,
    ):
        final_feat_1, feat_2 = self._encode_and_fuse(
            pre_data,
            post_data,
        )

        loc_logits_feat = self.clf_1(final_feat_1)
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

        final_feat_2 = self.fusion_layer_3(
            torch.cat([final_feat_1, feat_2], dim=1)
        )
        final_feat_2_roi = final_feat_2 * roi_weight
        main_logits_feat = self.clf_2(final_feat_2_roi)

        memory_stats = self.damage_proto_memory(
            feature=final_feat_2_roi,
            main_logits=main_logits_feat,
            damage_target=damage_target,
            event_ids=event_ids,
            current_iter=current_iter,
        )

        proto_ready = (
            bool(memory_stats["proto_active"])
            and not force_proto_off
        )
        class_scores = memory_stats["proto_class_scores"]
        apply_correction = (
            proto_ready
            and class_scores is not None
            and (
                (not self.training)
                or self.apply_proto_during_training
            )
        )

        final_logits_feat = main_logits_feat
        gate_damaged = None
        gate_destroyed = None
        evidence_damaged = None
        evidence_destroyed = None
        main_prediction = None

        if apply_correction:
            main_prob = torch.softmax(
                main_logits_feat.detach(),
                dim=1,
            )
            main_prediction = main_prob.argmax(
                dim=1,
                keepdim=True,
            )

            p_intact = main_prob[:, 1:2]
            p_damaged = main_prob[:, 2:3]
            p_destroyed = main_prob[:, 3:4]

            building_gate = F.interpolate(
                p_building.detach(),
                size=main_logits_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if self.rescue_only_main_intact:
                rescue_gate = (main_prediction == 1).float()
            else:
                rescue_gate = torch.ones_like(p_intact)

            # High only when the relevant pair carries most of the probability
            # mass and the main classifier is uncertain within that pair.
            gate_damaged = (
                building_gate
                * (1.0 - torch.abs(p_intact - p_damaged))
                * (p_intact + p_damaged)
                * rescue_gate
            ).clamp(0.0, 1.0)
            gate_destroyed = (
                building_gate
                * (1.0 - torch.abs(p_intact - p_destroyed))
                * (p_intact + p_destroyed)
                * rescue_gate
            ).clamp(0.0, 1.0)

            intact_score = class_scores[:, 0:1]
            evidence_damaged = torch.relu(
                (
                    class_scores[:, 1:2]
                    - intact_score
                    - self.evidence_margin
                )
                / self.damage_proto_memory.tau
            ).clamp(max=self.evidence_clip)
            evidence_destroyed = torch.relu(
                (
                    class_scores[:, 2:3]
                    - intact_score
                    - self.evidence_margin
                )
                / self.damage_proto_memory.tau
            ).clamp(max=self.evidence_clip)

            final_logits_feat = main_logits_feat.clone()
            final_logits_feat[:, 2:3] += (
                self.lambda_damaged
                * gate_damaged
                * evidence_damaged
            )
            final_logits_feat[:, 3:4] += (
                self.lambda_destroyed
                * gate_destroyed
                * evidence_destroyed
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
        correction_scale = final_feat_2_roi.new_tensor(
            max(self.lambda_damaged, self.lambda_destroyed)
            if apply_correction
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
            "main_prediction_feat": main_prediction,
            "proto_class_similarity": class_similarity_flat,
            "proto_class_scores": class_scores,
            "proto_gate_damaged": gate_damaged,
            "proto_gate_destroyed": gate_destroyed,
            "proto_evidence_damaged": evidence_damaged,
            "proto_evidence_destroyed": evidence_destroyed,
            "proto_gamma": correction_scale,
            "proto_correction_applied": apply_correction,
            **memory_stats,
        }
        return output_loc, output_dam, aux
