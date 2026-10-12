# -*- coding: utf-8 -*-
"""DamageFormer + Damage Prototype Memory V2.

V2 changes relative to BGMP V1:
1) Prototype memory is built directly in the 256-D task-supervised damage feature
   space. There is no randomly initialized 256->128 feature projector.
2) K=1 uses a class mean + EMA update.
3) K>1 uses spherical hard nearest-prototype assignment. Empty prototypes are
   reinitialized from features farthest from the current prototype set.
4) Query diagnostics expose prototype margin, attention entropy and actual
   prototype effect ratio.

No prototype auxiliary loss is introduced in V2. The original DamageFormer
CE + Lovasz objective can remain unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_BG import DamageFormer_BG
except ImportError:
    from DamageFormer_BG import DamageFormer_BG


class DamagePrototypeMemoryV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        prototypes_per_class: int = 1,
        momentum: float = 0.99,
        tau: float = 0.1,
        warmup_iters: int = 5000,
        max_samples_per_class: int = 4096,
        assignment_mode: str = "auto",
    ) -> None:
        super().__init__()
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0,1), got {momentum}")
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        if assignment_mode not in {"auto", "mean", "hard"}:
            raise ValueError("assignment_mode must be one of: auto, mean, hard")

        self.num_damage_classes = 3  # labels 1/2/3: intact/damaged/destroyed
        self.proto_dim = int(in_channels)
        self.k = int(prototypes_per_class)
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.warmup_iters = int(warmup_iters)
        self.max_samples_per_class = int(max_samples_per_class)
        self.assignment_mode = assignment_mode

        # V2 key: no random prototype embedding projector.
        self.feature_proj = nn.Identity()

        # Kept from V1 for the first controlled V2 experiment. This means that
        # K1 V1 -> K1 V2 mainly changes the prototype representation space.
        self.context_proj = nn.Sequential(
            nn.Conv2d(self.proto_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

        proto = F.normalize(
            torch.randn(self.num_damage_classes, self.k, self.proto_dim), dim=-1
        )
        self.register_buffer("prototypes", proto)
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.num_damage_classes, dtype=torch.bool),
        )
        # In V2 this is natural sample count, not balanced Sinkhorn mass.
        self.register_buffer(
            "prototype_usage",
            torch.zeros(self.num_damage_classes, self.k),
        )
        self.register_buffer(
            "empty_reinit_count",
            torch.zeros(self.num_damage_classes, self.k, dtype=torch.long),
        )

    def gamma(self) -> torch.Tensor:
        return 0.1 * torch.sigmoid(self.gamma_raw)

    def resolved_assignment_mode(self) -> str:
        if self.assignment_mode == "auto":
            return "mean" if self.k == 1 else "hard"
        if self.k == 1:
            return "mean"
        return self.assignment_mode

    @staticmethod
    def _prepare_target(target: torch.Tensor, size) -> torch.Tensor:
        if target.dim() == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.dim() != 3:
            raise ValueError(
                f"damage_target must be [B,H,W] or [B,1,H,W], got {tuple(target.shape)}"
            )
        target = F.interpolate(
            target.unsqueeze(1).float(), size=size, mode="nearest"
        )
        return target[:, 0].long()

    @torch.no_grad()
    def _diverse_init(self, feats: torch.Tensor) -> torch.Tensor:
        """Farthest-point initialization in cosine distance."""
        if self.k == 1:
            return F.normalize(feats.mean(dim=0, keepdim=True), dim=1)

        center = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
        first = int(torch.argmax((feats @ center.t())[:, 0]).item())
        selected = [first]
        min_dist = 1.0 - (feats @ feats[first:first + 1].t())[:, 0]
        for _ in range(1, self.k):
            idx = int(torch.argmax(min_dist).item())
            selected.append(idx)
            dist = 1.0 - (feats @ feats[idx:idx + 1].t())[:, 0]
            min_dist = torch.minimum(min_dist, dist)
        return F.normalize(feats[selected], dim=1)

    @torch.no_grad()
    def _hard_batch_prototypes(
        self,
        feats: torch.Tensor,
        proto: torch.Tensor,
        class_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spherical hard assignment with empty-prototype reinitialization."""
        scores = feats @ proto.t()  # [N, K], cosine similarity
        assign_idx = scores.argmax(dim=1)
        counts = torch.bincount(assign_idx, minlength=self.k).to(feats.dtype)

        centers = []
        # Used so multiple empty clusters do not repeatedly choose the same feature.
        excluded = torch.zeros(feats.shape[0], device=feats.device, dtype=torch.bool)
        working_proto = proto.clone()

        for proto_idx in range(self.k):
            mask = assign_idx == proto_idx
            if bool(mask.any().item()):
                center = F.normalize(
                    feats[mask].mean(dim=0, keepdim=True), dim=1
                )[0]
            else:
                current_scores = feats @ working_proto.t()
                max_similarity = current_scores.max(dim=1).values
                max_similarity = max_similarity.masked_fill(excluded, float("inf"))
                farthest_idx = int(torch.argmin(max_similarity).item())
                center = feats[farthest_idx]
                excluded[farthest_idx] = True
                self.empty_reinit_count[class_idx, proto_idx] += 1
                # Let later empty clusters see this newly selected direction too.
                working_proto[proto_idx] = center
            centers.append(center)

        return F.normalize(torch.stack(centers, dim=0), dim=1), counts

    @torch.no_grad()
    def update_memory(self, z: torch.Tensor, damage_target: torch.Tensor) -> None:
        target = self._prepare_target(damage_target, z.shape[-2:])
        feats_all = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)
        labels_all = target.reshape(-1)
        mode = self.resolved_assignment_mode()

        for class_idx in range(self.num_damage_classes):
            label_value = class_idx + 1
            feats = feats_all[labels_all == label_value]
            if feats.shape[0] < self.k:
                continue

            if feats.shape[0] > self.max_samples_per_class:
                perm = torch.randperm(feats.shape[0], device=feats.device)[
                    : self.max_samples_per_class
                ]
                feats = feats[perm]
            feats = F.normalize(feats, dim=1)

            if not bool(self.prototype_initialized[class_idx].item()):
                self.prototypes[class_idx].copy_(self._diverse_init(feats))
                self.prototype_initialized[class_idx] = True

            proto = F.normalize(self.prototypes[class_idx], dim=1)

            if mode == "mean":
                batch_proto = F.normalize(
                    feats.mean(dim=0, keepdim=True), dim=1
                )
                counts = torch.tensor(
                    [float(feats.shape[0])],
                    device=feats.device,
                    dtype=feats.dtype,
                )
            elif mode == "hard":
                batch_proto, counts = self._hard_batch_prototypes(
                    feats, proto, class_idx
                )
            else:
                raise RuntimeError(f"Unexpected assignment mode: {mode}")

            updated = self.momentum * proto + (1.0 - self.momentum) * batch_proto
            self.prototypes[class_idx].copy_(F.normalize(updated, dim=1))
            self.prototype_usage[class_idx].add_(
                counts.to(self.prototype_usage.dtype)
            )

    def _query(self, z: torch.Tensor):
        # Stable memory snapshot for the current prediction/autograd graph.
        proto = F.normalize(self.prototypes.detach().clone(), dim=-1)
        flat_proto = proto.reshape(
            self.num_damage_classes * self.k, self.proto_dim
        )
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)

        similarity = z_flat @ flat_proto.t()
        attention = torch.softmax(similarity / self.tau, dim=1)
        context = attention @ flat_proto
        context = context.view(
            z.shape[0], z.shape[2], z.shape[3], self.proto_dim
        ).permute(0, 3, 1, 2).contiguous()

        class_similarity = similarity.view(
            -1, self.num_damage_classes, self.k
        ).amax(dim=2)

        if similarity.shape[1] >= 2:
            top2 = torch.topk(similarity, k=2, dim=1).values
            query_margin = (top2[:, 0] - top2[:, 1]).mean()
        else:
            query_margin = similarity.new_tensor(0.0)

        attention_entropy = -(
            attention * torch.log(attention.clamp_min(1e-8))
        ).sum(dim=1).mean()

        return context, class_similarity, query_margin, attention_entropy

    def forward(
        self,
        feature: torch.Tensor,
        building_prob: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
    ):
        # V2 key: direct task-supervised 256-D damage feature space.
        z = F.normalize(self.feature_proj(feature), dim=1)
        all_ready = bool(self.prototype_initialized.all().item())
        active = all_ready and (
            current_iter is None or int(current_iter) >= self.warmup_iters
        )

        class_similarity = None
        query_margin = feature.new_tensor(float("nan"))
        attention_entropy = feature.new_tensor(float("nan"))
        effect_ratio = feature.new_tensor(0.0)

        if active:
            (
                context,
                class_similarity,
                query_margin,
                attention_entropy,
            ) = self._query(z)
            building_prob = F.interpolate(
                building_prob.float(),
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).detach().to(dtype=feature.dtype)
            delta = self.context_proj(context) * building_prob
            correction = self.gamma() * delta
            enhanced = feature + correction
            effect_ratio = (
                correction.norm(dim=1).mean()
                / feature.norm(dim=1).mean().clamp_min(1e-8)
            )
        else:
            enhanced = feature

        if self.training and damage_target is not None:
            self.update_memory(z.detach(), damage_target)

        usage = self.prototype_usage.detach().clone()
        usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
        stats: Dict[str, object] = {
            "proto_active": active,
            "proto_initialized": self.prototype_initialized.detach().clone(),
            "proto_usage_ratio": usage_ratio,
            "proto_empty_reinit_count": self.empty_reinit_count.detach().clone(),
            "proto_gamma": self.gamma().detach(),
            "proto_class_similarity": (
                class_similarity.detach() if class_similarity is not None else None
            ),
            "proto_query_margin": query_margin.detach(),
            "proto_attention_entropy": attention_entropy.detach(),
            "proto_effect_ratio": effect_ratio.detach(),
            "proto_assignment_mode": self.resolved_assignment_mode(),
        }
        return enhanced, stats


class DamageFormer_BGMP_V2(DamageFormer_BG):
    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        proto_k: int = 1,
        proto_momentum: float = 0.99,
        proto_tau: float = 0.1,
        proto_warmup_iters: int = 5000,
        proto_assignment_mode: str = "auto",
    ) -> None:
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
        )
        self.damage_proto_memory = DamagePrototypeMemoryV2(
            in_channels=256,
            prototypes_per_class=proto_k,
            momentum=proto_momentum,
            tau=proto_tau,
            warmup_iters=proto_warmup_iters,
            assignment_mode=proto_assignment_mode,
        )

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
        return_aux: bool = False,
    ):
        final_feat_1, feat_2 = self._encode_and_fuse(pre_data, post_data)

        loc_logits_feat = self.clf_1(final_feat_1)
        p_building = torch.softmax(loc_logits_feat, dim=1)[:, 1:2]
        building_prior = (
            p_building.detach() if self.detach_building_prior else p_building
        )
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        final_feat_2 = self.fusion_layer_3(
            torch.cat([final_feat_1, feat_2], dim=1)
        )
        final_feat_2_roi = final_feat_2 * roi_weight
        final_feat_2_proto, proto_stats = self.damage_proto_memory(
            feature=final_feat_2_roi,
            building_prob=p_building,
            damage_target=damage_target,
            current_iter=current_iter,
        )

        output_loc = F.interpolate(
            loc_logits_feat,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_dam = F.interpolate(
            self.clf_2(final_feat_2_proto),
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if not return_aux:
            return output_loc, output_dam

        aux = {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "final_feat_fused": final_feat_2,
            "final_feat_fused_roi": final_feat_2_roi,
            "final_feat_fused_proto": final_feat_2_proto,
            **proto_stats,
        }
        return output_loc, output_dam, aux
