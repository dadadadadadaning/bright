# -*- coding: utf-8 -*-
"""Building-guided multi-prototype DamageFormer.

V1 intentionally has no prototype auxiliary loss. Prototypes are class-conditioned
memory buffers updated by class-wise Sinkhorn assignment + EMA, and are queried to
provide a small residual context before the original damage classifier.
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


class DamagePrototypeMemory(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        proto_dim: int = 128,
        prototypes_per_class: int = 3,
        momentum: float = 0.99,
        tau: float = 0.1,
        warmup_iters: int = 5000,
        max_samples_per_class: int = 4096,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
    ):
        super().__init__()
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1")
        self.num_damage_classes = 3  # label 1/2/3: intact/damaged/destroyed
        self.proto_dim = int(proto_dim)
        self.k = int(prototypes_per_class)
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.warmup_iters = int(warmup_iters)
        self.max_samples_per_class = int(max_samples_per_class)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)

        self.feature_proj = nn.Conv2d(in_channels, self.proto_dim, 1, bias=False)
        self.context_proj = nn.Sequential(
            nn.Conv2d(self.proto_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

        proto = F.normalize(torch.randn(self.num_damage_classes, self.k, self.proto_dim), dim=-1)
        self.register_buffer("prototypes", proto)
        self.register_buffer("prototype_initialized", torch.zeros(self.num_damage_classes, dtype=torch.bool))
        self.register_buffer("prototype_usage", torch.zeros(self.num_damage_classes, self.k))

    def gamma(self) -> torch.Tensor:
        return 0.1 * torch.sigmoid(self.gamma_raw)

    @staticmethod
    def _prepare_target(target: torch.Tensor, size) -> torch.Tensor:
        if target.dim() == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.dim() != 3:
            raise ValueError(f"damage_target must be [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")
        target = F.interpolate(target.unsqueeze(1).float(), size=size, mode="nearest")
        return target[:, 0].long()

    @torch.no_grad()
    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: [N, K] cosine similarity
        scaled = scores.float() / self.sinkhorn_epsilon
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        q = torch.exp(scaled).t()  # [K, N]
        q = q / q.sum().clamp_min(1e-12)
        k, n = q.shape
        for _ in range(self.sinkhorn_iters):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q / float(k)
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / float(n)
        return (q * float(n)).t().to(dtype=scores.dtype)

    @torch.no_grad()
    def _diverse_init(self, feats: torch.Tensor) -> torch.Tensor:
        # Farthest-point style initialization in cosine distance.
        center = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
        first = int(torch.argmax((feats @ center.t())[:, 0]).item())
        selected = [first]
        min_dist = 1.0 - feats @ feats[first:first + 1].t()
        min_dist = min_dist[:, 0]
        for _ in range(1, self.k):
            idx = int(torch.argmax(min_dist).item())
            selected.append(idx)
            dist = 1.0 - feats @ feats[idx:idx + 1].t()
            min_dist = torch.minimum(min_dist, dist[:, 0])
        return F.normalize(feats[selected], dim=1)

    @torch.no_grad()
    def update_memory(self, z: torch.Tensor, damage_target: torch.Tensor) -> None:
        target = self._prepare_target(damage_target, z.shape[-2:])
        feats_all = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)
        labels_all = target.reshape(-1)

        for class_idx in range(self.num_damage_classes):
            label_value = class_idx + 1
            feats = feats_all[labels_all == label_value]
            if feats.shape[0] < self.k:
                continue
            if feats.shape[0] > self.max_samples_per_class:
                perm = torch.randperm(feats.shape[0], device=feats.device)[: self.max_samples_per_class]
                feats = feats[perm]
            feats = F.normalize(feats, dim=1)

            if not bool(self.prototype_initialized[class_idx].item()):
                self.prototypes[class_idx].copy_(self._diverse_init(feats))
                self.prototype_initialized[class_idx] = True

            proto = F.normalize(self.prototypes[class_idx], dim=1)
            scores = feats @ proto.t()
            assignment = self._sinkhorn(scores)
            mass = assignment.sum(dim=0).clamp_min(1e-6)
            batch_proto = (assignment.t() @ feats) / mass[:, None]
            batch_proto = F.normalize(batch_proto, dim=1)

            updated = self.momentum * proto + (1.0 - self.momentum) * batch_proto
            self.prototypes[class_idx].copy_(F.normalize(updated, dim=1))
            self.prototype_usage[class_idx].add_(assignment.sum(dim=0).to(self.prototype_usage.dtype))

    def _query(self, z: torch.Tensor):
        # Clone is deliberate: memory may be updated later in the same forward.
        # Autograd then keeps a stable prototype snapshot for gradients wrt z.
        proto = F.normalize(self.prototypes.detach().clone(), dim=-1)
        flat_proto = proto.reshape(self.num_damage_classes * self.k, self.proto_dim)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)

        similarity = z_flat @ flat_proto.t()
        attention = torch.softmax(similarity / self.tau, dim=1)
        context = attention @ flat_proto
        context = context.view(z.shape[0], z.shape[2], z.shape[3], self.proto_dim)
        context = context.permute(0, 3, 1, 2).contiguous()

        class_similarity = similarity.view(-1, self.num_damage_classes, self.k).amax(dim=2)
        return context, class_similarity

    def forward(
        self,
        feature: torch.Tensor,
        building_prob: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
    ):
        z = F.normalize(self.feature_proj(feature), dim=1)
        all_ready = bool(self.prototype_initialized.all().item())
        active = all_ready and (current_iter is None or int(current_iter) >= self.warmup_iters)

        class_similarity = None
        if active:
            context, class_similarity = self._query(z)
            building_prob = F.interpolate(
                building_prob.float(), size=feature.shape[-2:], mode="bilinear", align_corners=False
            ).detach().to(dtype=feature.dtype)
            delta = self.context_proj(context) * building_prob
            enhanced = feature + self.gamma() * delta
        else:
            enhanced = feature

        # The query above uses a cloned snapshot, so the current batch cannot rewrite
        # the prototype tensor that autograd needs for this prediction.
        if self.training and damage_target is not None:
            self.update_memory(z.detach(), damage_target)

        usage = self.prototype_usage.detach().clone()
        usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
        stats: Dict[str, object] = {
            "proto_active": active,
            "proto_initialized": self.prototype_initialized.detach().clone(),
            "proto_usage_ratio": usage_ratio,
            "proto_gamma": self.gamma().detach(),
            "proto_class_similarity": class_similarity.detach() if class_similarity is not None else None,
        }
        return enhanced, stats


class DamageFormer_BGMP(DamageFormer_BG):
    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        proto_k: int = 3,
        proto_dim: int = 128,
        proto_momentum: float = 0.99,
        proto_tau: float = 0.1,
        proto_warmup_iters: int = 5000,
    ):
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
        )
        self.damage_proto_memory = DamagePrototypeMemory(
            in_channels=256,
            proto_dim=proto_dim,
            prototypes_per_class=proto_k,
            momentum=proto_momentum,
            tau=proto_tau,
            warmup_iters=proto_warmup_iters,
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
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        final_feat_2 = self.fusion_layer_3(torch.cat([final_feat_1, feat_2], dim=1))
        final_feat_2_roi = final_feat_2 * roi_weight
        final_feat_2_proto, proto_stats = self.damage_proto_memory(
            feature=final_feat_2_roi,
            building_prob=p_building,
            damage_target=damage_target,
            current_iter=current_iter,
        )

        output_loc = F.interpolate(loc_logits_feat, size=pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_dam = F.interpolate(self.clf_2(final_feat_2_proto), size=post_data.shape[-2:], mode="bilinear", align_corners=False)

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
