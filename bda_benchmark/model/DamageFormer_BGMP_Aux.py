# -*- coding: utf-8 -*-
"""
S3V3 / S3V4:
Building-guided Prototype Auxiliary Supervision.

Key difference from old BGMP:
- Old: feature + gamma * prototype_context -> damage classifier
- New: no prototype residual injection.
       Prototype only produces auxiliary logits and auxiliary CE loss.
"""

from __future__ import annotations

from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_BG import DamageFormer_BG
except ImportError:
    from DamageFormer_BG import DamageFormer_BG


def parse_proto_k_list(proto_k_list: str) -> List[int]:
    vals = [int(x.strip()) for x in proto_k_list.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(
            f"proto_k_list must contain exactly 3 values for Intact,Damaged,Destroyed, got {proto_k_list}"
        )
    if min(vals) < 1:
        raise ValueError(f"Each K must be >= 1, got {vals}")
    return vals


class DamagePrototypeAuxMemory(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        proto_dim: int = 256,
        proto_k_list: str = "1,3,3",
        momentum: float = 0.99,
        tau: float = 0.1,
        warmup_iters: int = 5000,
        assign_mode: str = "sinkhorn",
        max_samples_per_class: int = 4096,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
    ):
        super().__init__()

        self.num_damage_classes = 3
        self.k_list = parse_proto_k_list(proto_k_list)
        self.max_k = max(self.k_list)

        self.in_channels = int(in_channels)
        self.proto_dim = int(proto_dim)
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.warmup_iters = int(warmup_iters)
        self.assign_mode = assign_mode.lower()
        self.max_samples_per_class = int(max_samples_per_class)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)

        if self.assign_mode not in {"hard", "soft", "sinkhorn"}:
            raise ValueError(f"Unknown assign_mode: {assign_mode}")

        if self.proto_dim == self.in_channels:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Conv2d(self.in_channels, self.proto_dim, kernel_size=1, bias=False)

        proto = F.normalize(
            torch.randn(self.num_damage_classes, self.max_k, self.proto_dim),
            dim=-1,
        )
        self.register_buffer("prototypes", proto)
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.num_damage_classes, dtype=torch.bool),
        )
        self.register_buffer(
            "prototype_usage",
            torch.zeros(self.num_damage_classes, self.max_k),
        )

        valid = torch.zeros(self.num_damage_classes, self.max_k, dtype=torch.bool)
        for c, k in enumerate(self.k_list):
            valid[c, :k] = True
        self.register_buffer("prototype_valid", valid)

    @staticmethod
    def _prepare_target(target: torch.Tensor, size) -> torch.Tensor:
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target[:, 0]
            else:
                target = torch.argmax(target, dim=1)

        if target.dim() != 3:
            raise ValueError(
                f"damage_target must be [B,H,W], [B,1,H,W], or [B,C,H,W], got {tuple(target.shape)}"
            )

        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=size,
            mode="nearest",
        )
        return target[:, 0].long()

    def _is_active(self, current_iter: Optional[int]) -> bool:
        all_ready = bool(self.prototype_initialized.all().item())
        if current_iter is None:
            return all_ready
        return all_ready and int(current_iter) >= self.warmup_iters

    @torch.no_grad()
    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores

        scaled = scores.float() / self.sinkhorn_epsilon
        scaled = scaled - scaled.max(dim=1, keepdim=True).values

        q = torch.exp(scaled).t()
        q = q / q.sum().clamp_min(1e-12)

        k, n = q.shape
        for _ in range(self.sinkhorn_iters):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q / float(k)
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / float(n)

        q = q * float(n)
        return q.t().to(dtype=scores.dtype)

    @torch.no_grad()
    def _diverse_init(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        feats = F.normalize(feats, dim=1)

        if k == 1:
            center = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
            return center

        center = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
        first_idx = int(torch.argmax((feats @ center.t())[:, 0]).item())
        selected = [first_idx]

        min_dist = 1.0 - feats @ feats[first_idx:first_idx + 1].t()
        min_dist = min_dist[:, 0]

        for _ in range(1, min(k, feats.shape[0])):
            next_idx = int(torch.argmax(min_dist).item())
            selected.append(next_idx)
            dist = 1.0 - feats @ feats[next_idx:next_idx + 1].t()
            min_dist = torch.minimum(min_dist, dist[:, 0])

        while len(selected) < k:
            selected.append(selected[-1])

        return F.normalize(feats[selected[:k]], dim=1)

    @torch.no_grad()
    def update_memory(self, z: torch.Tensor, damage_target: torch.Tensor) -> None:
        target = self._prepare_target(damage_target, z.shape[-2:])

        feats_all = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)
        labels_all = target.reshape(-1)

        for class_idx in range(self.num_damage_classes):
            label_value = class_idx + 1
            k_c = self.k_list[class_idx]

            feats = feats_all[labels_all == label_value]
            if feats.shape[0] < k_c:
                continue

            if feats.shape[0] > self.max_samples_per_class:
                perm = torch.randperm(feats.shape[0], device=feats.device)[: self.max_samples_per_class]
                feats = feats[perm]

            feats = F.normalize(feats, dim=1)

            if not bool(self.prototype_initialized[class_idx].item()):
                init_proto = self._diverse_init(feats, k_c)
                self.prototypes[class_idx, :k_c].copy_(init_proto)
                self.prototype_initialized[class_idx] = True

            proto = F.normalize(self.prototypes[class_idx, :k_c], dim=1)
            scores = feats @ proto.t()

            if k_c == 1:
                assignment = torch.ones(feats.shape[0], 1, device=feats.device, dtype=feats.dtype)
            elif self.assign_mode == "hard":
                idx = scores.argmax(dim=1)
                assignment = F.one_hot(idx, num_classes=k_c).to(dtype=feats.dtype)
            elif self.assign_mode == "soft":
                assignment = torch.softmax(scores / self.tau, dim=1)
            else:
                assignment = self._sinkhorn(scores)

            mass = assignment.sum(dim=0)
            valid = mass > 1e-6

            if bool(valid.any().item()):
                batch_proto = proto.clone()
                new_proto = (assignment.t() @ feats) / mass.clamp_min(1e-6)[:, None]
                new_proto = F.normalize(new_proto, dim=1)
                batch_proto[valid] = new_proto[valid]

                updated = self.momentum * proto + (1.0 - self.momentum) * batch_proto
                updated = F.normalize(updated, dim=1)
                self.prototypes[class_idx, :k_c].copy_(updated)

                self.prototype_usage[class_idx, :k_c].add_(
                    mass.to(dtype=self.prototype_usage.dtype)
                )

    def _query_logits(self, z: torch.Tensor):
        b, d, h, w = z.shape

        z_flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        z_flat = F.normalize(z_flat, dim=1)

        proto = F.normalize(self.prototypes.detach().clone(), dim=-1)

        sim = torch.einsum("nd,ckd->nck", z_flat, proto)

        active_mask = self.prototype_valid & self.prototype_initialized[:, None]
        sim = sim.masked_fill(~active_mask[None, :, :], -1e4)

        class_similarity = sim.amax(dim=2)
        proto_logits = torch.logsumexp(sim / self.tau, dim=2)

        proto_logits = proto_logits.view(b, h, w, self.num_damage_classes)
        proto_logits = proto_logits.permute(0, 3, 1, 2).contiguous()

        return proto_logits, class_similarity

    def forward(
        self,
        feature: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
    ) -> Dict[str, object]:

        z = F.normalize(self.feature_proj(feature), dim=1)

        active = self._is_active(current_iter)

        proto_logits = None
        class_similarity = None

        if active:
            proto_logits, class_similarity = self._query_logits(z)

        if self.training and damage_target is not None:
            self.update_memory(z.detach(), damage_target)

        usage = self.prototype_usage.detach() * self.prototype_valid.float()
        usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return {
            "proto_active": active,
            "proto_initialized": self.prototype_initialized.detach().clone(),
            "proto_usage_ratio": usage_ratio.detach().clone(),
            "proto_class_logits": proto_logits,
            "proto_class_similarity": class_similarity.detach() if class_similarity is not None else None,
            "prototype_valid": self.prototype_valid.detach().clone(),
        }


class DamageFormer_BGMP_Aux(DamageFormer_BG):
    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        proto_dim: int = 256,
        proto_k_list: str = "1,3,3",
        proto_momentum: float = 0.99,
        proto_tau: float = 0.1,
        proto_warmup_iters: int = 5000,
        proto_assign_mode: str = "sinkhorn",
    ):
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
        )

        self.damage_proto_memory = DamagePrototypeAuxMemory(
            in_channels=256,
            proto_dim=proto_dim,
            proto_k_list=proto_k_list,
            momentum=proto_momentum,
            tau=proto_tau,
            warmup_iters=proto_warmup_iters,
            assign_mode=proto_assign_mode,
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

        final_feat_2 = self.fusion_layer_3(
            torch.cat([final_feat_1, feat_2], dim=1)
        )
        final_feat_2_roi = final_feat_2 * roi_weight

        # Main damage prediction: NO prototype residual.
        dam_logits_feat = self.clf_2(final_feat_2_roi)

        proto_stats = self.damage_proto_memory(
            feature=final_feat_2_roi,
            damage_target=damage_target,
            current_iter=current_iter,
        )


        # -------------------------------------------------------------
        # Compatibility diagnostics for old BGMP training logger.
        # New Aux model has no feature residual, so gamma/effect_ratio = 0.
        # -------------------------------------------------------------
        proto_logits = proto_stats.get("proto_class_logits", None)
        class_similarity = proto_stats.get("proto_class_similarity", None)

        zero_diag = final_feat_2_roi.new_tensor(0.0)

        proto_margin = zero_diag
        proto_attention_entropy = zero_diag
        proto_cls_acc = zero_diag

        if class_similarity is not None:
            with torch.no_grad():
                sim = class_similarity
                if sim.ndim == 2 and sim.shape[1] >= 2:
                    top2 = torch.topk(sim, k=2, dim=1).values
                    proto_margin = (top2[:, 0] - top2[:, 1]).mean()

                    prob = torch.softmax(
                        sim / max(float(self.damage_proto_memory.tau), 1e-6),
                        dim=1,
                    )
                    proto_attention_entropy = -(
                        prob * torch.log(prob.clamp_min(1e-12))
                    ).sum(dim=1).mean()

        if proto_logits is not None and damage_target is not None:
            with torch.no_grad():
                proto_target = DamagePrototypeAuxMemory._prepare_target(
                    damage_target,
                    proto_logits.shape[-2:],
                )
                valid = (proto_target >= 1) & (proto_target <= 3)
                if bool(valid.any().item()):
                    pred = proto_logits.argmax(dim=1)
                    proto_cls_acc = (
                        pred[valid] == (proto_target[valid] - 1)
                    ).float().mean()

        proto_stats["proto_gamma"] = zero_diag
        proto_stats["proto_effect_ratio"] = zero_diag
        proto_stats["proto_margin"] = proto_margin.detach()
        proto_stats["proto_attention_entropy"] = proto_attention_entropy.detach()
        proto_stats["proto_cls_acc"] = proto_cls_acc.detach()

        output_loc = F.interpolate(
            loc_logits_feat,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_dam = F.interpolate(
            dam_logits_feat,
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
            # keep this key for compatibility with old diagnostics
            "final_feat_fused_proto": final_feat_2_roi,
            **proto_stats,
        }
        return output_loc, output_dam, aux
