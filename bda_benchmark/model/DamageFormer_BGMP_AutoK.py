# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DamageFormer_BG import DamageFormer_BG
except ImportError:
    from DamageFormer_BG import DamageFormer_BG


class AutoKPrototypeMemory(nn.Module):
    """
    Auto-K prototype memory.

    Key idea:
    - Do not manually set K=1/3/5 for each class.
    - Give each class Kmax prototype slots.
    - Learn gate probabilities over slots.
    - The effective K is determined by learned gate distribution.

    Classes:
    0 -> Intact
    1 -> Damaged
    2 -> Destroyed
    """

    def __init__(
        self,
        in_channels: int = 256,
        proto_dim: int = 256,
        kmax: int = 5,
        momentum: float = 0.99,
        tau: float = 0.1,
        gate_tau: float = 1.0,
        warmup_iters: int = 5000,
        max_samples_per_class: int = 4096,
    ):
        super().__init__()

        self.num_damage_classes = 3
        self.kmax = int(kmax)
        self.proto_dim = int(proto_dim)
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.gate_tau = float(gate_tau)
        self.warmup_iters = int(warmup_iters)
        self.max_samples_per_class = int(max_samples_per_class)

        if self.proto_dim == in_channels:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Conv2d(in_channels, self.proto_dim, kernel_size=1, bias=False)

        proto = F.normalize(
            torch.randn(self.num_damage_classes, self.kmax, self.proto_dim),
            dim=-1,
        )
        self.register_buffer("prototypes", proto)
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.num_damage_classes, dtype=torch.bool),
        )
        self.register_buffer(
            "prototype_usage",
            torch.zeros(self.num_damage_classes, self.kmax),
        )

        # Learnable gate. This is the key part for Auto-K.
        # For each class, softmax over Kmax slots.
        self.gate_logits = nn.Parameter(torch.zeros(self.num_damage_classes, self.kmax))

        # Learnable scale for prototype logits.
        # Use sigmoid to keep it stable.
        self.proto_logit_scale_raw = nn.Parameter(torch.tensor(-2.0))

    def proto_logit_scale(self):
        # Range roughly [0, 0.3]
        return 0.3 * torch.sigmoid(self.proto_logit_scale_raw)

    def gate_probs(self):
        return torch.softmax(self.gate_logits / max(self.gate_tau, 1e-6), dim=1)

    def effective_k(self):
        """
        Effective K = exp(entropy).
        If gate=[1,0,0,0,0], effective K ≈ 1.
        If gate=[0.2,0.2,0.2,0.2,0.2], effective K ≈ 5.
        """
        p = self.gate_probs()
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
        return torch.exp(entropy)

    @staticmethod
    def _resize_target(target: torch.Tensor, size):
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target[:, 0]
            else:
                target = torch.argmax(target, dim=1)

        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=size,
            mode="nearest",
        )[:, 0].long()
        return target

    def _active(self, current_iter: Optional[int]):
        ready = bool(self.prototype_initialized.all().item())
        if current_iter is None:
            return ready
        return ready and int(current_iter) >= self.warmup_iters

    @torch.no_grad()
    def _diverse_init(self, feats: torch.Tensor):
        feats = F.normalize(feats, dim=1)

        if feats.shape[0] < self.kmax:
            center = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
            return center.repeat(self.kmax, 1)

        center = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
        first = int(torch.argmax((feats @ center.t())[:, 0]).item())
        selected = [first]

        min_dist = 1.0 - feats @ feats[first:first + 1].t()
        min_dist = min_dist[:, 0]

        for _ in range(1, self.kmax):
            nxt = int(torch.argmax(min_dist).item())
            selected.append(nxt)
            dist = 1.0 - feats @ feats[nxt:nxt + 1].t()
            min_dist = torch.minimum(min_dist, dist[:, 0])

        return F.normalize(feats[selected], dim=1)

    @torch.no_grad()
    def update_memory(self, z: torch.Tensor, damage_target: torch.Tensor):
        target = self._resize_target(damage_target, z.shape[-2:])

        feats_all = z.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)
        labels_all = target.reshape(-1)

        gate = self.gate_probs().detach()

        for c in range(self.num_damage_classes):
            label_value = c + 1
            feats = feats_all[labels_all == label_value]

            if feats.shape[0] < 8:
                continue

            if feats.shape[0] > self.max_samples_per_class:
                idx = torch.randperm(feats.shape[0], device=feats.device)[: self.max_samples_per_class]
                feats = feats[idx]

            feats = F.normalize(feats, dim=1)

            if not bool(self.prototype_initialized[c].item()):
                init_proto = self._diverse_init(feats)
                self.prototypes[c].copy_(init_proto)
                self.prototype_initialized[c] = True

            proto = F.normalize(self.prototypes[c], dim=1)

            # Gate-aware assignment.
            # Slots with larger learned gate get more assignment.
            sim = feats @ proto.t()
            score = sim / max(self.tau, 1e-6) + torch.log(gate[c].clamp_min(1e-6))[None, :]
            assign = torch.softmax(score, dim=1)

            mass = assign.sum(dim=0)
            batch_proto = proto.clone()

            valid = mass > 1e-6
            if bool(valid.any().item()):
                new_proto = (assign.t() @ feats) / mass.clamp_min(1e-6)[:, None]
                new_proto = F.normalize(new_proto, dim=1)
                batch_proto[valid] = new_proto[valid]

                updated = self.momentum * proto + (1.0 - self.momentum) * batch_proto
                updated = F.normalize(updated, dim=1)
                self.prototypes[c].copy_(updated)

                self.prototype_usage[c].add_(mass.to(self.prototype_usage.dtype))

    def query_logits(self, z: torch.Tensor):
        b, d, h, w = z.shape

        z_flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        z_flat = F.normalize(z_flat, dim=1)

        proto = F.normalize(self.prototypes.detach().clone(), dim=-1)
        sim = torch.einsum("nd,ckd->nck", z_flat, proto)

        gate = self.gate_probs()
        slot_score = sim / max(self.tau, 1e-6) + torch.log(gate.clamp_min(1e-6))[None, :, :]

        # Mixture over prototype slots.
        proto_logits = torch.logsumexp(slot_score, dim=2)

        proto_logits = proto_logits.view(b, h, w, self.num_damage_classes)
        proto_logits = proto_logits.permute(0, 3, 1, 2).contiguous()

        return proto_logits, sim

    def forward(
        self,
        feature: torch.Tensor,
        damage_target: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
    ) -> Dict[str, object]:

        z = F.normalize(self.feature_proj(feature), dim=1)

        active = self._active(current_iter)

        proto_logits = None
        sim = None

        if active:
            proto_logits, sim = self.query_logits(z)

        if self.training and damage_target is not None:
            self.update_memory(z.detach(), damage_target)

        usage = self.prototype_usage.detach()
        usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return {
            "proto_active": active,
            "proto_class_logits": proto_logits,
            "proto_similarity": sim.detach() if sim is not None else None,
            "proto_initialized": self.prototype_initialized.detach().clone(),
            "proto_usage_ratio": usage_ratio.detach().clone(),
            "proto_gate_probs": self.gate_probs().detach().clone(),
            "proto_effective_k": self.effective_k().detach().clone(),
            "proto_logit_scale": self.proto_logit_scale().detach().clone(),
        }


class DamageFormer_BGMP_AutoK(DamageFormer_BG):
    """
    S3V4-AutoK-Logit:
    - Building prior from S1.
    - Auto-K prototype memory.
    - Prototype logits are fused into final damage logits.
    - No prototype feature residual.
    - No manual K=1,3,3.
    """

    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        use_pretrained_swin: bool = True,
        proto_dim: int = 256,
        proto_kmax: int = 5,
        proto_momentum: float = 0.99,
        proto_tau: float = 0.1,
        proto_gate_tau: float = 1.0,
        proto_warmup_iters: int = 5000,
    ):
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=use_pretrained_swin,
        )

        self.damage_proto_memory = AutoKPrototypeMemory(
            in_channels=256,
            proto_dim=proto_dim,
            kmax=proto_kmax,
            momentum=proto_momentum,
            tau=proto_tau,
            gate_tau=proto_gate_tau,
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

        dam_logits_feat = self.clf_2(final_feat_2_roi)

        proto_stats = self.damage_proto_memory(
            final_feat_2_roi,
            damage_target=damage_target,
            current_iter=current_iter,
        )

        proto_logits = proto_stats.get("proto_class_logits", None)
        if proto_logits is not None:
            scale = self.damage_proto_memory.proto_logit_scale()
            dam_logits_feat = dam_logits_feat.clone()
            dam_logits_feat[:, 1:4] = dam_logits_feat[:, 1:4] + scale * proto_logits

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
            **proto_stats,
        }

        return output_loc, output_dam, aux
