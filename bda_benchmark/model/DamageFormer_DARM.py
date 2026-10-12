from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .DamageFormer_S8_IA_MLA import DamageFormer_S8_IA_MLA
    from .region_ops import (
        InstanceRecord,
        binary_dilate,
        binary_erode,
        infer_instance_labels,
        resize_instance_map,
    )
else:  # Preserve direct-file smoke tests from the benchmark root.
    from model.DamageFormer_S8_IA_MLA import DamageFormer_S8_IA_MLA
    from model.region_ops import (
        InstanceRecord,
        binary_dilate,
        binary_erode,
        infer_instance_labels,
        resize_instance_map,
    )


class MaskedMultiStatisticPool(nn.Module):
    """Inside/boundary/ring + mean/top-k/attention pooling."""

    def __init__(self, channels: int = 256, hidden: int = 128, topk_ratio: float = 0.1):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1)
        )
        self.region_gate = nn.Sequential(
            nn.Linear(channels * 9, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 9)
        )
        self.out = nn.Sequential(
            nn.Linear(channels, channels), nn.LayerNorm(channels), nn.ReLU(inplace=True)
        )

    @staticmethod
    def _masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = feat[:, mask]
        return x.mean(dim=1) if x.numel() else feat.new_zeros(feat.shape[0])

    def _masked_topk(self, feat: torch.Tensor, score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = feat[:, mask]
        s = score[mask]
        if x.numel() == 0:
            return feat.new_zeros(feat.shape[0])
        k = max(1, int(round(x.shape[1] * self.topk_ratio)))
        idx = torch.topk(s, k=min(k, s.numel()), largest=True).indices
        return x[:, idx].mean(dim=1)

    def _masked_attention(self, feat: torch.Tensor, attn_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = feat[:, mask]
        a = attn_logits[mask]
        if x.numel() == 0:
            return feat.new_zeros(feat.shape[0])
        w = torch.softmax(a, dim=0)
        return (x * w.unsqueeze(0)).sum(dim=1)

    def forward_one(
        self,
        feat_chw: torch.Tensor,
        damage_score_hw: torch.Tensor,
        inside: torch.Tensor,
        boundary: torch.Tensor,
        ring: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_map = self.attn(feat_chw.unsqueeze(0))[0, 0]
        vectors: List[torch.Tensor] = []
        for region in (inside, boundary, ring):
            vectors.extend([
                self._masked_mean(feat_chw, region),
                self._masked_topk(feat_chw, damage_score_hw, region),
                self._masked_attention(feat_chw, attn_map, region),
            ])
        cat = torch.cat(vectors, dim=0)
        weights = torch.softmax(self.region_gate(cat), dim=0)
        stacked = torch.stack(vectors, dim=0)
        pooled = (stacked * weights[:, None]).sum(dim=0)
        return self.out(pooled), weights


class HierarchicalDamageHead(nn.Module):
    """First Intact-vs-Damage, then Damaged-vs-Destroyed."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.exist = nn.Linear(dim, 2)
        self.severity = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        exist = self.exist(x)      # 0 intact, 1 damage
        severity = self.severity(x)  # 0 damaged, 1 destroyed
        pe = torch.softmax(exist, dim=-1)
        ps = torch.softmax(severity, dim=-1)
        p3 = torch.stack([
            pe[:, 0],
            pe[:, 1] * ps[:, 0],
            pe[:, 1] * ps[:, 1],
        ], dim=-1).clamp_min(1e-8)
        logits3 = p3.log()
        return {"exist_logits": exist, "severity_logits": severity, "building_logits3": logits3}


class PrototypeBank(nn.Module):
    """EMA multi-prototype bank for building classes Intact/Damaged/Destroyed."""

    def __init__(self, dim: int = 256, k: int = 3, momentum: float = 0.99):
        super().__init__()
        self.k = int(k)
        self.momentum = float(momentum)
        proto = F.normalize(torch.randn(3, self.k, dim), dim=-1)
        self.register_buffer("prototypes", proto)
        self.register_buffer("usage", torch.zeros(3, self.k))
        self.register_buffer("initialized", torch.zeros(3, self.k, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels_1to3: torch.Tensor) -> None:
        if feats.numel() == 0:
            return
        feats = F.normalize(feats.detach(), dim=-1)
        for cls in (1, 2, 3):
            sel = feats[labels_1to3 == cls]
            if sel.numel() == 0:
                continue
            c = cls - 1
            for f in sel:
                if not self.initialized[c].all():
                    idx = int((~self.initialized[c]).nonzero()[0].item())
                    self.prototypes[c, idx] = f
                    self.initialized[c, idx] = True
                else:
                    sim = self.prototypes[c] @ f
                    idx = int(sim.argmax().item())
                    self.prototypes[c, idx] = F.normalize(
                        self.momentum * self.prototypes[c, idx] + (1 - self.momentum) * f,
                        dim=0,
                    )
                self.usage[c, idx] += 1

    def retrieve(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(feats, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("nd,ckd->nck", q, p)
        flat = sim.reshape(feats.shape[0], -1)
        idx = flat.argmax(dim=-1)
        cls = idx // self.k
        kid = idx % self.k
        chosen = p[cls, kid]
        return chosen, flat.max(dim=-1).values


class InstanceMemoryBank(nn.Module):
    """FIFO instance bank. It stores detached training features only."""

    def __init__(self, dim: int = 256, capacity: int = 4096):
        super().__init__()
        self.capacity = int(capacity)
        self.register_buffer("features", torch.zeros(capacity, dim))
        self.register_buffer("labels", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        feats = F.normalize(feats.detach(), dim=-1)
        for f, y in zip(feats, labels.detach()):
            i = int(self.ptr.item())
            self.features[i] = f
            self.labels[i] = y
            self.valid[i] = True
            self.ptr[...] = (i + 1) % self.capacity

    def topk(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.valid.any():
            n = query.shape[0]
            return query.new_zeros(n, k, query.shape[-1]), query.new_zeros(n, k, dtype=torch.long), query.new_zeros(n, k)
        bank = F.normalize(self.features[self.valid], dim=-1)
        labels = self.labels[self.valid]
        q = F.normalize(query, dim=-1)
        sim = q @ bank.t()
        kk = min(k, bank.shape[0])
        vals, idx = sim.topk(kk, dim=-1)
        nbr = bank[idx]
        lab = labels[idx]
        if kk < k:
            padn = k - kk
            nbr = torch.cat([nbr, nbr.new_zeros(nbr.shape[0], padn, nbr.shape[2])], dim=1)
            lab = torch.cat([lab, lab.new_zeros(lab.shape[0], padn)], dim=1)
            vals = torch.cat([vals, vals.new_zeros(vals.shape[0], padn)], dim=1)
        return nbr, lab, vals


class DamageFormer_DARM(nn.Module):
    """S8 + damage-aware regional MIL + hierarchical heads + optional dual memory."""

    def __init__(
        self,
        topk_ratio: float = 0.1,
        boundary_radius: int = 1,
        ring_radius: int = 3,
        lambda_instance_residual_init: float = 0.0,
        use_prototypes: bool = False,
        proto_k: int = 3,
        use_retrieval: bool = False,
        retrieval_k: int = 5,
        uncertainty_threshold: float = 0.55,
        memory_capacity: int = 4096,
        **s8_kwargs,
    ):
        super().__init__()
        self.base = DamageFormer_S8_IA_MLA(return_dict=True, **s8_kwargs)
        self.pool = MaskedMultiStatisticPool(256, 128, topk_ratio)
        self.instance_head = HierarchicalDamageHead(256)
        self.gamma_inst = nn.Parameter(torch.tensor(float(lambda_instance_residual_init)))
        self.boundary_radius = int(boundary_radius)
        self.ring_radius = int(ring_radius)
        self.use_prototypes = bool(use_prototypes)
        self.use_retrieval = bool(use_retrieval)
        self.retrieval_k = int(retrieval_k)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.prototype_bank = PrototypeBank(256, proto_k) if self.use_prototypes else None
        self.memory_bank = InstanceMemoryBank(256, memory_capacity) if self.use_retrieval else None
        self.memory_fuse = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True)
        )

    def _collect_instances(
        self,
        feat: torch.Tensor,
        pixel_logits: torch.Tensor,
        instance_map: torch.Tensor,
        damage_gt: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        h, w = feat.shape[-2:]
        inst_small = resize_instance_map(instance_map, (h, w))
        score = torch.softmax(F.interpolate(pixel_logits, (h, w), mode="bilinear", align_corners=False), dim=1)[:, 2:4].sum(1)
        if damage_gt is not None:
            gt_small = F.interpolate(damage_gt[:,None].float(), (h,w), mode="nearest")[:,0].long()
            records, labels = infer_instance_labels(inst_small, gt_small)
        else:
            records, labels = [], feat.new_empty(0, dtype=torch.long)
            for b in range(inst_small.shape[0]):
                for iid in range(1, int(inst_small[b].max().item()) + 1):
                    m = inst_small[b] == iid
                    if m.any():
                        records.append(InstanceRecord(b, iid, 0, int(m.sum().item())))
        vectors, weights = [], []
        valid_records = []
        for rec in records:
            inside = inst_small[rec.batch_index] == rec.instance_id
            if not inside.any():
                continue
            dil = binary_dilate(inside, self.ring_radius)
            ero = binary_erode(inside, self.boundary_radius)
            boundary = inside & (~ero)
            ring = dil & (~inside)
            vec, w9 = self.pool.forward_one(feat[rec.batch_index], score[rec.batch_index], inside, boundary, ring)
            vectors.append(vec)
            weights.append(w9)
            valid_records.append(rec)
        if vectors:
            return {
                "records": valid_records,
                "features": torch.stack(vectors),
                "pool_weights": torch.stack(weights),
                "labels": labels[:len(vectors)] if damage_gt is not None else feat.new_empty(0, dtype=torch.long),
                "inst_small": inst_small,
            }
        return {
            "records": [], "features": feat.new_zeros((0, feat.shape[1])),
            "pool_weights": feat.new_zeros((0,9)), "labels": feat.new_empty(0,dtype=torch.long),
            "inst_small": inst_small,
        }

    def _memory_enhance(self, feats: torch.Tensor, logits3: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feats.numel() == 0:
            return {"enhanced_features": feats, "retrieval_gate": feats.new_zeros((0,))}
        p = torch.softmax(logits3, dim=-1)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1) / torch.log(torch.tensor(3.0, device=p.device))
        gate = (entropy > self.uncertainty_threshold).float()
        context = feats.new_zeros(feats.shape)
        if self.use_prototypes and self.prototype_bank is not None:
            proto, _ = self.prototype_bank.retrieve(feats)
            context = context + proto
        if self.use_retrieval and self.memory_bank is not None:
            nbr, _, sims = self.memory_bank.topk(feats, self.retrieval_k)
            w = torch.softmax(sims, dim=-1)
            context = context + (nbr * w[...,None]).sum(1)
        enhanced = feats + gate[:,None] * self.memory_fuse(torch.cat([feats, context], dim=-1))
        return {"enhanced_features": enhanced, "retrieval_gate": gate, "instance_entropy": entropy}

    def _scatter_instance_logits(
        self,
        records: List[InstanceRecord],
        logits3: torch.Tensor,
        inst_small: torch.Tensor,
        out_size: Tuple[int,int],
    ) -> torch.Tensor:
        b, h, w = inst_small.shape
        canvas = logits3.new_zeros((b, 4, h, w))
        counts = logits3.new_zeros((b, 1, h, w))
        for i, rec in enumerate(records):
            m = inst_small[rec.batch_index] == rec.instance_id
            canvas[rec.batch_index, 1:, m] += logits3[i][:,None]
            counts[rec.batch_index, 0, m] += 1
        canvas = canvas / counts.clamp_min(1.0)
        return F.interpolate(canvas, out_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        instance_map: Optional[torch.Tensor] = None,
        damage_gt: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out = self.base(pre_data, post_data, return_dict=True)
        if instance_map is None:
            out.update({
                "instance_features": out["final_feat_damage"].new_zeros((0,256)),
                "instance_labels": out["final_feat_damage"].new_empty(0,dtype=torch.long),
                "gamma_inst": self.gamma_inst,
            })
            return out
        pack = self._collect_instances(out["final_feat_damage"], out["output_dam"], instance_map, damage_gt)
        feats = pack["features"]
        first = self.instance_head(feats) if feats.numel() else {
            "exist_logits": feats.new_zeros((0,2)), "severity_logits": feats.new_zeros((0,2)), "building_logits3": feats.new_zeros((0,3))
        }
        mem = self._memory_enhance(feats, first["building_logits3"])
        second = self.instance_head(mem["enhanced_features"]) if feats.numel() else first
        inst_map_logits = self._scatter_instance_logits(pack["records"], second["building_logits3"], pack["inst_small"], out["output_dam"].shape[-2:])
        out["output_dam_pixel"] = out["output_dam"]
        out["instance_logit_map"] = inst_map_logits
        out["output_dam"] = out["output_dam"] + self.gamma_inst * inst_map_logits
        out["main_dam_logits"] = out["output_dam"]
        out.update({
            "instance_features": feats,
            "instance_features_enhanced": mem["enhanced_features"],
            "instance_labels": pack["labels"],
            "instance_records": pack["records"],
            "pool_weights": pack["pool_weights"],
            "exist_logits": second["exist_logits"],
            "severity_logits": second["severity_logits"],
            "building_logits3": second["building_logits3"],
            "retrieval_gate": mem.get("retrieval_gate", feats.new_zeros((feats.shape[0],))),
            "instance_entropy": mem.get("instance_entropy", feats.new_zeros((feats.shape[0],))),
            "gamma_inst": self.gamma_inst,
        })
        if update_memory and damage_gt is not None and feats.numel():
            labels = pack["labels"]
            if self.prototype_bank is not None:
                self.prototype_bank.update(feats, labels)
            if self.memory_bank is not None:
                self.memory_bank.enqueue(feats, labels)
        return out


DamageFormer = DamageFormer_DARM
