from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t

from .priority_region_ops import (
    InstanceRecord,
    binary_dilate,
    binary_erode,
    collect_instance_records,
    resize_instance_map,
)


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        maps: List[torch.Tensor] = []
        for layer in self.backbone.features:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                maps.append(x)
        if len(maps) != 5:
            raise RuntimeError(f"Expected 5 Swin maps, got {len(maps)}")
        return maps


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class StageDifferenceBlock(nn.Module):
    def __init__(self, in_channels: int, diff_channels: int):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(in_channels, diff_channels, 1, bias=False), nn.BatchNorm2d(diff_channels), nn.ReLU(True))
        self.sar = nn.Sequential(nn.Conv2d(in_channels, diff_channels, 1, bias=False), nn.BatchNorm2d(diff_channels), nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(2 * diff_channels, diff_channels, 3, padding=1, bias=False), nn.BatchNorm2d(diff_channels), nn.ReLU(True))

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        r, s = self.rgb(rgb), self.sar(sar)
        return self.fuse(torch.cat([torch.abs(r - s), r * s], dim=1))


class ControlledHCDM(nn.Module):
    STAGE_CHANNELS = (96, 192, 384, 768)

    def __init__(
        self,
        diff_channels: int = 64,
        out_channels: int = 256,
        roi_alpha: float = 0.1,
        gamma_mode: Literal["fixed_zero", "free", "positive"] = "positive",
        gamma_init: float = 0.0025,
        gamma_max: float = 0.1,
        detach_building_prior: bool = True,
    ):
        super().__init__()
        self.roi_alpha = float(roi_alpha)
        self.detach_building_prior = bool(detach_building_prior)
        self.gamma_mode = gamma_mode
        self.gamma_max = float(gamma_max)
        self.blocks = nn.ModuleList([StageDifferenceBlock(c, diff_channels) for c in self.STAGE_CHANNELS])
        self.scale_gate = nn.Sequential(
            nn.Conv2d(4 * diff_channels, diff_channels, 1, bias=False),
            nn.BatchNorm2d(diff_channels), nn.ReLU(True), nn.Conv2d(diff_channels, 4, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(diff_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )
        if gamma_mode == "fixed_zero":
            self.register_buffer("gamma_buffer", torch.zeros(()))
        elif gamma_mode == "free":
            self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))
        elif gamma_mode == "positive":
            target = min(max(float(gamma_init) / max(gamma_max, 1e-8), 1e-6), 1 - 1e-6)
            raw = math.log(target / (1 - target))
            self.gamma_raw = nn.Parameter(torch.tensor(raw))
        else:
            raise ValueError(f"Unsupported gamma_mode={gamma_mode}")

    def gamma(self) -> torch.Tensor:
        if self.gamma_mode == "fixed_zero":
            return self.gamma_buffer
        if self.gamma_mode == "free":
            return self.gamma_raw
        return torch.sigmoid(self.gamma_raw) * self.gamma_max

    def forward(self, rgb_stages, sar_stages, building_prob, target_size):
        prior = building_prob.detach() if self.detach_building_prior else building_prob
        diffs, diag = [], {}
        for idx, (block, rgb, sar) in enumerate(zip(self.blocks, rgb_stages, sar_stages), 1):
            d = block(rgb, sar)
            p = F.interpolate(prior, size=d.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
            d = d * (self.roi_alpha + (1 - self.roi_alpha) * p)
            diag[f"hcdm_stage{idx}_abs_mean"] = d.detach().abs().mean()
            d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False) if d.shape[-2:] != target_size else d
            diffs.append(d)
        cat = torch.cat(diffs, dim=1)
        weights = torch.softmax(self.scale_gate(cat), dim=1)
        fused = sum(diffs[i] * weights[:, i:i+1] for i in range(4))
        fused = self.out_proj(fused)
        diag["scale_weights"] = weights
        diag["gamma_hcdm"] = self.gamma().detach()
        return fused, diag


class AlignmentProjection(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(in_channels, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.sar = nn.Sequential(nn.Conv2d(in_channels, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True))

    def forward(self, r, s):
        return self.rgb(r), self.sar(s)


class InstancePooler(nn.Module):
    def __init__(self, channels: int = 256, topk_ratio: float = 0.1):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.mean_topk_fuse = nn.Sequential(nn.Linear(2 * channels, channels), nn.LayerNorm(channels), nn.ReLU(True))
        self.area_fuse = nn.Sequential(nn.Linear(3 * channels, channels), nn.LayerNorm(channels), nn.ReLU(True))

    @staticmethod
    def masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return feat.new_zeros(feat.shape[0])
        return feat[:, mask].mean(dim=1)

    def masked_topk(self, feat: torch.Tensor, score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return feat.new_zeros(feat.shape[0])
        f, s = feat[:, mask], score[mask]
        k = max(1, int(math.ceil(s.numel() * self.topk_ratio)))
        idx = s.topk(min(k, s.numel())).indices
        return f[:, idx].mean(dim=1)

    def forward_one(self, feat, score, inside, boundary, ring, pool_mode: str, area_mode: str):
        mean = self.masked_mean(feat, inside)
        if pool_mode == "mean":
            inside_vec = mean
        elif pool_mode == "mean_topk":
            topk = self.masked_topk(feat, score, inside)
            inside_vec = self.mean_topk_fuse(torch.cat([mean, topk], dim=0))
        else:
            raise ValueError(pool_mode)
        if area_mode == "inside":
            return inside_vec
        boundary_vec = self.masked_mean(feat, boundary)
        ring_vec = self.masked_mean(feat, ring) if area_mode == "inside_boundary_ring" else feat.new_zeros(feat.shape[0])
        return self.area_fuse(torch.cat([inside_vec, boundary_vec, ring_vec], dim=0))


class FlatInstanceHead(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(True), nn.Linear(dim, 3))

    def forward(self, x):
        return {"building_logits3": self.classifier(x)}


class HierarchicalInstanceHead(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(True))
        self.exist = nn.Linear(dim, 2)
        self.severity = nn.Linear(dim, 2)

    def forward(self, x):
        h = self.shared(x)
        exist, severity = self.exist(h), self.severity(h)
        pe = torch.softmax(exist, -1)
        ps = torch.softmax(severity, -1)
        p3 = torch.stack([pe[:, 0], pe[:, 1] * ps[:, 0], pe[:, 1] * ps[:, 1]], dim=-1)
        return {"exist_logits": exist, "severity_logits": severity, "building_logits3": torch.log(p3.clamp_min(1e-8))}


class DamageFormerPriorityModular(nn.Module):
    def __init__(
        self,
        gamma_mode: str = "positive",
        gamma_init: float = 0.0025,
        gamma_max: float = 0.1,
        diff_channels: int = 64,
        align_dim: int = 128,
        roi_alpha: float = 0.1,
        instance_enabled: bool = False,
        pool_mode: str = "mean",
        area_mode: str = "inside",
        hierarchical_head: bool = False,
        topk_ratio: float = 0.1,
        boundary_radius: int = 2,
        ring_radius: int = 4,
        use_pretrained_swin: bool = True,
    ):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if use_pretrained_swin else None
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_t(weights=weights))
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_t(weights=weights))
        self.fusion_layer_1 = ResBlock(1440, 256)
        self.fusion_layer_2 = ResBlock(1440, 256)
        self.fusion_layer_3 = ResBlock(512, 256)
        self.clf_1 = nn.Conv2d(256, 2, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)
        self.hcdm = ControlledHCDM(diff_channels, 256, roi_alpha, gamma_mode, gamma_init, gamma_max)
        self.ia_stage3 = AlignmentProjection(384, align_dim)
        self.ia_stage4 = AlignmentProjection(768, align_dim)
        self.instance_enabled = bool(instance_enabled)
        self.pool_mode, self.area_mode = pool_mode, area_mode
        self.boundary_radius, self.ring_radius = int(boundary_radius), int(ring_radius)
        if self.instance_enabled:
            self.pooler = InstancePooler(256, topk_ratio)
            self.instance_head = HierarchicalInstanceHead(256) if hierarchical_head else FlatInstanceHead(256)

    @staticmethod
    def _to_nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _aggregate(stages, fusion):
        size = stages[0].shape[-2:]
        xs = [stages[0]] + [F.interpolate(x, size=size, mode="bilinear", align_corners=False) for x in stages[1:]]
        return fusion(torch.cat([xs[3], xs[2], xs[1], xs[0]], dim=1))

    def _instance_forward(self, feat, output_dam, instance_map, damage_gt):
        size = feat.shape[-2:]
        inst = resize_instance_map(instance_map, size)
        gt_small = None if damage_gt is None else F.interpolate(damage_gt[:, None].float(), size=size, mode="nearest")[:, 0].long()
        score = torch.softmax(F.interpolate(output_dam, size=size, mode="bilinear", align_corners=False), 1)[:, 2:4].sum(1)
        records = collect_instance_records(inst, gt_small)
        vectors, valid_records = [], []
        for rec in records:
            inside = inst[rec.batch_index] == rec.instance_id
            if not inside.any():
                continue
            eroded = binary_erode(inside, self.boundary_radius)
            boundary = inside & ~eroded
            dilated = binary_dilate(inside, self.ring_radius)
            ring = dilated & ~inside
            vec = self.pooler.forward_one(
                feat[rec.batch_index], score[rec.batch_index], inside, boundary, ring,
                self.pool_mode, self.area_mode,
            )
            vectors.append(vec)
            valid_records.append(rec)
        if not vectors:
            z = feat.new_zeros((0, feat.shape[1]))
            return {"instance_features": z, "instance_labels": feat.new_empty((0,), dtype=torch.long), "instance_records": [], "building_logits3": feat.new_zeros((0, 3))}
        vectors_t = torch.stack(vectors)
        head = self.instance_head(vectors_t)
        labels = torch.tensor([r.label for r in valid_records], device=feat.device, dtype=torch.long)
        return {"instance_features": vectors_t, "instance_labels": labels, "instance_records": valid_records, **head}

    def forward(self, pre_data, post_data, instance_map: Optional[torch.Tensor] = None, damage_gt: Optional[torch.Tensor] = None):
        pre_maps, post_maps = self.encoder_1(pre_data), self.encoder_2(post_data)
        rgb = [self._to_nchw(x) for x in pre_maps[1:5]]
        sar = [self._to_nchw(x) for x in post_maps[1:5]]
        fr, fs = self._aggregate(rgb, self.fusion_layer_1), self._aggregate(sar, self.fusion_layer_2)
        loc_small = self.clf_1(fr)
        building_prob = torch.softmax(loc_small, 1)[:, 1:2]
        base = self.fusion_layer_3(torch.cat([fr, fs], 1))
        diff, diag = self.hcdm(rgb, sar, building_prob, base.shape[-2:])
        damage_feat = base + self.hcdm.gamma() * diff
        dam_small = self.clf_2(damage_feat)
        output_loc = F.interpolate(loc_small, pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_dam = F.interpolate(dam_small, post_data.shape[-2:], mode="bilinear", align_corners=False)
        r3, s3 = self.ia_stage3(rgb[2], sar[2])
        r4, s4 = self.ia_stage4(rgb[3], sar[3])
        out: Dict[str, torch.Tensor] = {
            "output_loc": output_loc, "output_dam": output_dam,
            "final_feat_damage": damage_feat, "final_feat_damage_base": base,
            "hcdm_feat": diff, "building_prob": building_prob,
            "ia_rgb_s3": r3, "ia_sar_s3": s3, "ia_rgb_s4": r4, "ia_sar_s4": s4,
            **diag,
        }
        if self.instance_enabled and instance_map is not None:
            out.update(self._instance_forward(damage_feat, output_dam, instance_map, damage_gt))
        else:
            out.update({"instance_features": damage_feat.new_zeros((0, 256)), "instance_labels": damage_feat.new_empty((0,), dtype=torch.long), "building_logits3": damage_feat.new_zeros((0, 3)), "instance_records": []})
        return out


DamageFormer = DamageFormerPriorityModular
