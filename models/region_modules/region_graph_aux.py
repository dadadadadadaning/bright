"""Region graph auxiliary classification loss for BRIGHT.

This second-stage module avoids PyTorch Geometric.  It builds a tiny dense KNN
graph between connected building regions and applies two simple GCN layers.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .region_pooling_aux import _connected_components, _empty_info


class SimpleGCNLayer(nn.Module):
    """Dense adjacency GCN layer: neighbor average -> Linear -> ReLU."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        x_nei = adj @ x / degree
        return F.relu(self.linear(x_nei), inplace=True)


class RegionGraphAuxLoss(nn.Module):
    """Connected-region KNN graph auxiliary classifier."""

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 128,
        lambda_region: float = 0.05,
        min_region_pixels: int = 8,
        purity_thr: float = 0.60,
        knn_k: int = 4,
        class_weights: Iterable[float] = (1.0, 2.0, 2.0),
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.lambda_region = float(lambda_region)
        self.min_region_pixels = int(min_region_pixels)
        self.purity_thr = float(purity_thr)
        self.knn_k = int(knn_k)
        self.ignore_index = int(ignore_index)
        self.gcn1 = SimpleGCNLayer(feat_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        weights = torch.tensor(tuple(class_weights), dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    @staticmethod
    def _resize_label(label: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(label.float().unsqueeze(1), size=size, mode="nearest").squeeze(1).long()

    def _collect_regions_one(self, feat_b: torch.Tensor, label_b: torch.Tensor):
        device = feat_b.device
        label_np = label_b.detach().cpu().numpy()
        building_mask = (label_np > 0) & (label_np != self.ignore_index)
        num_cc, cc = _connected_components(building_mask)
        feats: List[torch.Tensor] = []
        targets: List[int] = []
        centers: List[Tuple[float, float]] = []

        for region_id in range(1, num_cc + 1):
            region_np = cc == region_id
            area = int(region_np.sum())
            if area < self.min_region_pixels:
                continue
            region_labels = label_np[region_np]
            valid = region_labels[(region_labels >= 1) & (region_labels <= 3)]
            if valid.size == 0:
                continue
            counts = np.bincount(valid.astype(np.int64), minlength=4)
            majority_label = int(np.argmax(counts[1:4]) + 1)
            purity = float(counts[majority_label]) / float(max(1, valid.size))
            if purity < self.purity_thr:
                continue

            ys, xs = np.where(region_np)
            centers.append((float(ys.mean()), float(xs.mean())))
            mask = torch.from_numpy(region_np).to(device=device, dtype=torch.bool)
            feats.append(feat_b[:, mask].mean(dim=1))
            targets.append(majority_label - 1)
        return feats, targets, centers

    def _knn_adj(self, centers: List[Tuple[float, float]], device: torch.device) -> torch.Tensor:
        n = len(centers)
        adj = torch.eye(n, device=device)
        if n <= 1:
            return adj
        pts = torch.tensor(centers, device=device, dtype=torch.float32)
        dist = torch.cdist(pts, pts)
        k = min(self.knn_k + 1, n)
        nn_idx = dist.topk(k=k, largest=False).indices
        for i in range(n):
            adj[i, nn_idx[i]] = 1.0
        adj = torch.maximum(adj, adj.t())
        return adj

    def forward(self, damage_feat: torch.Tensor, damage_label: torch.Tensor):
        if damage_feat.dim() != 4:
            raise ValueError(f"damage_feat must be [B,C,h,w], got {tuple(damage_feat.shape)}")
        if damage_label.dim() == 4 and damage_label.size(1) == 1:
            damage_label = damage_label.squeeze(1)
        if damage_label.dim() != 3:
            raise ValueError(f"damage_label must be [B,H,W], got {tuple(damage_label.shape)}")

        b, _, h, w = damage_feat.shape
        label_small = self._resize_label(damage_label, (h, w))
        losses: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for idx in range(b):
            feats, targets, centers = self._collect_regions_one(damage_feat[idx], label_small[idx])
            if len(feats) == 0:
                continue
            x = torch.stack(feats, dim=0)
            target = torch.tensor(targets, device=damage_feat.device, dtype=torch.long)
            adj = self._knn_adj(centers, device=damage_feat.device).to(dtype=x.dtype)
            x = self.gcn1(x, adj)
            x = self.gcn2(x, adj)
            logits = self.classifier(x)
            weights = self.class_weights.to(device=damage_feat.device, dtype=logits.dtype)
            losses.append(F.cross_entropy(logits, target, weight=weights))
            all_targets.append(target)

        info = _empty_info(damage_feat.device)
        if not losses:
            return damage_feat.sum() * 0.0, info

        loss = torch.stack(losses).mean()
        target_all = torch.cat(all_targets, dim=0)
        with torch.no_grad():
            info["num_regions"] = torch.tensor(float(target_all.numel()), device=damage_feat.device)
            info["num_intact_regions"] = (target_all == 0).float().sum()
            info["num_damaged_regions"] = (target_all == 1).float().sum()
            info["num_destroyed_regions"] = (target_all == 2).float().sum()
        return self.lambda_region * loss, info

