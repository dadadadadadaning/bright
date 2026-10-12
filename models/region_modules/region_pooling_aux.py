"""Region pooling auxiliary classification loss for BRIGHT.

The module is intentionally independent from DamageFormer.  It consumes a
feature map and the BRIGHT damage label, extracts connected building regions,
pools a feature per region, and applies a small MLP classifier for
intact/damaged/destroyed supervision.

Input:
    damage_feat:  [B, C, h, w]
    damage_label: [B, H, W], labels 0 background, 1 intact, 2 damaged,
                  3 destroyed, 255 ignore.

Output:
    loss_region: scalar tensor, already multiplied by lambda_region.
    info: dict with region counts for logging.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """Return connected components for a 2D boolean mask.

    Prefer cv2/scipy when available, but include a tiny BFS fallback so the
    module can still import and run in minimal environments.
    """

    mask = mask.astype(bool)
    if not mask.any():
        return 0, np.zeros(mask.shape, dtype=np.int32)

    try:
        import cv2  # type: ignore

        num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return int(num - 1), labels.astype(np.int32)
    except Exception:
        pass

    try:
        from scipy import ndimage  # type: ignore

        labels, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        return int(num), labels.astype(np.int32)
    except Exception:
        pass

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comp_id = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            comp_id += 1
            labels[y, x] = comp_id
            q: deque[Tuple[int, int]] = deque([(y, x)])
            while q:
                cy, cx = q.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = comp_id
                        q.append((ny, nx))
    return comp_id, labels


def _empty_info(device: torch.device) -> Dict[str, torch.Tensor]:
    z = torch.tensor(0.0, device=device)
    return {
        "num_regions": z,
        "num_intact_regions": z,
        "num_damaged_regions": z,
        "num_destroyed_regions": z,
    }


class RegionPoolingAuxLoss(nn.Module):
    """Connected-region average pooling auxiliary classifier."""

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 128,
        lambda_region: float = 0.05,
        min_region_pixels: int = 8,
        purity_thr: float = 0.60,
        class_weights: Iterable[float] = (1.0, 2.0, 2.0),
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.lambda_region = float(lambda_region)
        self.min_region_pixels = int(min_region_pixels)
        self.purity_thr = float(purity_thr)
        self.ignore_index = int(ignore_index)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        weights = torch.tensor(tuple(class_weights), dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    @staticmethod
    def _resize_label(label: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        label = label.float().unsqueeze(1)
        label_small = F.interpolate(label, size=size, mode="nearest").squeeze(1)
        return label_small.long()

    def _collect_regions(
        self,
        feat_b: torch.Tensor,
        label_b: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Collect differentiable pooled region features and 0/1/2 targets."""

        device = feat_b.device
        label_np = label_b.detach().cpu().numpy()
        building_mask = (label_np > 0) & (label_np != self.ignore_index)
        num_cc, cc = _connected_components(building_mask)
        region_feats: List[torch.Tensor] = []
        region_targets: List[int] = []

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

            mask = torch.from_numpy(region_np).to(device=device, dtype=torch.bool)
            pooled = feat_b[:, mask].mean(dim=1)
            region_feats.append(pooled)
            region_targets.append(majority_label - 1)

        return region_feats, region_targets

    def forward(self, damage_feat: torch.Tensor, damage_label: torch.Tensor):
        if damage_feat.dim() != 4:
            raise ValueError(f"damage_feat must be [B,C,h,w], got {tuple(damage_feat.shape)}")
        if damage_label.dim() == 4 and damage_label.size(1) == 1:
            damage_label = damage_label.squeeze(1)
        if damage_label.dim() != 3:
            raise ValueError(f"damage_label must be [B,H,W], got {tuple(damage_label.shape)}")

        b, _, h, w = damage_feat.shape
        label_small = self._resize_label(damage_label, (h, w))
        all_feats: List[torch.Tensor] = []
        all_targets: List[int] = []

        for idx in range(b):
            feats_i, targets_i = self._collect_regions(damage_feat[idx], label_small[idx])
            all_feats.extend(feats_i)
            all_targets.extend(targets_i)

        info = _empty_info(damage_feat.device)
        if len(all_feats) == 0:
            zero = damage_feat.sum() * 0.0
            return zero, info

        region_features = torch.stack(all_feats, dim=0)
        target = torch.tensor(all_targets, device=damage_feat.device, dtype=torch.long)
        logits = self.classifier(region_features)
        weights = self.class_weights.to(device=damage_feat.device, dtype=logits.dtype)
        loss = F.cross_entropy(logits, target, weight=weights)

        with torch.no_grad():
            info["num_regions"] = torch.tensor(float(target.numel()), device=damage_feat.device)
            info["num_intact_regions"] = (target == 0).float().sum()
            info["num_damaged_regions"] = (target == 1).float().sum()
            info["num_destroyed_regions"] = (target == 2).float().sum()

        return self.lambda_region * loss, info

