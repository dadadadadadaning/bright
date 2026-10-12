from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InstanceRecord:
    batch_index: int
    instance_id: int
    label: int
    area: int


def _connected_components_4(binary: np.ndarray) -> np.ndarray:
    """Dependency-free 4-connected components for a single HxW binary mask."""
    h, w = binary.shape
    out = np.zeros((h, w), dtype=np.int32)
    current = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or out[y, x] != 0:
                continue
            current += 1
            q = deque([(y, x)])
            out[y, x] = current
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and out[ny, nx] == 0:
                        out[ny, nx] = current
                        q.append((ny, nx))
    return out


def build_instance_map_from_binary(
    building_mask: torch.Tensor,
    min_area: int = 8,
) -> torch.Tensor:
    """Create instance IDs from a binary building mask.

    This is a runnable baseline only. Adjacent touching buildings may be merged.
    Input: [B,H,W] or [B,1,H,W]. Output: [B,H,W], IDs start at 1 per image.
    """
    if building_mask.ndim == 4:
        building_mask = building_mask[:, 0]
    arr = (building_mask.detach().cpu().numpy() > 0).astype(np.uint8)
    outputs = []
    for b in range(arr.shape[0]):
        comp = _connected_components_4(arr[b])
        if min_area > 1:
            ids, counts = np.unique(comp, return_counts=True)
            for idx, count in zip(ids, counts):
                if idx != 0 and count < min_area:
                    comp[comp == idx] = 0
            # compact IDs
            unique = [x for x in np.unique(comp) if x != 0]
            remap = np.zeros_like(comp)
            for new_id, old_id in enumerate(unique, start=1):
                remap[comp == old_id] = new_id
            comp = remap
        outputs.append(torch.from_numpy(comp))
    return torch.stack(outputs, dim=0).to(building_mask.device, dtype=torch.long)


def infer_instance_labels(
    instance_map: torch.Tensor,
    damage_gt: torch.Tensor,
    ignore_index: int = 255,
    min_valid_pixels: int = 4,
) -> Tuple[List[InstanceRecord], torch.Tensor]:
    """Assign one building label by majority vote over GT classes 1/2/3.

    Returns records and labels tensor aligned with records.
    """
    if instance_map.ndim == 4:
        instance_map = instance_map[:, 0]
    if damage_gt.ndim == 4:
        damage_gt = damage_gt[:, 0]
    records: List[InstanceRecord] = []
    labels: List[int] = []
    for b in range(instance_map.shape[0]):
        max_id = int(instance_map[b].max().item())
        for iid in range(1, max_id + 1):
            m = instance_map[b] == iid
            vals = damage_gt[b][m]
            vals = vals[(vals != ignore_index) & (vals >= 1) & (vals <= 3)]
            if vals.numel() < min_valid_pixels:
                continue
            hist = torch.bincount(vals.long(), minlength=4)
            label = int(hist[1:4].argmax().item() + 1)
            records.append(InstanceRecord(b, iid, label, int(m.sum().item())))
            labels.append(label)
    if labels:
        return records, torch.tensor(labels, device=damage_gt.device, dtype=torch.long)
    return records, torch.empty(0, device=damage_gt.device, dtype=torch.long)


def resize_instance_map(instance_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if instance_map.ndim == 3:
        x = instance_map[:, None].float()
    else:
        x = instance_map.float()
    return F.interpolate(x, size=size, mode="nearest")[:, 0].long()


def binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float()[None, None]
    k = 2 * radius + 1
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)[0, 0] > 0)


def binary_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    return ~binary_dilate(~mask.bool(), radius)
