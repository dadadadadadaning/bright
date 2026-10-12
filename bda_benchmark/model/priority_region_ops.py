from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class InstanceRecord:
    batch_index: int
    instance_id: int
    label: int
    area: int


def _connected_components_one(mask: torch.Tensor, min_area: int = 8) -> torch.Tensor:
    """Dependency-free 4-neighbour connected components for one 2-D bool mask.

    This is intended as a runnable baseline. Polygon/instance-ID annotations are
    preferable because touching buildings can be merged by connected components.
    """
    src = mask.detach().to("cpu", torch.bool)
    h, w = src.shape
    out = torch.zeros((h, w), dtype=torch.long)
    visited = torch.zeros((h, w), dtype=torch.bool)
    next_id = 1
    for y0 in range(h):
        for x0 in range(w):
            if not bool(src[y0, x0]) or bool(visited[y0, x0]):
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and bool(src[yy, xx]) and not bool(visited[yy, xx]):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            if len(pixels) >= min_area:
                for y, x in pixels:
                    out[y, x] = next_id
                next_id += 1
    return out.to(mask.device)


def build_instance_map_from_binary(binary: torch.Tensor, min_area: int = 8) -> torch.Tensor:
    if binary.ndim == 4 and binary.shape[1] == 1:
        binary = binary[:, 0]
    if binary.ndim != 3:
        raise ValueError(f"binary must be [B,H,W], got {tuple(binary.shape)}")
    return torch.stack([_connected_components_one(x, min_area) for x in binary], dim=0)


def resize_instance_map(instance_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(instance_map[:, None].float(), size=size, mode="nearest")[:, 0].long()


def binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    k = 2 * radius + 1
    return F.max_pool2d(mask[None, None].float(), kernel_size=k, stride=1, padding=radius)[0, 0] > 0


def binary_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    return ~binary_dilate(~mask.bool(), radius)


def majority_instance_label(mask: torch.Tensor, damage_gt: torch.Tensor) -> int:
    values = damage_gt[mask]
    values = values[(values >= 1) & (values <= 3)]
    if values.numel() == 0:
        return 0
    return int(torch.bincount(values, minlength=4)[1:4].argmax().item() + 1)


def collect_instance_records(
    instance_map: torch.Tensor,
    damage_gt: torch.Tensor | None,
) -> List[InstanceRecord]:
    records: List[InstanceRecord] = []
    for b in range(instance_map.shape[0]):
        max_id = int(instance_map[b].max().item())
        for iid in range(1, max_id + 1):
            m = instance_map[b] == iid
            area = int(m.sum().item())
            if area == 0:
                continue
            label = majority_instance_label(m, damage_gt[b]) if damage_gt is not None else 0
            records.append(InstanceRecord(b, iid, label, area))
    return records
