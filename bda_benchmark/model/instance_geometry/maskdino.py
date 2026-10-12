"""BRIGHT-to-MaskDINO targets, query budget, and S8 feature pyramid adapter."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..DamageFormer_S9Ic_ReliableGeometry import select_geometry_feature


def _majority_damage_label(
    labels: Tensor,
    mask: Tensor,
    ignore_index: int,
) -> Optional[int]:
    values = labels[mask]
    values = values[(values >= 1) & (values <= 3) & (values != ignore_index)]
    if values.numel() == 0:
        return None
    counts = torch.bincount(values.to(torch.long), minlength=4)
    return int(torch.argmax(counts[1:4]).item()) + 1


def _mask_box_cxcywh(mask: Tensor) -> Tensor:
    points = torch.nonzero(mask, as_tuple=False)
    y0 = points[:, 0].min().to(torch.float32)
    x0 = points[:, 1].min().to(torch.float32)
    y1 = points[:, 0].max().to(torch.float32) + 1.0
    x1 = points[:, 1].max().to(torch.float32) + 1.0
    height, width = mask.shape
    return torch.stack(
        (
            ((x0 + x1) * 0.5) / float(width),
            ((y0 + y1) * 0.5) / float(height),
            (x1 - x0) / float(width),
            (y1 - y0) / float(height),
        )
    )


def instances_to_maskdino_target(
    instance_map: Tensor | np.ndarray,
    damage_label: Tensor | np.ndarray,
    *,
    ignore_index: int = 255,
    min_area: int = 1,
    image_id: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Convert official instance ids to one-query-per-building MaskDINO targets.

    BRIGHT damage labels 1/2/3 become contiguous query labels 0/1/2.
    Instance id 0 is background.
    """

    instance_map = torch.as_tensor(instance_map, dtype=torch.long)
    damage_label = torch.as_tensor(damage_label, dtype=torch.long)
    if instance_map.ndim != 2 or damage_label.ndim != 2:
        raise ValueError("instance_map and damage_label must be [H,W]")
    if instance_map.shape != damage_label.shape:
        raise ValueError("instance_map and damage_label must share a shape")
    if min_area < 1:
        raise ValueError("min_area must be positive")

    masks = []
    labels = []
    boxes = []
    instance_ids = []
    areas = []
    for instance_id in torch.unique(instance_map).tolist():
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        mask = instance_map == instance_id
        area = int(mask.sum().item())
        if area < min_area:
            continue
        damage = _majority_damage_label(
            damage_label,
            mask,
            ignore_index,
        )
        if damage is None:
            continue
        masks.append(mask)
        labels.append(damage - 1)
        boxes.append(_mask_box_cxcywh(mask))
        instance_ids.append(instance_id)
        areas.append(area)

    height, width = instance_map.shape
    device = instance_map.device
    target: Dict[str, Tensor] = {
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "masks": (
            torch.stack(masks, dim=0)
            if masks
            else torch.zeros((0, height, width), dtype=torch.bool, device=device)
        ),
        "boxes": (
            torch.stack(boxes, dim=0).to(device=device)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32, device=device)
        ),
        "instance_ids": torch.tensor(
            instance_ids,
            dtype=torch.long,
            device=device,
        ),
        "areas": torch.tensor(areas, dtype=torch.long, device=device),
        "orig_size": torch.tensor(
            [height, width],
            dtype=torch.long,
            device=device,
        ),
        "size": torch.tensor(
            [height, width],
            dtype=torch.long,
            device=device,
        ),
    }
    if image_id is not None:
        target["image_id"] = torch.tensor(
            [int(image_id)],
            dtype=torch.long,
            device=device,
        )
    return target


def compute_query_budget(
    instance_counts: Sequence[int],
    *,
    percentile: float = 99.0,
    headroom: float = 1.25,
    minimum: int = 100,
    maximum: int = 500,
    round_to: int = 10,
) -> Dict[str, Any]:
    """Compute a reproducible query count from training-set instance counts."""

    if not instance_counts:
        raise ValueError("instance_counts must not be empty")
    values = np.asarray(instance_counts, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("instance counts must be non-negative")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0,100]")
    if headroom < 1.0 or minimum < 1 or maximum < minimum or round_to < 1:
        raise ValueError("invalid query-budget bounds")

    percentile_value = float(np.percentile(values, percentile))
    raw = int(math.ceil(percentile_value * headroom))
    rounded = int(math.ceil(raw / round_to) * round_to)
    recommended = min(maximum, max(minimum, rounded))
    overflow = int((values > recommended).sum())
    return {
        "recommended_queries": recommended,
        "sample_count": int(values.size),
        "maximum_instances": int(values.max()),
        "mean_instances": float(values.mean()),
        "percentile": float(percentile),
        "percentile_instances": percentile_value,
        "headroom": float(headroom),
        "round_to": int(round_to),
        "overflow_images": overflow,
        "overflow_rate": overflow / float(values.size),
    }


class _Projection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3 if stride > 1 else 1,
                stride=stride,
                padding=1 if stride > 1 else 0,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class S8MultiScaleFeatureAdapter(nn.Module):
    """Turn frozen S8 ``fused_roi`` into MaskDINO-style res2..res5 features."""

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        *,
        feature_key: str = "fused_roi",
        base_stride: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_key = feature_key
        self.base_stride = int(base_stride)
        self.res2 = _Projection(in_channels, out_channels, stride=1)
        self.res3 = _Projection(out_channels, out_channels, stride=2)
        self.res4 = _Projection(out_channels, out_channels, stride=2)
        self.res5 = _Projection(out_channels, out_channels, stride=2)

    def forward(self, s8_auxiliary: Mapping[str, Any]) -> "OrderedDict[str, Tensor]":
        key, feature = select_geometry_feature(
            s8_auxiliary,
            requested_key=self.feature_key,
        )
        if feature.shape[1] != self.in_channels:
            raise RuntimeError(
                f"S8 feature {key!r} has {feature.shape[1]} channels; "
                f"adapter expects {self.in_channels}"
            )
        res2 = self.res2(feature)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        return OrderedDict(
            (("res2", res2), ("res3", res3), ("res4", res4), ("res5", res5))
        )

    def output_shape(self) -> Dict[str, Dict[str, int]]:
        return {
            f"res{level}": {
                "channels": self.out_channels,
                "stride": self.base_stride * (2 ** (level - 2)),
            }
            for level in range(2, 6)
        }

