from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import find_boundaries

from .config import RGIDConfig


@dataclass
class TargetBuildStats:
    valid_instances: int = 0
    ignored_border_instances: int = 0
    center_collisions: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def binary_mask_to_instances(
    mask: np.ndarray,
    connectivity: int = 1,
) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, _ = ndimage.label(array, structure=structure)
    return labels.astype(np.int32, copy=False)


def _deterministic_unique_center(mask: np.ndarray) -> Tuple[int, int]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("instance mask must be a non-empty 2D array")
    distance = ndimage.distance_transform_edt(binary)
    candidates = np.argwhere(distance == distance.max())
    centroid = np.argwhere(binary).mean(axis=0)
    squared = np.square(candidates - centroid[None, :]).sum(axis=1)
    order = np.lexsort((candidates[:, 1], candidates[:, 0], squared))
    y, x = candidates[int(order[0])]
    return int(y), int(x)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def _map_coordinate(value: int, src_size: int, dst_size: int) -> int:
    mapped = round(((float(value) + 0.5) * dst_size / src_size) - 0.5)
    return int(np.clip(mapped, 0, dst_size - 1))


def build_center_target(
    instance_ids: np.ndarray,
    output_size: Tuple[int, int],
    config: RGIDConfig,
) -> Tuple[np.ndarray, TargetBuildStats]:
    labels = np.asarray(instance_ids)
    if labels.ndim != 2:
        raise ValueError("instance_ids must be two-dimensional")
    out_h, out_w = map(int, output_size)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_size must be positive")
    src_h, src_w = labels.shape
    target = np.zeros((out_h, out_w), dtype=np.float32)
    stats = TargetBuildStats()
    yy, xx = np.mgrid[0:out_h, 0:out_w]

    for instance_id in np.unique(labels):
        if int(instance_id) <= 0:
            continue
        mask = labels == instance_id
        if _touches_border(mask):
            stats.ignored_border_instances += 1
            continue
        center_y, center_x = _deterministic_unique_center(mask)
        mapped_y = _map_coordinate(center_y, src_h, out_h)
        mapped_x = _map_coordinate(center_x, src_w, out_w)
        if target[mapped_y, mapped_x] >= 1.0:
            stats.center_collisions += 1
        scaled_area = float(mask.sum()) * out_h / src_h * out_w / src_w
        radius = math.sqrt(max(scaled_area, 0.0) / math.pi)
        sigma = float(
            np.clip(
                radius / config.gaussian_radius_divisor,
                config.gaussian_sigma_min,
                config.gaussian_sigma_max,
            )
        )
        gaussian = np.exp(
            -(
                np.square(yy - mapped_y) + np.square(xx - mapped_x)
            )
            / (2.0 * sigma * sigma)
        ).astype(np.float32)
        target = np.maximum(target, gaussian)
        target[mapped_y, mapped_x] = 1.0
        stats.valid_instances += 1
    return target, stats


def build_boundary_target(
    instance_ids: np.ndarray,
    output_size: Tuple[int, int],
    boundary_width: int = 1,
) -> np.ndarray:
    labels = np.asarray(instance_ids)
    if labels.ndim != 2:
        raise ValueError("instance_ids must be two-dimensional")
    if boundary_width < 1:
        raise ValueError("boundary_width must be positive")
    boundary = find_boundaries(
        labels,
        mode="inner",
        connectivity=1,
    )
    if boundary_width > 1:
        boundary = ndimage.binary_dilation(
            boundary,
            iterations=boundary_width - 1,
        ) & (labels > 0)
    tensor = torch.from_numpy(boundary.astype(np.float32))[None, None]
    pooled = F.adaptive_max_pool2d(tensor, tuple(map(int, output_size)))
    return pooled[0, 0].numpy().astype(np.float32, copy=False)


def _as_instance_sequence(instance_ids_batch) -> List[np.ndarray]:
    if isinstance(instance_ids_batch, torch.Tensor):
        array = instance_ids_batch.detach().cpu().numpy()
        if array.ndim == 2:
            array = array[None]
        return [array[index] for index in range(array.shape[0])]
    array = np.asarray(instance_ids_batch)
    if array.dtype != object and array.ndim in (2, 3):
        if array.ndim == 2:
            array = array[None]
        return [array[index] for index in range(array.shape[0])]
    return [np.asarray(item) for item in instance_ids_batch]


def build_rgid_targets(
    instance_ids_batch,
    output_size,
    config: RGIDConfig,
    device=None,
):
    center_targets: List[np.ndarray] = []
    boundary_targets: List[np.ndarray] = []
    stats: List[TargetBuildStats] = []
    for instance_ids in _as_instance_sequence(instance_ids_batch):
        center, row_stats = build_center_target(instance_ids, output_size, config)
        boundary = build_boundary_target(
            instance_ids,
            output_size,
            boundary_width=config.boundary_width,
        )
        center_targets.append(center)
        boundary_targets.append(boundary)
        stats.append(row_stats)
    center_tensor = torch.from_numpy(np.stack(center_targets))[:, None].float()
    boundary_tensor = torch.from_numpy(np.stack(boundary_targets))[:, None].float()
    if device is not None:
        center_tensor = center_tensor.to(device)
        boundary_tensor = boundary_tensor.to(device)
    return {
        "center_target": center_tensor,
        "boundary_target": boundary_tensor,
    }, stats


class CenterNetFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        target_f = target.float()
        pred = torch.sigmoid(logits_f).clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )
        positive = target_f == 1.0
        negative = target_f < 1.0
        negative_weight = torch.pow(1.0 - target_f, self.beta)
        positive_loss = (
            torch.log(pred)
            * torch.pow(1.0 - pred, self.alpha)
            * positive.float()
        )
        negative_loss = (
            torch.log1p(-pred)
            * torch.pow(pred, self.alpha)
            * negative_weight
            * negative.float()
        )
        positive_count = positive.float().sum()
        if bool((positive_count > 0).item()):
            return -(positive_loss.sum() + negative_loss.sum()) / positive_count
        return -negative_loss.sum()


class BoundaryBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 4.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        target_f = target.float()
        weight = torch.as_tensor(
            self.pos_weight,
            device=logits_f.device,
            dtype=torch.float32,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits_f,
            target_f,
            pos_weight=weight,
        )
        probability = torch.sigmoid(logits_f)
        intersection = (probability * target_f).sum(dtype=torch.float32)
        denominator = probability.sum(dtype=torch.float32) + target_f.sum(
            dtype=torch.float32
        )
        dice = 1.0 - (2.0 * intersection + self.eps) / (
            denominator + self.eps
        )
        return bce + dice


class RGIDLoss(nn.Module):
    def __init__(self, config: RGIDConfig) -> None:
        super().__init__()
        self.config = config
        self.center_loss = CenterNetFocalLoss(
            alpha=config.center_focal_alpha,
            beta=config.center_focal_beta,
        )
        self.boundary_loss = BoundaryBCEDiceLoss(
            pos_weight=config.boundary_pos_weight
        )

    def forward(self, outputs, targets):
        center_loss = self.center_loss(
            outputs["center_logits"], targets["center_target"]
        )
        boundary_loss = self.boundary_loss(
            outputs["boundary_logits"], targets["boundary_target"]
        )
        total = (
            self.config.center_loss_weight * center_loss
            + self.config.boundary_loss_weight * boundary_loss
        )
        return {
            "loss": total,
            "center_loss": center_loss,
            "boundary_loss": boundary_loss,
        }
