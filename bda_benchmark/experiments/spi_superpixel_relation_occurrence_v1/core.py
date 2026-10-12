"""Label-free structural-region operations used by the superpixel adapter."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def _grid_ids(height: int, width: int, n_segments: int, device: torch.device) -> torch.Tensor:
    side = max(2, int(math.sqrt(height * width / max(int(n_segments), 1))))
    cols = (width + side - 1) // side
    return (torch.arange(height, device=device)[:, None] // side * cols + torch.arange(width, device=device)[None] // side).long()


IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
REFERENCE_FEATURE_AREA = 160 * 160  # 640px training crop at P3 stride four.


def _slic_one(image: torch.Tensor, n_segments: int, compactness: float) -> np.ndarray:
    """Run SLIC on RGB in [0, 1]; a missing dependency is a hard error."""
    array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    try:
        from skimage.segmentation import slic
        return slic(array, n_segments=int(n_segments), compactness=float(compactness), start_label=0,
                    channel_axis=-1, convert2lab=True).astype(np.int64, copy=False)
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required for SLIC variants; refusing to replace "
            "SLIC silently with a regular grid"
        ) from error


@torch.no_grad()
def make_region_ids(pre_rgb: torch.Tensor, size: tuple[int, int], *, mode: str, n_segments: int, compactness: float) -> torch.Tensor:
    """Build IDs after the dataset's joint augmentation, so RGB/SAR/labels stay aligned.

    IDs are structure only: this function reads pre-disaster RGB and no GT/P3
    semantic label. ``shuffled`` uses a spatially mirrored RGB view without
    mapping its partition back, preserving SLIC statistics while breaking its
    correspondence with the current image.
    """
    batch, _, _, _ = pre_rgb.shape; height, width = size
    if mode == "none":
        return torch.zeros((batch, height, width), device=pre_rgb.device, dtype=torch.long)
    scaled_segments = max(2, int(round(float(n_segments) * height * width / REFERENCE_FEATURE_AREA)))
    if mode == "grid":
        return _grid_ids(height, width, scaled_segments, pre_rgb.device)[None].expand(batch, -1, -1).contiguous()
    source = F.interpolate(pre_rgb.float(), size=size, mode="bilinear", align_corners=False)
    mean = source.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = source.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    source = ((source * std + mean) / 255.0).clamp(0.0, 1.0)
    labels = []
    for row in source:
        if mode == "shuffled":
            # SLIC on a mirrored image has comparable region counts, sizes and
            # shapes, but its boundaries are deliberately wrong for this image.
            # Do not flip the returned ID map back.
            row = torch.flip(row, dims=(-1,))
        value = torch.from_numpy(_slic_one(row, scaled_segments, compactness)).to(pre_rgb.device)
        labels.append(value.long())
    return torch.stack(labels)


def split_regions_by_building(ids: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
    """Prevent a SLIC mean from mixing predicted-building and background pixels."""
    if building.dim() == 4:
        building = building[:, 0]
    return ids * 2 + building.to(torch.long)


def pooled_region_feature(feature: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Mean-pool each feature channel within an arbitrary hard region map."""
    batch, channels, height, width = feature.shape
    result = torch.empty_like(feature)
    for row in range(batch):
        group = ids[row].reshape(-1); values = feature[row].reshape(channels, -1)
        count = int(group.max()) + 1
        sums = torch.zeros((channels, count), dtype=feature.dtype, device=feature.device)
        sums.scatter_add_(1, group[None].expand(channels, -1), values)
        sizes = torch.bincount(group, minlength=count).to(feature.dtype).clamp_min(1)
        result[row] = (sums / sizes[None]).index_select(1, group).reshape(channels, height, width)
    return result


def region_boundary(ids: torch.Tensor) -> torch.Tensor:
    """One-pixel boundary of the ID map; values are 0/1, not semantic labels."""
    edge = torch.zeros_like(ids, dtype=torch.bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]; edge[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :, 1:] |= ids[:, :, 1:] != ids[:, :, :-1]; edge[:, :, :-1] |= ids[:, :, 1:] != ids[:, :, :-1]
    return edge[:, None]


def reliable_region_consistency(logits: torch.Tensor, ids: torch.Tensor, safe_gate: torch.Tensor) -> torch.Tensor:
    """Low-weight, label-free Bernoulli KL inside reliable structural regions."""
    probability = logits.float().sigmoid()
    mean = pooled_region_feature(probability, ids).detach().clamp(1e-4, 1. - 1e-4)
    value = probability.clamp(1e-4, 1. - 1e-4)
    kl = mean * (mean.log() - value.log()) + (1. - mean) * ((1. - mean).log() - (1. - value).log())
    return (kl * safe_gate.detach()).sum() / safe_gate.detach().sum().clamp_min(1.)


def candidate_token_targets(
    parent_logits: torch.Tensor,
    p_building: torch.Tensor,
    damage: torch.Tensor,
    location: torch.Tensor,
    min_building_fraction: float = 0.25,
):
    """One candidate target per output token; never upsample a token for loss.

    Positive: P3 says Background/Intact in a predicted building token but at
    least one valid GT pixel is Affected.  Negative: every valid pixel is
    Background/Intact.  Mixed/ignored tokens are excluded from training.
    """
    size = parent_logits.shape[-2:]
    valid = ((damage != 255) & (location > 0)).float()[:, None]
    affected = ((damage >= 2) & (damage != 255) & (location > 0)).float()[:, None]
    nonaffected = ((damage <= 1) & (damage != 255) & (location > 0)).float()[:, None]
    valid_area = F.adaptive_avg_pool2d(valid, size)
    positive = F.adaptive_max_pool2d(affected, size) > 0
    all_nonaffected = F.adaptive_avg_pool2d(nonaffected, size) >= (valid_area - 1e-6)
    parent_candidate = parent_logits.detach().argmax(1, keepdim=True) <= 1
    building_candidate = p_building.detach() >= .5
    if not 0.0 < float(min_building_fraction) <= 1.0:
        raise ValueError("min_building_fraction must be in (0, 1]")
    candidate = parent_candidate & building_candidate & (valid_area >= float(min_building_fraction))
    pos = candidate & positive
    neg = candidate & all_nonaffected & ~positive
    return pos[:, 0], neg[:, 0]


def candidate_losses(occurrence_logits: torch.Tensor, delta: torch.Tensor, parent_logits: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, shift_cap: float):
    logits, delta = occurrence_logits[:, 0].float(), delta[:, 0].float()
    bce = F.binary_cross_entropy_with_logits(logits, positive.float(), reduction="none")
    pos_loss = bce[positive].mean() if bool(positive.any()) else logits.sum() * 0.
    neg_loss = bce[negative].mean() if bool(negative.any()) else logits.sum() * 0.
    required = (parent_logits[:, :2].detach().amax(1) - parent_logits[:, 2:4].detach().amax(1)).clamp_min(0.)
    eligible = positive & (required <= float(shift_cap))
    margin = F.softplus(required[eligible] - delta[eligible]).mean() if bool(eligible.any()) else logits.sum() * 0.
    return .5 * (pos_loss + neg_loss), margin, {"candidate_tokens": int((positive | negative).sum()), "positive_tokens": int(positive.sum()), "negative_tokens": int(negative.sum()), "margin_positive_tokens": int(eligible.sum())}
