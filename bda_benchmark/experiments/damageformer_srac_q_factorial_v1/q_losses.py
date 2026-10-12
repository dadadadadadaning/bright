"""Train-only semantic reconstruction targets for controlled Q ablations.

The four-class GT is used only here, after Q was computed from model inputs.
It never determines a candidate seed, a cue, or any Val/Test query.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool


def _area_targets(gt: torch.Tensor, shape: tuple[int, int], classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if gt.ndim != 3 or classes not in (2, 4):
        raise ValueError("Expected BxHxW labels and 2 or 4 classes")
    valid = (gt >= 0) & (gt < 4)
    class_gt = gt.clamp(0, 3) if classes == 4 else (gt > 0).long()
    one_hot = F.one_hot(class_gt, classes).permute(0, 3, 1, 2).float() * valid[:, None]
    area = F.interpolate(one_hot, size=shape, mode="area")
    valid_area = F.interpolate(valid[:, None].float(), size=shape, mode="area")
    return area, valid_area


def soft_label_reconstruction(
    q: torch.Tensor, ids: torch.Tensor, shape: tuple[int, int],
    count: int, gt: torch.Tensor, classes: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GT -> Q weighted regional label -> Q pixel reproject -> masked CE.

    Ignored pixels are excluded from each region's label denominator; they do
    not silently become background.  The loss also excludes all-ignore pixels.
    This is a training regularizer, not a measured upper bound on model IoU.
    """
    if q.ndim != 3 or q.shape[1] != 9 or q.shape[-1] != shape[0] * shape[1]:
        raise ValueError("Q and grid shape disagree")
    if ids.shape != (9, q.shape[-1]) or count < 1:
        raise ValueError("Candidate IDs or region count disagree")
    with torch.autocast(device_type=q.device.type, enabled=False):
        area, valid_area = _area_targets(gt.long(), shape, classes)
        area, valid_area = area.to(q.device), valid_area.to(q.device)
        q_float = q.float()
        pooled = soft_pool_unpool(torch.cat((area, valid_area), dim=1), q_float, ids, count)
        region = pooled["tokens"][..., :classes]
        region_valid = pooled["tokens"][..., classes:]
        region = region / region_valid.clamp_min(1e-8)
        region = region / region.sum(-1, keepdim=True).clamp_min(1e-8)
        batch, nine, pixels = q_float.shape
        reconstructed = q_float.new_zeros((batch, pixels, classes))
        for candidate in range(nine):
            labels = region.gather(1, ids[candidate][None, :, None].expand(batch, -1, classes))
            reconstructed += q_float[:, candidate, :, None] * labels
        reconstructed = reconstructed / reconstructed.sum(-1, keepdim=True).clamp_min(1e-8)
        target = area.flatten(2).transpose(1, 2)
        valid_flat = valid_area.flatten(2).transpose(1, 2)
        if bool((valid_flat > 0).any()):
            loss = -(target * reconstructed.clamp_min(1e-8).log()).sum() / valid_flat.sum().clamp_min(1)
        else:
            loss = q_float.sum() * 0
    return loss, {
        "valid_grid_mass": valid_flat.detach().sum(),
        "nonempty_regions": (pooled["mass"] > 1e-6).sum().detach(),
        "labelled_regions": (region_valid[..., 0] > 1e-6).sum().detach(),
    }


def auxiliary_q_loss(arm: str, result: dict, gt: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if arm == "building_reconstruction":
        return soft_label_reconstruction(result["q"], result["ids"], result["shape"],
                                         result["count"], gt, classes=2)
    if arm == "semantic_reconstruction":
        return soft_label_reconstruction(result["q"], result["ids"], result["shape"],
                                         result["count"], gt, classes=4)
    return result["q"].sum() * 0, {}
