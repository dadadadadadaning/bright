from __future__ import annotations

from pathlib import Path

import torch


def save_rgid_real_batch(
    fused_roi: torch.Tensor,
    instance_ids: torch.Tensor,
    parent_mask: torch.Tensor,
    save_path,
) -> None:
    if fused_roi.ndim != 4 or fused_roi.shape[1] != 256:
        raise ValueError("fused_roi must be [B,256,Hh,Wh]")
    if instance_ids.ndim != 3 or instance_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("instance_ids must be a 3D integer tensor")
    if parent_mask.ndim != 3:
        raise ValueError("parent_mask must be [B,H,W]")
    if not (
        fused_roi.shape[0] == instance_ids.shape[0] == parent_mask.shape[0]
    ):
        raise ValueError("batch sizes do not match")
    if instance_ids.shape[-2:] != parent_mask.shape[-2:]:
        raise ValueError("instance_ids and parent_mask spatial shapes differ")
    if not torch.isfinite(fused_roi).all():
        raise ValueError("fused_roi contains NaN or Inf")
    if (instance_ids < 0).any():
        raise ValueError("instance_ids contains negative values")
    if not parent_mask.bool().any():
        raise ValueError("parent_mask has no foreground")
    payload = {
        "fused_roi": fused_roi.detach().cpu().float(),
        "instance_ids": instance_ids.detach().cpu().long(),
        "parent_mask": parent_mask.detach().cpu().bool(),
        "metadata": {
            "fused_roi_shape": list(fused_roi.shape),
            "instance_ids_shape": list(instance_ids.shape),
            "parent_mask_shape": list(parent_mask.shape),
            "fused_roi_source": "frozen_S8_fused_roi",
            "instance_protocol": "GT-derived pseudo-instance",
        },
    }
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
