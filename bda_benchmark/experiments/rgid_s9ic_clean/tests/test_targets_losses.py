from __future__ import annotations

import numpy as np
import torch

from rgid.cache_dataset import RGIDCacheDataset, collate_cached_samples
from rgid.config import RGIDConfig
from rgid.targets_losses import (
    BoundaryBCEDiceLoss,
    CenterNetFocalLoss,
    build_boundary_target,
    build_center_target,
    build_rgid_targets,
)


def test_rectangle_has_one_gaussian_maximum():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 10:22] = 1
    target, stats = build_center_target(labels, (8, 8), RGIDConfig())
    assert int((target == 1.0).sum()) == 1
    assert stats.valid_instances == 1
    assert stats.center_collisions == 0


def test_border_instance_has_no_center():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[0:10, 5:15] = 1
    target, stats = build_center_target(labels, (8, 8), RGIDConfig())
    assert float(target.max()) == 0.0
    assert stats.valid_instances == 0
    assert stats.ignored_border_instances == 1


def test_perfect_center_prediction_has_near_zero_loss():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 4] = 1.0
    logits = torch.full_like(target, -20.0)
    logits[0, 0, 3, 4] = 20.0
    loss = CenterNetFocalLoss()(logits, target)
    assert float(loss) < 1e-6


def test_center_focal_backward_is_finite():
    torch.manual_seed(4)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 3, 3] = 1.0
    loss = CenterNetFocalLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bf16_extreme_center_boundary_losses_and_gradients_are_finite():
    values = torch.tensor([-20.0, 20.0, 8.0, -8.0], dtype=torch.bfloat16)
    center_logits = values.reshape(1, 1, 2, 2).clone().requires_grad_(True)
    boundary_logits = values.flip(0).reshape(1, 1, 2, 2).clone().requires_grad_(True)
    center_target = torch.tensor(
        [[[[0.0, 1.0], [0.0, 0.5]]]], dtype=torch.bfloat16
    )
    boundary_target = torch.tensor(
        [[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.bfloat16
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        center_loss = CenterNetFocalLoss()(center_logits, center_target)
        boundary_loss = BoundaryBCEDiceLoss()(boundary_logits, boundary_target)
        total = center_loss + boundary_loss
    total.backward()
    assert center_loss.dtype == torch.float32
    assert boundary_loss.dtype == torch.float32
    assert torch.isfinite(total)
    assert torch.isfinite(center_logits.grad).all()
    assert torch.isfinite(boundary_logits.grad).all()


def test_boundary_and_batch_target_shapes():
    labels = np.zeros((2, 32, 32), dtype=np.int64)
    labels[:, 8:16, 8:16] = 1
    labels[:, 16:24, 16:24] = 2
    boundary = build_boundary_target(labels[0], (8, 8), boundary_width=1)
    targets, stats = build_rgid_targets(
        torch.from_numpy(labels), (8, 8), RGIDConfig()
    )
    assert boundary.shape == (8, 8)
    assert targets["center_target"].shape == (2, 1, 8, 8)
    assert targets["boundary_target"].shape == (2, 1, 8, 8)
    assert len(stats) == 2


def _cached_sample(name="sample", feature_size=(8, 8), label_size=(32, 32)):
    return {
        "fused_roi": torch.randn(256, *feature_size),
        "instance_ids": torch.zeros(label_size, dtype=torch.long),
        "parent_mask": torch.zeros(label_size, dtype=torch.bool),
        "name": name,
    }


def test_cache_dataset_reads_sorted_independent_samples(tmp_path):
    torch.save(_cached_sample("second"), tmp_path / "000001.pt")
    torch.save(_cached_sample("first"), tmp_path / "000000.pt")
    dataset = RGIDCacheDataset(tmp_path)
    assert len(dataset) == 2
    assert dataset[0]["name"] == "first"
    assert dataset[0]["fused_roi"].shape == (256, 8, 8)
    batch = collate_cached_samples([dataset[0], dataset[1]])
    assert batch["fused_roi"].shape == (2, 256, 8, 8)
    assert batch["instance_ids"].shape == (2, 32, 32)
    assert batch["parent_mask"].dtype == torch.bool
    assert batch["name"] == ["first", "second"]


def test_cache_collate_rejects_any_inconsistent_shape():
    first = _cached_sample("first")
    second = _cached_sample("second", feature_size=(7, 8))
    try:
        collate_cached_samples([first, second])
    except ValueError as error:
        assert "fused_roi shape" in str(error)
    else:
        raise AssertionError("inconsistent cache shapes must be rejected")
