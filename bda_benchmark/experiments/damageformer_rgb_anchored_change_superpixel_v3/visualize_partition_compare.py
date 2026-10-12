#!/usr/bin/env python3
"""Visual comparison of matched Grid, SLICO and learned building regions.

The forward partition never reads GT.  GT is used only after inference to
select a diagnostic crop and calculate region-quality statistics.  All three
partitions share the same predicted-building support and feature resolution,
which makes the visual comparison materially fairer than a GT-masked SLIC
illustration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import find_boundaries

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    load_a1_parent,
    load_v2_checkpoint,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import (
    read_list,
)

from .audit_partition import _grid_ids, _learned_ids, _sample_metrics, _slico_ids
from .model import DamageFormerRGBAnchoredChangeSuperpixelV3


IMAGENET_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
GT_COLORS = np.asarray(
    [[255, 255, 255], [70, 181, 121], [228, 189, 139], [182, 70, 69]],
    dtype=np.uint8,
)


def _choose_sample(
    dataset_path: Path,
    names: Iterable[str],
    sample_id: str,
    focus_class: int,
) -> str:
    names = list(names)
    if sample_id:
        if sample_id not in names:
            raise ValueError(f"sample_id {sample_id!r} is not present in the supplied validation list")
        return sample_id
    best = None
    for name in names:
        target_path = dataset_path / "target" / f"{name}_building_damage.tif"
        target = np.asarray(imageio.imread(target_path))
        if focus_class in (1, 2, 3):
            score = int((target == focus_class).sum())
        else:
            score = int(((target >= 1) & (target <= 3)).sum())
        candidate = (score, name)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("validation list is empty")
    return str(best[1])


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    value = value * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(value, 0, 255).astype(np.uint8)


def _sar_display(tensor: torch.Tensor) -> np.ndarray:
    raw = _denormalize(tensor)[..., 0].astype(np.float32)
    finite = raw[np.isfinite(raw)]
    lo, hi = np.percentile(finite, [1, 99])
    return np.clip((raw - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)


def _upsample_ids(value: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    result = F.interpolate(
        value[None, None].float(), size=size, mode="nearest"
    )[0, 0]
    return result.long().cpu().numpy()


def _upsample_scalar(value: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    result = F.interpolate(
        value[None, None].float(), size=size, mode="bilinear", align_corners=False
    )[0, 0]
    return result.cpu().numpy()


def _diagnostic_crop(target: np.ndarray, predicted: np.ndarray, margin: int) -> Tuple[slice, slice]:
    foreground = (target >= 1) & (target <= 3)
    labels, count = ndimage.label(foreground)
    candidates = []
    for idx in range(1, int(count) + 1):
        mask = labels == idx
        if not mask.any():
            continue
        affected = int(((target >= 2) & (target <= 3) & mask).sum())
        overlap = int((predicted & mask).sum())
        candidates.append(((affected > 0, affected, overlap, int(mask.sum())), mask))
    if candidates:
        mask = max(candidates, key=lambda row: row[0])[1]
    else:
        mask = predicted
    if not mask.any():
        return slice(0, target.shape[0]), slice(0, target.shape[1])
    yy, xx = np.where(mask)
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(target.shape[0], int(yy.max()) + margin + 1)
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(target.shape[1], int(xx.max()) + margin + 1)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(target.shape[0] - side, cy - side // 2))
    x0 = max(0, min(target.shape[1] - side, cx - side // 2))
    return slice(y0, y0 + side), slice(x0, x0 + side)


def _boundary_overlay(rgb: np.ndarray, labels: np.ndarray, building: np.ndarray) -> np.ndarray:
    view = rgb.astype(np.float32) / 255.0
    outside = ~building
    view[outside] *= 0.58
    edge = find_boundaries(labels, mode="inner") & building
    outer = find_boundaries(building.astype(np.uint8), mode="outer")
    view[edge] = np.asarray([1.0, 1.0, 1.0])
    view[outer] = np.asarray([0.98, 0.45, 0.53])
    return np.clip(view, 0.0, 1.0)


def _count_regions(ids: torch.Tensor, building: torch.Tensor) -> int:
    values = torch.unique(ids[building])
    return int(values.numel())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Visualize matched Grid/SLICO/learned partitions")
    parser.add_argument("--a1_ckpt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--val_list", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_id", default="")
    parser.add_argument("--focus_class", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid_cell", type=int, default=4)
    parser.add_argument("--segments", type=int, default=0)
    parser.add_argument("--crop_margin", type=int, default=64)
    args = parser.parse_args()

    if "test" in args.val_list.name.lower():
        raise RuntimeError("TEST split is forbidden for this diagnostic")
    names = read_list(str(args.val_list))
    sample_id = _choose_sample(args.dataset_path, names, args.sample_id, args.focus_class)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", "p5_anchored_change"))
    stage = str(payload.get("stage", "change"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerRGBAnchoredChangeSuperpixelV3(variant).to(device)
    parent_report = load_a1_parent(model, str(args.a1_ckpt))
    checkpoint_report = load_v2_checkpoint(model, str(args.checkpoint))
    model.configure_stage(stage)
    model.eval()

    dataset = MultimodalDamageAssessmentDatset(
        str(args.dataset_path), [sample_id], crop_size=1024, type="val"
    )
    pre_np, post_np, _, target_np, returned_id = dataset[0]
    if str(returned_id) != sample_id:
        raise RuntimeError(f"dataset identity mismatch: expected {sample_id}, got {returned_id}")
    pre = torch.from_numpy(np.asarray(pre_np)).unsqueeze(0).float().to(device)
    post = torch.from_numpy(np.asarray(post_np)).unsqueeze(0).float().to(device)
    target = torch.from_numpy(np.asarray(target_np)).long().to(device)

    _, _, aux = model(pre, post, return_aux=True)
    building = aux["building_ids"] > 0
    grid = _grid_ids(aux["building_ids"], args.grid_cell)
    slico = _slico_ids(pre, aux["building_ids"], args.segments)
    learned = _learned_ids(aux)
    h, w = aux["building_ids"].shape[-2:]
    target_small = F.interpolate(
        target[None, None].float(), size=(h, w), mode="nearest"
    )[0, 0].long()

    methods: Dict[str, torch.Tensor] = {
        "grid": grid[0],
        "slico": slico[0],
        "learned": learned[0],
    }
    metrics = {
        name: {
            **_sample_metrics(ids, target_small, building[0]),
            "region_count": _count_regions(ids, building[0]),
        }
        for name, ids in methods.items()
    }

    input_size = tuple(int(v) for v in target.shape)
    rgb = _denormalize(pre[0])
    sar = _sar_display(post[0])
    target_full = target.detach().cpu().numpy()
    building_full = _upsample_ids(building[0].long(), input_size).astype(bool)
    id_full = {name: _upsample_ids(ids, input_size) for name, ids in methods.items()}
    association_entropy = aux["association_entropy"][0].reshape(h, w)
    entropy_full = _upsample_scalar(association_entropy, input_size)
    crop = _diagnostic_crop(target_full, building_full, args.crop_margin)

    gt_view = GT_COLORS[np.clip(target_full, 0, 3)]
    gt_view[target_full == 255] = np.asarray([80, 80, 80], dtype=np.uint8)
    pred_view = rgb.astype(np.float32) / 255.0
    pred_edge = find_boundaries(building_full.astype(np.uint8), mode="outer")
    pred_view[pred_edge] = np.asarray([0.98, 0.45, 0.53])
    views = [
        ("Pre-event RGB", rgb[crop]),
        ("Post-event SAR", sar[crop]),
        ("GT classes (evaluation only)", gt_view[crop]),
        ("Predicted building support", pred_view[crop]),
        (f"Matched grid ({metrics['grid']['region_count']} regions)", _boundary_overlay(rgb, id_full["grid"], building_full)[crop]),
        (f"RGB SLICO ({metrics['slico']['region_count']} regions)", _boundary_overlay(rgb, id_full["slico"], building_full)[crop]),
        (f"Learned hard view of Q ({metrics['learned']['region_count']} regions)", _boundary_overlay(rgb, id_full["learned"], building_full)[crop]),
        ("Learned association entropy", entropy_full[crop]),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 8.2))
    for ax, (title, view) in zip(axes.flat, views):
        if view.ndim == 2 and title != "Post-event SAR":
            image = ax.imshow(view, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
            figure.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        else:
            ax.imshow(view, cmap="gray" if view.ndim == 2 else None, vmin=0, vmax=1 if view.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    figure.suptitle(
        f"{sample_id} | same predicted-building support and region scale",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "GT selects/evaluates the diagnostic case only; Grid, SLICO and learned Q do not read GT during forward inference.",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.07, wspace=0.08, hspace=0.16)
    figure.savefig(args.output_dir / f"{sample_id}_partition_comparison.png", dpi=220)
    figure.savefig(args.output_dir / f"{sample_id}_partition_comparison.pdf")
    plt.close(figure)

    np.savez_compressed(
        args.output_dir / f"{sample_id}_partition_arrays.npz",
        target=target_full,
        predicted_building=building_full,
        grid=id_full["grid"],
        slico=id_full["slico"],
        learned=id_full["learned"],
        association_entropy=entropy_full,
    )
    report = {
        "protocol": "MATCHED_PARTITION_VISUAL_COMPARISON_V1",
        "sample_id": sample_id,
        "focus_class": args.focus_class,
        "variant": variant,
        "stage": stage,
        "feature_size": [h, w],
        "input_size": list(input_size),
        "crop_yx": [crop[0].start, crop[0].stop, crop[1].start, crop[1].stop],
        "metrics": metrics,
        "association_entropy_mean_on_predicted_building": float(
            association_entropy[building[0]].float().mean().cpu()
        ) if bool(building[0].any().item()) else 0.0,
        "parent": parent_report,
        "checkpoint": checkpoint_report,
        "gt_used_in_partition_forward": False,
        "gt_usage": "diagnostic sample selection, crop selection and post-hoc metrics only",
        "test_accessed": False,
    }
    (args.output_dir / f"{sample_id}_partition_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
