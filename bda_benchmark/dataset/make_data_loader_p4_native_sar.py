# -*- coding: utf-8 -*-
"""Paired RGB/SAR loader for the S8-P4 A0...A4 controlled ablation.

The spatial augmentation is applied once to raw RGB, raw single-channel SAR
and the label.  Two SAR tensors are then produced:

* ``post_img``: variant-specific student input;
* ``teacher_post_img``: the exact P3 pseudo-RGB/ImageNet-normalized view.

This prevents teacher/student crops from silently becoming misaligned.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from torch.utils.data import Dataset

from configs.experiment_registry import P4_NATIVE_SAR_VARIANTS


IMAGENET_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
VALID_VARIANTS = P4_NATIVE_SAR_VARIANTS


def img_loader(path: str) -> np.ndarray:
    return np.asarray(imageio.imread(path), dtype=np.float32)


def read_data_list(path: str | os.PathLike[str]) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def to_single_channel_sar(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        channels = image[..., :3]
        if np.allclose(channels[..., 0], channels[..., 1]) and np.allclose(
            channels[..., 0], channels[..., 2]
        ):
            return channels[..., 0]
        raise ValueError(
            "post-event TIFF has non-identical RGB channels; declare the SAR "
            "representation instead of silently choosing one channel"
        )
    raise ValueError(f"expected scalar SAR [H,W] or repeated SAR [H,W,C], got {image.shape}")


def imagenet_normalize(image_hwc: np.ndarray) -> np.ndarray:
    image = np.asarray(image_hwc, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"ImageNet normalization expects HWC RGB, got {image.shape}")
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def load_sar_stats(path: str | os.PathLike[str]) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    source: Mapping[str, object] = payload.get("normalization", payload)
    required = ("p01", "p99", "mean", "std")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"SAR stats JSON is missing {missing}")
    result = {key: float(source[key]) for key in required}
    if not result["p99"] > result["p01"]:
        raise ValueError("SAR stats require p99 > p01")
    if not result["std"] > 0:
        raise ValueError("SAR stats require std > 0")
    return result


# Backward-compatible private alias for older imports/checkpoints.
_load_stats = load_sar_stats


def normalize_native_sar(sar: np.ndarray, stats: Mapping[str, float]) -> np.ndarray:
    clipped = np.clip(sar.astype(np.float32), stats["p01"], stats["p99"])
    return (clipped - stats["mean"]) / stats["std"]


def normalize_native_sar_affine(
    sar: np.ndarray, stats: Mapping[str, float]
) -> np.ndarray:
    """Invertible A4 normalisation used by the function-preserving fold.

    No clipping is allowed here: clipping would make the one-channel input no
    longer algebraically equivalent to P3's raw-display pseudo-RGB view.
    """

    if not float(stats["std"]) > 0.0:
        raise ValueError("SAR statistics require std > 0")
    return (sar.astype(np.float32) - float(stats["mean"])) / float(stats["std"])


def make_sar_views(
    raw_sar: np.ndarray,
    variant: str,
    stats: Mapping[str, float] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return student CHW and exact-P3 teacher CHW tensors."""

    variant = str(variant).lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown P4 variant {variant!r}")
    raw = to_single_channel_sar(raw_sar)
    pseudo_rgb = np.repeat(raw[..., None], 3, axis=2)
    teacher = imagenet_normalize(pseudo_rgb).transpose(2, 0, 1)
    if variant == "a0":
        student = teacher.copy()
    else:
        if stats is None:
            raise ValueError(f"{variant.upper()} requires train-only SAR statistics")
        native = (
            normalize_native_sar_affine(raw, stats)
            if variant == "a4"
            else normalize_native_sar(raw, stats)
        )
        if variant == "a1":
            student = np.repeat(native[None, ...], 3, axis=0)
        else:
            student = native[None, ...]
    return (
        np.ascontiguousarray(student, dtype=np.float32),
        np.ascontiguousarray(teacher, dtype=np.float32),
    )


def _random_crop(
    rgb: np.ndarray,
    sar: np.ndarray,
    label: np.ndarray,
    crop_size: int,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = label.shape
    height, width = max(crop_size, h), max(crop_size, w)
    pad_rgb = np.zeros((height, width, 3), dtype=np.float32)
    pad_sar = np.zeros((height, width), dtype=np.float32)
    pad_label = np.full((height, width), ignore_index, dtype=np.float32)
    y_pad = int(np.random.randint(height - h + 1))
    x_pad = int(np.random.randint(width - w + 1))
    pad_rgb[y_pad : y_pad + h, x_pad : x_pad + w] = rgb
    pad_sar[y_pad : y_pad + h, x_pad : x_pad + w] = sar
    pad_label[y_pad : y_pad + h, x_pad : x_pad + w] = label

    y0 = x0 = 0
    for _ in range(10):
        y0 = random.randrange(0, height - crop_size + 1)
        x0 = random.randrange(0, width - crop_size + 1)
        crop_label = pad_label[y0 : y0 + crop_size, x0 : x0 + crop_size]
        _, counts = np.unique(crop_label[crop_label != ignore_index], return_counts=True)
        if counts.size > 1 and float(counts.max()) / float(counts.sum()) < 0.75:
            break
    y1, x1 = y0 + crop_size, x0 + crop_size
    return pad_rgb[y0:y1, x0:x1], pad_sar[y0:y1, x0:x1], pad_label[y0:y1, x0:x1]


def _spatial_augment(
    rgb: np.ndarray,
    sar: np.ndarray,
    label: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb, sar, label = _random_crop(rgb, sar, label, crop_size)
    if random.random() > 0.5:
        rgb, sar, label = np.fliplr(rgb), np.fliplr(sar), np.fliplr(label)
    if random.random() > 0.5:
        rgb, sar, label = np.flipud(rgb), np.flipud(sar), np.flipud(label)
    # Keep the established P3 policy: always rotate by 90, 180 or 270 degrees.
    k = random.randrange(3) + 1
    return (
        np.rot90(rgb, k).copy(),
        np.rot90(sar, k).copy(),
        np.rot90(label, k).copy(),
    )


class P4NativeSARDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_list: Sequence[str],
        crop_size: int,
        split: str,
        variant: str,
        sar_stats_json: str = "",
        suffix: str = ".tif",
        data_loader=img_loader,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.data_list = list(data_list)
        self.crop_size = int(crop_size)
        self.split = str(split).lower()
        self.variant = str(variant).lower()
        self.suffix = str(suffix)
        self.loader = data_loader
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}")
        self.sar_stats_json = str(sar_stats_json)
        self.sar_stats = (
            load_sar_stats(self.sar_stats_json) if self.variant != "a0" else None
        )
        if not self.data_list:
            raise ValueError("data_list is empty")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict[str, object]:
        data_id = self.data_list[index]
        pre_path = os.path.join(
            self.dataset_path, "pre-event", data_id + "_pre_disaster" + self.suffix
        )
        post_path = os.path.join(
            self.dataset_path, "post-event", data_id + "_post_disaster" + self.suffix
        )
        label_path = os.path.join(
            self.dataset_path, "target", data_id + "_building_damage" + self.suffix
        )
        pre = np.asarray(self.loader(pre_path), dtype=np.float32)[..., :3]
        sar = to_single_channel_sar(self.loader(post_path))
        damage = np.asarray(self.loader(label_path), dtype=np.float32)
        if damage.ndim == 3:
            damage = damage[..., 0]
        if not (pre.shape[:2] == sar.shape == damage.shape):
            raise ValueError(
                f"unaligned sample {data_id}: RGB={pre.shape}, SAR={sar.shape}, "
                f"label={damage.shape}"
            )
        if not np.isfinite(pre).all():
            raise ValueError(f"non-finite RGB values in sample {data_id}")
        if not np.isfinite(sar).all():
            raise ValueError(f"non-finite SAR values in sample {data_id}")
        if not np.isfinite(damage).all():
            raise ValueError(f"non-finite labels in sample {data_id}")
        if "train" in self.split:
            pre, sar, damage = _spatial_augment(pre, sar, damage, self.crop_size)

        pre_chw = imagenet_normalize(pre).transpose(2, 0, 1)
        student_sar, teacher_sar = make_sar_views(
            sar, self.variant, self.sar_stats
        )
        location = damage.copy()
        location[(location == 2) | (location == 3)] = 1
        return {
            "pre_img": np.ascontiguousarray(pre_chw, dtype=np.float32),
            "post_img": student_sar,
            "teacher_post_img": teacher_sar,
            "label_loc": np.ascontiguousarray(location, dtype=np.int64),
            "label_dam": np.ascontiguousarray(damage, dtype=np.int64),
            "name": data_id,
        }


def build_p4_dataset(
    dataset_path: str,
    data_list_path: str,
    crop_size: int,
    split: str,
    variant: str,
    sar_stats_json: str = "",
) -> P4NativeSARDataset:
    return P4NativeSARDataset(
        dataset_path=dataset_path,
        data_list=read_data_list(data_list_path),
        crop_size=crop_size,
        split=split,
        variant=variant,
        sar_stats_json=sar_stats_json,
    )


__all__ = [
    "P4NativeSARDataset",
    "build_p4_dataset",
    "imagenet_normalize",
    "make_sar_views",
    "load_sar_stats",
    "normalize_native_sar",
    "normalize_native_sar_affine",
    "read_data_list",
    "to_single_channel_sar",
]
