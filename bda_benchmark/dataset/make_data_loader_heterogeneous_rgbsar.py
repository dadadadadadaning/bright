# -*- coding: utf-8 -*-
"""Paired RGB/SAR loader for the heterogeneous DamageFormer branch.

RGB and SAR receive the same geometric transform but different radiometric
preprocessing.  H0 receives the historical repeated three-channel SAR view;
H1-H3 receive one native SAR channel normalized with train-split-only stats.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from torch.utils.data import Dataset

from bda_benchmark.configs.experiment_registry import (
    HETEROGENEOUS_RGBSAR_VARIANTS,
)


RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
VALID_HETEROGENEOUS_RGBSAR_VARIANTS = HETEROGENEOUS_RGBSAR_VARIANTS


def read_data_list(path: str | os.PathLike[str]) -> List[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read(path: str | os.PathLike[str]) -> np.ndarray:
    return np.asarray(imageio.imread(path), dtype=np.float32)


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
            "post-event image has non-identical channels; this loader requires "
            "a scalar SAR representation and will not silently discard bands"
        )
    raise ValueError(f"expected SAR [H,W] or repeated [H,W,C], got {image.shape}")


def load_sar_stats(path: str | os.PathLike[str]) -> Dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source: Mapping[str, object] = payload.get("normalization", payload)
    required = ("p01", "p99", "mean", "std")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"SAR stats are missing fields: {missing}")
    stats = {key: float(source[key]) for key in required}
    if not stats["p99"] > stats["p01"]:
        raise ValueError("SAR stats require p99 > p01")
    if not stats["std"] > 0:
        raise ValueError("SAR stats require std > 0")
    return stats


def _pad_and_crop(
    rgb: np.ndarray,
    sar: np.ndarray,
    label: np.ndarray,
    crop_size: int,
    focus_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = label.shape
    padded_height = max(height, crop_size)
    padded_width = max(width, crop_size)
    rgb_pad = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    sar_pad = np.zeros((padded_height, padded_width), dtype=np.float32)
    label_pad = np.full((padded_height, padded_width), 255, dtype=np.float32)
    top = random.randint(0, padded_height - height)
    left = random.randint(0, padded_width - width)
    rgb_pad[top : top + height, left : left + width] = rgb
    sar_pad[top : top + height, left : left + width] = sar
    label_pad[top : top + height, left : left + width] = label
    focus = None
    if focus_class is not None:
        candidates = np.argwhere(label == int(focus_class))
        if candidates.size:
            row, column = candidates[random.randrange(len(candidates))]
            focus = (int(row) + top, int(column) + left)
    if focus is None:
        crop_top = random.randint(0, padded_height - crop_size)
        crop_left = random.randint(0, padded_width - crop_size)
    else:
        row, column = focus
        top_min = max(0, row - crop_size + 1)
        top_max = min(row, padded_height - crop_size)
        left_min = max(0, column - crop_size + 1)
        left_max = min(column, padded_width - crop_size)
        crop_top = random.randint(top_min, max(top_min, top_max))
        crop_left = random.randint(left_min, max(left_min, left_max))
    rows = slice(crop_top, crop_top + crop_size)
    cols = slice(crop_left, crop_left + crop_size)
    return rgb_pad[rows, cols], sar_pad[rows, cols], label_pad[rows, cols]


def _joint_augmentation(
    rgb: np.ndarray,
    sar: np.ndarray,
    label: np.ndarray,
    crop_size: int,
    focus_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb, sar, label = _pad_and_crop(
        rgb, sar, label, crop_size, focus_class=focus_class
    )
    if random.random() < 0.5:
        rgb, sar, label = np.fliplr(rgb), np.fliplr(sar), np.fliplr(label)
    if random.random() < 0.5:
        rgb, sar, label = np.flipud(rgb), np.flipud(sar), np.flipud(label)
    rotations = random.randint(0, 3)
    if rotations:
        rgb = np.rot90(rgb, rotations)
        sar = np.rot90(sar, rotations)
        label = np.rot90(label, rotations)
    return rgb.copy(), sar.copy(), label.copy()


class HeterogeneousRGBSARDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_list: List[str] | str,
        crop_size: int,
        max_iters: int | None = None,
        split: str = "train",
        variant: str = "h3",
        sar_stats_path: str = "",
        suffix: str = ".tif",
        focus_probability: float = 0.0,
        focus_class_weights: Sequence[float] = (0.0, 1.0, 2.0, 2.0),
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.data_list = (
            read_data_list(data_list) if isinstance(data_list, str) else list(data_list)
        )
        if not self.data_list:
            raise ValueError("data list is empty")
        if max_iters is not None:
            repeats = int(math.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = (self.data_list * repeats)[: int(max_iters)]
        self.crop_size = int(crop_size)
        self.split = str(split).lower()
        self.variant = str(variant).lower()
        # x1/x2 are whole-model external-CD baselines.  Their data contract is
        # identical to the native one-channel H1-H3 path; only the network
        # factory changes.  Keep h0 as the sole pseudo-RGB exception.
        if self.variant not in VALID_HETEROGENEOUS_RGBSAR_VARIANTS:
            raise ValueError(
                "variant must be h0-h4, x1/x2, or an OA-HSDNet n0/n1_shared/"
                "n1_stage/n1_cond/n2 or P3-SRAC "
                "sr0/sr1/sr1f/sr1p/sr1pa/sr1pb/sr1pai/sr1pib/sr1paa/sr2 variant"
            )
        self.suffix = suffix
        self.focus_probability = float(focus_probability)
        if not 0.0 <= self.focus_probability <= 1.0:
            raise ValueError("focus_probability must lie in [0, 1]")
        if len(focus_class_weights) != 4 or min(focus_class_weights) < 0:
            raise ValueError("focus_class_weights must contain four non-negative values")
        self.focus_class_weights = tuple(float(value) for value in focus_class_weights)
        self.sar_stats = None
        if self.variant != "h0":
            if not sar_stats_path:
                raise ValueError(f"{self.variant.upper()} requires --sar_stats_path")
            self.sar_stats = load_sar_stats(sar_stats_path)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        name = self.data_list[index]
        pre_path = self.dataset_path / "pre-event" / f"{name}_pre_disaster{self.suffix}"
        post_path = self.dataset_path / "post-event" / f"{name}_post_disaster{self.suffix}"
        label_path = self.dataset_path / "target" / f"{name}_building_damage{self.suffix}"
        rgb = _read(pre_path)[..., :3]
        sar = to_single_channel_sar(_read(post_path))
        label = _read(label_path)
        if label.ndim == 3:
            label = label[..., 0]
        if rgb.shape[:2] != sar.shape or sar.shape != label.shape:
            raise ValueError(
                f"unaligned sample {name}: RGB={rgb.shape}, SAR={sar.shape}, "
                f"label={label.shape}"
            )
        if self.split == "train":
            focus_class = None
            if self.focus_probability > 0.0 and random.random() < self.focus_probability:
                present = []
                weights = []
                for class_index, class_weight in enumerate(self.focus_class_weights):
                    if class_weight > 0.0 and np.any(label == class_index):
                        present.append(class_index)
                        weights.append(class_weight)
                if present:
                    focus_class = random.choices(present, weights=weights, k=1)[0]
            rgb, sar, label = _joint_augmentation(
                rgb, sar, label, self.crop_size, focus_class=focus_class
            )

        rgb = ((rgb.astype(np.float32) - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        if self.variant == "h0":
            pseudo_rgb = np.repeat(sar[..., None], 3, axis=2)
            sar_tensor = ((pseudo_rgb - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        else:
            assert self.sar_stats is not None
            raw_sar = sar.astype(np.float32)
            clipped_sar = np.clip(
                raw_sar, self.sar_stats["p01"], self.sar_stats["p99"]
            )
            native_sar = (
                clipped_sar - self.sar_stats["mean"]
            ) / self.sar_stats["std"]
            if self.variant in (
                "h4", "sr0", "sr1", "sr1f", "sr1p", "sr1pa", "sr1pb",
                "sr1pai", "sr1pib", "sr1paa", "sr2"
            ):
                # Channel 0 is the declared native-SAR training view.
                # Channel 1 retains the un-clipped display value solely so a frozen P3 branch can
                # reconstruct its historical pseudo-RGB input exactly.
                p3_audit_sar = raw_sar
                sar_tensor = np.stack([native_sar, p3_audit_sar], axis=0)
            else:
                sar_tensor = native_sar[None, ...]

        damage = label.astype(np.int64)
        localization = damage.copy()
        building = (localization >= 1) & (localization <= 3)
        localization[building] = 1
        localization[localization == 255] = 255
        return (
            np.ascontiguousarray(rgb, dtype=np.float32),
            np.ascontiguousarray(sar_tensor, dtype=np.float32),
            np.ascontiguousarray(localization, dtype=np.int64),
            np.ascontiguousarray(damage, dtype=np.int64),
            name,
        )


__all__ = [
    "HeterogeneousRGBSARDataset",
    "load_sar_stats",
    "read_data_list",
    "to_single_channel_sar",
]
