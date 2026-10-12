"""BRIGHT dataset that exposes post-event SAR as one affine channel."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader

from changedetection.datasets import imutils
from changedetection.datasets.common import BaseChangeDataset, img_loader


def load_sar_stats(path: str) -> Dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("normalization", payload)
    missing = [name for name in ("mean", "std") if name not in source]
    if missing:
        raise ValueError(f"SAR stats are missing {missing}")
    stats = {"mean": float(source["mean"]), "std": float(source["std"])}
    if not stats["std"] > 0.0:
        raise ValueError("SAR std must be positive")
    return stats


class NativeSARBRIGHTDataset(BaseChangeDataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        sar_stats_path,
        max_iters=None,
        batch_size=1,
        split="train",
        suffix=".tif",
        data_loader=img_loader,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            data_loader=data_loader,
        )
        self.suffix = suffix
        self.sar_stats = load_sar_stats(sar_stats_path)

    def __getitem__(self, index):
        item_name = self.data_list[index]
        pre_path = os.path.join(
            self.dataset_path, "pre-event", f"{item_name}_pre_disaster{self.suffix}"
        )
        post_path = os.path.join(
            self.dataset_path, "post-event", f"{item_name}_post_disaster{self.suffix}"
        )
        label_path = os.path.join(
            self.dataset_path, "target", f"{item_name}_building_damage{self.suffix}"
        )

        pre_img = self.loader(pre_path)[..., :3]
        raw_sar = np.asarray(self.loader(post_path), dtype=np.float32)
        if raw_sar.ndim == 3:
            raw_sar = raw_sar[..., 0]
        if raw_sar.ndim != 2:
            raise ValueError(f"SAR image must be 2D, got {raw_sar.shape}")
        post_img = raw_sar[..., None]
        clf_label = self.loader(label_path)

        if self.is_train:
            pre_img, post_img, clf_label = imutils.random_crop_new(
                pre_img, post_img, clf_label, self.crop_size, mean_rgb=0.0
            )
            pre_img, post_img, clf_label = imutils.random_fliplr(
                pre_img, post_img, clf_label
            )
            pre_img, post_img, clf_label = imutils.random_flipud(
                pre_img, post_img, clf_label
            )
            pre_img, post_img, clf_label = imutils.random_rot(
                pre_img, post_img, clf_label
            )

        pre_tensor = imutils.to_channel_first(imutils.normalize_img(pre_img))
        post_tensor = imutils.to_channel_first(
            (post_img.astype(np.float32) - self.sar_stats["mean"])
            / self.sar_stats["std"]
        )
        clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[(loc_label == 2) | (loc_label == 3)] = 1
        return pre_tensor, post_tensor, loc_label, clf_label, item_name


def build_native_sar_loader(
    *,
    dataset_path,
    data_list,
    crop_size,
    sar_stats_path,
    split,
    batch_size,
    num_workers,
    max_iters=None,
    shuffle=False,
):
    dataset = NativeSARBRIGHTDataset(
        dataset_path=dataset_path,
        data_list=data_list,
        crop_size=crop_size,
        sar_stats_path=sar_stats_path,
        max_iters=max_iters,
        batch_size=batch_size,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

