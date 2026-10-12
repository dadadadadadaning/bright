# -*- coding: utf-8 -*-
"""Compute frozen damage-class priors from the training split only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser("Compute H4 train-only damage priors")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--suffix", default=".tif")
    args = parser.parse_args()

    dataset = Path(args.dataset_path)
    train_list = Path(args.train_data_list_path)
    names = [
        line.strip()
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError("training list is empty")

    counts = np.zeros(4, dtype=np.int64)
    ignored = 0
    for index, name in enumerate(names, start=1):
        path = dataset / "target" / f"{name}_building_damage{args.suffix}"
        label = np.asarray(imageio.imread(path))
        if label.ndim == 3:
            label = label[..., 0]
        valid = (label >= 0) & (label <= 3)
        counts += np.bincount(label[valid].astype(np.int64), minlength=4)[:4]
        ignored += int((~valid).sum())
        if index % 100 == 0 or index == len(names):
            print(f"label prior pass: {index}/{len(names)}", flush=True)

    if np.any(counts <= 0):
        raise ValueError(f"each damage class must occur in training: {counts.tolist()}")
    payload = {
        "protocol": "H4_train_only_damage_priors_v1",
        "dataset_path": str(dataset.resolve()),
        "train_data_list_path": str(train_list.resolve()),
        "train_list_sha256": sha256(train_list),
        "train_image_count": len(names),
        "class_names": ["background", "intact", "damaged", "destroyed"],
        "pixel_counts": counts.tolist(),
        "pixel_priors": (counts / counts.sum()).tolist(),
        "ignored_pixel_count": int(ignored),
        "hierarchical_counts": {
            "building": [int(counts[0]), int(counts[1:].sum())],
            "affected": [int(counts[1]), int(counts[2:].sum())],
            "destroyed": [int(counts[2]), int(counts[3])],
        },
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved frozen H4 damage priors: {destination}", flush=True)


if __name__ == "__main__":
    main()
