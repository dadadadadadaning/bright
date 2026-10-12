from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset


def _load_cache_file(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"cache file must contain a mapping: {path}")
    return payload


def _validate_sample(payload: Mapping[str, object], source: str) -> Dict[str, object]:
    required = ("fused_roi", "instance_ids", "parent_mask", "name")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"cache sample {source} is missing keys: {missing}")

    fused_roi = torch.as_tensor(payload["fused_roi"]).detach().cpu().float()
    instance_ids = torch.as_tensor(payload["instance_ids"]).detach().cpu().long()
    parent_mask = torch.as_tensor(payload["parent_mask"]).detach().cpu().bool()
    name = payload["name"]
    if fused_roi.ndim != 3 or int(fused_roi.shape[0]) != 256:
        raise ValueError(
            f"cache sample {source} fused_roi must be [256,Hh,Wh], "
            f"got {tuple(fused_roi.shape)}"
        )
    if instance_ids.ndim != 2:
        raise ValueError(
            f"cache sample {source} instance_ids must be [H,W], "
            f"got {tuple(instance_ids.shape)}"
        )
    if parent_mask.ndim != 2:
        raise ValueError(
            f"cache sample {source} parent_mask must be [H,W], "
            f"got {tuple(parent_mask.shape)}"
        )
    if instance_ids.shape != parent_mask.shape:
        raise ValueError(
            f"cache sample {source} instance_ids and parent_mask differ: "
            f"{tuple(instance_ids.shape)} vs {tuple(parent_mask.shape)}"
        )
    if not isinstance(name, str):
        raise TypeError(f"cache sample {source} name must be str, got {type(name)!r}")
    if not torch.isfinite(fused_roi).all():
        raise ValueError(f"cache sample {source} fused_roi contains NaN or Inf")
    if bool((instance_ids < 0).any().item()):
        raise ValueError(f"cache sample {source} instance_ids contains negatives")

    output: Dict[str, object] = {
        "fused_roi": fused_roi.contiguous(),
        "instance_ids": instance_ids.contiguous(),
        "parent_mask": parent_mask.contiguous(),
        "name": name,
    }
    if "metadata" in payload:
        output["metadata"] = payload["metadata"]
    return output


class RGIDCacheDataset(Dataset):
    """Read independently exported RGID ``.pt`` samples from one directory."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise NotADirectoryError(f"RGID cache directory not found: {self.cache_dir}")
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"RGID cache contains no .pt files: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[int(index)]
        return _validate_sample(_load_cache_file(path), str(path))


def collate_cached_samples(samples: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty RGID sample list")
    validated: List[Dict[str, object]] = [
        _validate_sample(sample, f"batch sample {index}")
        for index, sample in enumerate(samples)
    ]
    reference_shapes = {
        key: tuple(validated[0][key].shape)
        for key in ("fused_roi", "instance_ids", "parent_mask")
    }
    for index, sample in enumerate(validated[1:], start=1):
        for key, expected in reference_shapes.items():
            observed = tuple(sample[key].shape)
            if observed != expected:
                raise ValueError(
                    f"cannot collate RGID batch: sample {index} {key} shape "
                    f"{observed} differs from sample 0 shape {expected}"
                )
    return {
        "fused_roi": torch.stack(
            [sample["fused_roi"] for sample in validated], dim=0
        ),
        "instance_ids": torch.stack(
            [sample["instance_ids"] for sample in validated], dim=0
        ),
        "parent_mask": torch.stack(
            [sample["parent_mask"] for sample in validated], dim=0
        ),
        "name": [str(sample["name"]) for sample in validated],
    }


__all__ = ["RGIDCacheDataset", "collate_cached_samples"]
