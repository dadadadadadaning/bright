from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def bootstrap(experiment_root: str | Path) -> Path:
    root = Path(experiment_root).expanduser().resolve()
    if not (root / "rgid").is_dir():
        raise NotADirectoryError(f"RGID experiment root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def torch_load(path: str | Path, *, weights_only: bool = True) -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_label(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    elif path.suffix.lower() in {".pt", ".pth"}:
        value = torch_load(path, weights_only=False)
        if isinstance(value, Mapping):
            for key in ("damage_label", "label", "mask", "instance_ids"):
                if key in value:
                    value = value[key]
                    break
    else:
        try:
            import imageio.v2 as imageio

            value = imageio.imread(path)
        except ImportError:
            from PIL import Image

            value = np.asarray(Image.open(path))
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.squeeze(np.asarray(value))
    if array.ndim != 2:
        raise ValueError(f"Label must be 2D after squeeze: {path}, {array.shape}")
    return array


def compact_labels(labels: np.ndarray, min_area: int = 1) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for old_id in (int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == old_id
        if int(mask.sum()) >= int(min_area):
            output[mask] = next_id
            next_id += 1
    return output


def binary_cc(mask: np.ndarray, min_area: int = 32) -> np.ndarray:
    from scipy import ndimage

    labels, _ = ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=ndimage.generate_binary_structure(2, 1),
    )
    return compact_labels(labels, min_area=min_area)


def damage_class_cc(
    damage: np.ndarray,
    *,
    class_ids: Iterable[int] = (1, 2, 3),
    ignore_index: int = 255,
    min_area: int = 32,
) -> np.ndarray:
    damage = np.asarray(damage)
    output = np.zeros(damage.shape, dtype=np.int32)
    next_id = 1
    for class_id in class_ids:
        components = binary_cc(
            (damage == int(class_id)) & (damage != int(ignore_index)),
            min_area=min_area,
        )
        for component_id in range(1, int(components.max()) + 1):
            mask = components == component_id
            if mask.any():
                output[mask] = next_id
                next_id += 1
    return output


class VariantCacheDataset(Dataset):
    def __init__(self, cache_dir: str | Path, sidecar_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.sidecar_dir = (
            Path(sidecar_dir).expanduser().resolve() if sidecar_dir else None
        )
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cache .pt files: {self.cache_dir}")
        if self.sidecar_dir is not None:
            missing = [p.name for p in self.files if not (self.sidecar_dir / p.name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(missing)} sidecars; first={missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.files[int(index)]
        payload = torch_load(source)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Cache payload is not a mapping: {source}")
        instances = payload["instance_ids"]
        gt_variant = "binary_cc"
        if self.sidecar_dir is not None:
            sidecar = torch_load(self.sidecar_dir / source.name)
            instances = sidecar["instance_ids"]
            gt_variant = str(sidecar.get("gt_variant", "damage_class_cc"))
        return {
            "fused_roi": torch.as_tensor(payload["fused_roi"]).float().contiguous(),
            "instance_ids": torch.as_tensor(instances).long().contiguous(),
            "parent_mask": torch.as_tensor(payload["parent_mask"]).bool().contiguous(),
            "name": str(payload["name"]),
            "source_file": source.name,
            "gt_variant": gt_variant,
        }


def collate_variant(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "fused_roi": torch.stack([torch.as_tensor(s["fused_roi"]) for s in samples]),
        "instance_ids": torch.stack([torch.as_tensor(s["instance_ids"]) for s in samples]),
        "parent_mask": torch.stack([torch.as_tensor(s["parent_mask"]) for s in samples]),
        "name": [str(s["name"]) for s in samples],
        "source_file": [str(s["source_file"]) for s in samples],
        "gt_variant": [str(s["gt_variant"]) for s in samples],
    }


def parse_steps(value: str, final_step: int) -> tuple[int, ...]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    return tuple(step for step in steps if 0 < step <= final_step)

