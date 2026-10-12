from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def bootstrap_project(project_root: str | Path) -> tuple[Path, Path]:
    """Add BRIGHT and the isolated RGID experiment to sys.path, read-only."""
    root = Path(project_root).expanduser().resolve()
    if (root / "bda_benchmark").is_dir():
        bda_root = root / "bda_benchmark"
        package_parent = root
    elif root.name == "bda_benchmark" and (root / "experiments").is_dir():
        bda_root = root
        package_parent = root.parent
    else:
        raise FileNotFoundError(
            f"Cannot find bda_benchmark under project root: {root}"
        )
    experiment_root = bda_root / "experiments" / "rgid_s9ic_clean"
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"Missing RGID experiment: {experiment_root}")
    for item in (package_parent, bda_root, experiment_root):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    return bda_root, experiment_root


def torch_load_cpu(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def iter_cache(cache_dir: str | Path, limit: int | None = None):
    cache = Path(cache_dir).expanduser().resolve()
    paths = sorted(cache.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt cache samples found in {cache}")
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    for path in paths:
        payload = torch_load_cpu(path)
        required = ("instance_ids", "parent_mask", "name")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        yield path, payload


def as_numpy_2d(value: Any, name: str) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got {array.shape}")
    return array


def read_label(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return as_numpy_2d(np.load(path), str(path))
    if suffix in {".pt", ".pth"}:
        value = torch_load_cpu(path)
        if isinstance(value, dict):
            for key in ("label", "damage_label", "instance_ids", "mask"):
                if key in value:
                    value = value[key]
                    break
        return as_numpy_2d(value, str(path))
    try:
        import imageio.v2 as imageio

        return as_numpy_2d(imageio.imread(path), str(path))
    except ImportError:
        from PIL import Image

        return as_numpy_2d(np.asarray(Image.open(path)), str(path))


def resolve_aux_label(
    payload: dict[str, Any],
    *,
    key: str,
    root: str | Path | None,
    template: str,
) -> tuple[np.ndarray | None, str | None]:
    if key in payload:
        return as_numpy_2d(payload[key], key), f"cache:{key}"
    if root is None:
        return None, None
    name = str(payload["name"])
    path = Path(root).expanduser().resolve() / template.format(name=name)
    if not path.is_file():
        return None, str(path)
    return read_label(path), str(path)


def compact_labels(labels: np.ndarray, min_area: int = 1) -> np.ndarray:
    source = np.asarray(labels)
    output = np.zeros(source.shape, dtype=np.int32)
    next_id = 1
    for old_id in (int(v) for v in np.unique(source) if int(v) > 0):
        mask = source == old_id
        if int(mask.sum()) < int(min_area):
            continue
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
    min_area: int = 32,
    class_ids: Iterable[int] = (1, 2, 3),
    ignore_index: int = 255,
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


def instance_count(labels: np.ndarray) -> int:
    return int(sum(int(v) > 0 for v in np.unique(labels)))


def edt_center(mask: np.ndarray) -> tuple[int, int]:
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Cannot compute a center for an empty mask")
    distance = ndimage.distance_transform_edt(mask)
    candidates = np.argwhere(distance == distance.max())
    centroid = np.argwhere(mask).mean(axis=0)
    squared = np.square(candidates - centroid[None, :]).sum(axis=1)
    order = np.lexsort((candidates[:, 1], candidates[:, 0], squared))
    y, x = candidates[int(order[0])]
    return int(y), int(x)


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p10": None, "p25": None,
                "median": None, "p75": None, "p90": None, "max": None,
                "mean": None}
    qs = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(array.size), "min": float(array.min()),
        "p10": float(qs[0]), "p25": float(qs[1]),
        "median": float(qs[2]), "p75": float(qs[3]),
        "p90": float(qs[4]), "max": float(array.max()),
        "mean": float(array.mean()),
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row}) if rows else ["name"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
