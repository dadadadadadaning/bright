from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .common import (
    as_numpy_2d,
    binary_cc,
    damage_class_cc,
    edt_center,
    iter_cache,
    quantiles,
    resolve_aux_label,
    write_csv,
    write_json,
)


def select_gt_labels(
    payload: dict[str, Any],
    *,
    source: str,
    dataset_root: str | Path | None,
    damage_template: str,
    true_instance_root: str | Path | None,
    true_instance_template: str,
    min_area: int,
) -> tuple[np.ndarray, str]:
    cached = as_numpy_2d(payload["instance_ids"], "instance_ids").astype(np.int32)
    if source == "cached":
        return cached, "cache:instance_ids"
    if source == "damage":
        damage, path = resolve_aux_label(
            payload, key="damage_label", root=dataset_root, template=damage_template
        )
        if damage is None:
            raise FileNotFoundError(f"Missing damage label for {payload['name']}: {path}")
        return damage_class_cc(damage, min_area=min_area), str(path)
    if source == "true":
        labels, path = resolve_aux_label(
            payload,
            key="true_instance_ids",
            root=true_instance_root,
            template=true_instance_template,
        )
        if labels is None:
            raise FileNotFoundError(f"Missing true instance label for {payload['name']}: {path}")
        return as_numpy_2d(labels, "true_instance_ids").astype(np.int32), str(path)
    raise ValueError("GT source must be cached, damage, or true")


def analyze_parents(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    gt_source: str = "cached",
    dataset_root: str | Path | None = None,
    damage_template: str = "target/{name}_building_damage.tif",
    true_instance_root: str | Path | None = None,
    true_instance_template: str = "{name}.tif",
    min_area: int = 32,
    min_overlap_pixels: int = 1,
    min_gt_fraction_in_parent: float = 0.0,
    max_markers: int = 4,
    head_stride: float = 4.0,
    min_peak_distance_head: float = 3.0,
    limit: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    count_hist = Counter()
    nearest_original: list[float] = []
    nearest_head: list[float] = []
    totals = Counter()

    for _, payload in iter_cache(cache_dir, limit=limit):
        name = str(payload["name"])
        parent_mask = as_numpy_2d(payload["parent_mask"], "parent_mask") > 0
        parents = binary_cc(parent_mask, min_area=1)
        gt, source_path = select_gt_labels(
            payload,
            source=gt_source,
            dataset_root=dataset_root,
            damage_template=damage_template,
            true_instance_root=true_instance_root,
            true_instance_template=true_instance_template,
            min_area=min_area,
        )
        if gt.shape != parents.shape:
            raise ValueError(f"Shape mismatch for {name}: GT {gt.shape}, parent {parents.shape}")
        totals["samples"] += 1
        totals["gt_instances"] += int(len(np.unique(gt)) - (0 in np.unique(gt)))
        for parent_id in range(1, int(parents.max()) + 1):
            parent = parents == parent_id
            child_ids: list[int] = []
            centers: list[tuple[int, int]] = []
            for gt_id in (int(v) for v in np.unique(gt[parent]) if int(v) > 0):
                instance = gt == gt_id
                overlap = int((instance & parent).sum())
                fraction = overlap / max(int(instance.sum()), 1)
                if overlap < int(min_overlap_pixels) or fraction < float(min_gt_fraction_in_parent):
                    continue
                child_ids.append(gt_id)
                centers.append(edt_center(instance & parent))
            count = len(child_ids)
            bucket = str(count) if count < 5 else "5+"
            count_hist[bucket] += 1
            totals["parents"] += 1
            totals["multi_instance_parents"] += int(count >= 2)
            totals["parents_over_marker_cap"] += int(count > int(max_markers))
            parent_distances: list[float] = []
            if len(centers) >= 2:
                coords = np.asarray(centers, dtype=np.float64)
                matrix = np.sqrt(np.square(coords[:, None] - coords[None, :]).sum(axis=2))
                np.fill_diagonal(matrix, np.inf)
                parent_distances = matrix.min(axis=1).tolist()
                nearest_original.extend(parent_distances)
                nearest_head.extend([value / float(head_stride) for value in parent_distances])
            rows.append(
                {
                    "name": name,
                    "parent_id": parent_id,
                    "parent_area": int(parent.sum()),
                    "gt_children": count,
                    "over_marker_cap": count > int(max_markers),
                    "min_nearest_center_distance_px": min(parent_distances) if parent_distances else "",
                    "mean_nearest_center_distance_px": (
                        float(np.mean(parent_distances)) if parent_distances else ""
                    ),
                    "gt_source": source_path,
                }
            )

    threshold_original = float(min_peak_distance_head) * float(head_stride)
    distance_array = np.asarray(nearest_original, dtype=np.float64)
    report = {
        "protocol": "parent_gt_count_and_center_distance_v1",
        "gt_source": gt_source,
        "parameters": {
            "min_overlap_pixels": int(min_overlap_pixels),
            "min_gt_fraction_in_parent": float(min_gt_fraction_in_parent),
            "max_markers": int(max_markers),
            "head_stride": float(head_stride),
            "min_peak_distance_head": float(min_peak_distance_head),
            "equivalent_min_peak_distance_original_px": threshold_original,
        },
        "summary": {
            **dict(totals),
            "gt_children_per_parent_histogram": dict(count_hist),
            "multi_instance_parent_fraction": int(totals["multi_instance_parents"]) / max(int(totals["parents"]), 1),
            "parent_over_marker_cap_fraction": int(totals["parents_over_marker_cap"]) / max(int(totals["parents"]), 1),
            "nearest_center_distance_original_px": quantiles(nearest_original),
            "nearest_center_distance_head_px": quantiles(nearest_head),
            "nearest_centers_below_current_nms_distance": int((distance_array < threshold_original).sum()),
            "nearest_centers_below_current_nms_fraction": float((distance_array < threshold_original).mean()) if distance_array.size else 0.0,
        },
        "parents": rows,
    }
    output = Path(output_dir)
    write_json(output / "parent_center_stats.json", report)
    write_csv(output / "parent_stats.csv", rows)
    return report
