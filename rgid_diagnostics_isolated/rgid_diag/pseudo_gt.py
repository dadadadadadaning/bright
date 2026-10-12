from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    as_numpy_2d,
    binary_cc,
    damage_class_cc,
    instance_count,
    iter_cache,
    resolve_aux_label,
    write_csv,
    write_json,
)


def _comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    # Imported only after the runner has bootstrapped the original experiment.
    from rgid.metrics import evaluate_instance_labels

    return dict(evaluate_instance_labels(reference, candidate))


def audit_pseudo_gt(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path | None = None,
    damage_template: str = "target/{name}_building_damage.tif",
    true_instance_root: str | Path | None = None,
    true_instance_template: str = "{name}.tif",
    min_area: int = 32,
    ignore_index: int = 255,
    limit: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_damage: list[str] = []
    missing_true: list[str] = []
    aggregate = Counter()
    metric_rows: dict[str, list[dict[str, Any]]] = {"binary": [], "damage": []}

    for _, payload in iter_cache(cache_dir, limit=limit):
        name = str(payload["name"])
        cached = as_numpy_2d(payload["instance_ids"], "instance_ids").astype(np.int32)
        cached_binary = cached > 0
        reconstructed_binary = binary_cc(cached_binary, min_area=min_area)
        damage, damage_source = resolve_aux_label(
            payload, key="damage_label", root=dataset_root, template=damage_template
        )
        true_instances, true_source = resolve_aux_label(
            payload,
            key="true_instance_ids",
            root=true_instance_root,
            template=true_instance_template,
        )
        row: dict[str, Any] = {
            "name": name,
            "cached_instance_count": instance_count(cached),
            "binary_cc_count": instance_count(reconstructed_binary),
            "cache_matches_reconstructed_binary": bool(
                np.array_equal(cached, reconstructed_binary)
            ),
            "damage_source": damage_source or "",
            "true_instance_source": true_source or "",
        }
        aggregate["samples"] += 1
        aggregate["cached_instances"] += row["cached_instance_count"]
        aggregate["binary_instances"] += row["binary_cc_count"]

        if damage is None:
            missing_damage.append(name)
            row["damage_available"] = False
        else:
            if damage.shape != cached.shape:
                raise ValueError(
                    f"Damage label shape mismatch for {name}: {damage.shape} vs {cached.shape}"
                )
            class_instances = damage_class_cc(
                damage, min_area=min_area, ignore_index=ignore_index
            )
            class_count = instance_count(class_instances)
            split_binary_components = 0
            mixed_class_binary_components = 0
            for binary_id in range(1, int(reconstructed_binary.max()) + 1):
                region = reconstructed_binary == binary_id
                child_ids = np.unique(class_instances[region])
                child_ids = child_ids[child_ids > 0]
                damage_ids = np.unique(damage[region])
                damage_ids = damage_ids[(damage_ids >= 1) & (damage_ids <= 3)]
                split_binary_components += int(len(child_ids) > 1)
                mixed_class_binary_components += int(len(damage_ids) > 1)
            added = class_count - int(row["binary_cc_count"])
            row.update(
                damage_available=True,
                damage_class_cc_count=class_count,
                instances_added_by_class_separation=added,
                binary_components_split_by_class=split_binary_components,
                mixed_damage_binary_components=mixed_class_binary_components,
            )
            aggregate["damage_available_samples"] += 1
            aggregate["damage_class_instances"] += class_count
            aggregate["instances_added_by_class_separation"] += added
            aggregate["binary_components_split_by_class"] += split_binary_components
            aggregate["mixed_damage_binary_components"] += mixed_class_binary_components

        if true_instances is None:
            if true_instance_root is not None:
                missing_true.append(name)
            row["true_instance_available"] = False
        else:
            true_instances = as_numpy_2d(true_instances, "true_instance_ids").astype(np.int32)
            if true_instances.shape != cached.shape:
                raise ValueError(
                    f"True instance shape mismatch for {name}: {true_instances.shape} vs {cached.shape}"
                )
            row["true_instance_available"] = True
            binary_metrics = _comparison_metrics(true_instances, reconstructed_binary)
            metric_rows["binary"].append(binary_metrics)
            for key in ("precision", "recall", "f1", "mean_matched_iou", "merge", "split"):
                row[f"binary_vs_true_{key}"] = binary_metrics[key]
            if damage is not None:
                damage_metrics = _comparison_metrics(true_instances, class_instances)
                metric_rows["damage"].append(damage_metrics)
                for key in ("precision", "recall", "f1", "mean_matched_iou", "merge", "split"):
                    row[f"damage_vs_true_{key}"] = damage_metrics[key]
            aggregate["true_instance_available_samples"] += 1
        rows.append(row)

    summary: dict[str, Any] = dict(aggregate)
    available = int(aggregate["damage_available_samples"])
    summary["damage_label_coverage"] = available / max(int(aggregate["samples"]), 1)
    summary["mean_instances_added_per_damage_sample"] = (
        int(aggregate["instances_added_by_class_separation"]) / max(available, 1)
    )
    summary["important_limitation"] = (
        "Without true polygon/instance IDs this audit measures only the lower-bound "
        "error exposed by different damage classes; touching buildings of the same "
        "damage class remain unresolved."
    )
    if metric_rows["binary"]:
        from rgid.metrics import aggregate_metric_dicts

        summary["binary_cc_vs_true"] = dict(aggregate_metric_dicts(metric_rows["binary"]))
    if metric_rows["damage"]:
        from rgid.metrics import aggregate_metric_dicts

        summary["damage_class_cc_vs_true"] = dict(
            aggregate_metric_dicts(metric_rows["damage"])
        )
    report = {
        "protocol": "pseudo_instance_gt_quality_audit_v1",
        "min_area": int(min_area),
        "summary": summary,
        "missing_damage_labels": missing_damage,
        "missing_true_instance_labels": missing_true,
        "samples": rows,
    }
    output = Path(output_dir)
    write_json(output / "pseudo_gt_audit.json", report)
    write_csv(output / "pseudo_gt_samples.csv", rows)
    return report
