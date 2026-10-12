#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from variant_common import (
    VariantCacheDataset,
    atomic_json,
    binary_cc,
    bootstrap,
)


def _map(value: int, source: int, target: int) -> int:
    mapped = round(((float(value) + 0.5) * target / source) - 0.5)
    return int(np.clip(mapped, 0, target - 1))


def _edt_center(mask: np.ndarray) -> tuple[int, int]:
    distance = ndimage.distance_transform_edt(mask)
    candidates = np.argwhere(distance == distance.max())
    centroid = np.argwhere(mask).mean(axis=0)
    squared = np.square(candidates - centroid[None]).sum(axis=1)
    order = np.lexsort((candidates[:, 1], candidates[:, 0], squared))
    y, x = candidates[int(order[0])]
    return int(y), int(x)


def _head_parents(parents: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    value = torch.from_numpy(parents.astype(np.float32))[None, None]
    return (
        F.interpolate(value, size=size, mode="nearest")[0, 0]
        .numpy()
        .astype(np.int32)
    )


def _plateau_peaks(
    probability: np.ndarray,
    head_parents: np.ndarray,
    *,
    kernel: int,
    threshold: float | None,
) -> list[tuple[int, int, float, int]]:
    tensor = torch.from_numpy(probability)[None, None]
    maxima = F.max_pool2d(
        tensor, kernel_size=kernel, stride=1, padding=kernel // 2
    )[0, 0].numpy()
    mask = (probability == maxima) & (head_parents > 0)
    if threshold is not None:
        mask &= probability >= float(threshold)
    peaks = []
    for parent_id in (int(v) for v in np.unique(head_parents[mask]) if int(v) > 0):
        labels, count = ndimage.label(
            mask & (head_parents == parent_id),
            structure=ndimage.generate_binary_structure(2, 2),
        )
        for component in range(1, int(count) + 1):
            coords = np.argwhere(labels == component)
            scores = probability[coords[:, 0], coords[:, 1]]
            best = float(scores.max())
            tied = coords[scores == best]
            tied = tied[np.lexsort((tied[:, 1], tied[:, 0]))]
            y, x = tied[0]
            peaks.append((int(y), int(x), best, parent_id))
    return peaks


def _distance_and_cap(
    peaks: list[tuple[int, int, float, int]],
    *,
    min_distance: float,
    cap: int,
) -> tuple[list[tuple[int, int, float, int]], list[tuple[int, int, float, int]]]:
    after_distance = []
    after_cap = []
    for parent_id in sorted({item[3] for item in peaks}):
        ordered = sorted(
            (item for item in peaks if item[3] == parent_id),
            key=lambda item: (-item[2], item[0], item[1]),
        )
        selected = []
        for item in ordered:
            if all(
                (item[0] - old[0]) ** 2 + (item[1] - old[1]) ** 2
                >= float(min_distance) ** 2
                for old in selected
            ):
                selected.append(item)
        after_distance.extend(selected)
        after_cap.extend(selected[: int(cap)])
    return after_distance, after_cap


def _gt_relations(
    gt: np.ndarray,
    parents: np.ndarray,
    head_parents: np.ndarray,
    head_size: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], int]:
    relations = []
    lost = 0
    src_h, src_w = gt.shape
    for parent_id in range(1, int(parents.max()) + 1):
        parent = parents == parent_id
        child_ids = (int(v) for v in np.unique(gt[parent]) if int(v) > 0)
        for gt_id in child_ids:
            intersection = (gt == gt_id) & parent
            y, x = _edt_center(intersection)
            if not np.any(head_parents == parent_id):
                lost += 1
                relations.append((parent_id, -1, -1))
                continue
            hy = _map(y, src_h, head_size[0])
            hx = _map(x, src_w, head_size[1])
            valid = np.argwhere(head_parents == parent_id)
            distances = np.square(valid[:, 0] - hy) + np.square(valid[:, 1] - hx)
            nearest = valid[int(np.argmin(distances))]
            relations.append((parent_id, int(nearest[0]), int(nearest[1])))
    return relations, lost


def _match(
    gt: list[tuple[int, int, int]],
    peaks: list[tuple[int, int, float, int]],
    radius: float,
) -> dict[str, int | float]:
    matched = 0
    near_predictions = 0
    valid_gt = [item for item in gt if item[1] >= 0]
    for parent_id in sorted({item[0] for item in valid_gt} | {item[3] for item in peaks}):
        left = [item for item in valid_gt if item[0] == parent_id]
        right = [item for item in peaks if item[3] == parent_id]
        if not left or not right:
            continue
        a = np.asarray([(item[1], item[2]) for item in left], dtype=np.float64)
        b = np.asarray([(item[0], item[1]) for item in right], dtype=np.float64)
        distance = np.sqrt(np.square(a[:, None] - b[None]).sum(axis=2))
        rows, columns = linear_sum_assignment(distance)
        matched += sum(distance[row, column] <= radius for row, column in zip(rows, columns))
        near_predictions += int((distance.min(axis=0) <= radius).sum())
    total_gt = len(gt)
    total_pred = len(peaks)
    return {
        "gt": total_gt,
        "pred": total_pred,
        "matched": int(matched),
        "recall": matched / max(total_gt, 1),
        "precision": matched / max(total_pred, 1),
        "near_gt_predictions": near_predictions,
        "duplicate_or_unassigned_near_gt": max(near_predictions - matched, 0),
    }


def _has_near(
    relation: tuple[int, int, int],
    peaks: list[tuple[int, int, float, int]],
    radius: float,
) -> bool:
    parent, y, x = relation
    if y < 0:
        return False
    return any(
        item[3] == parent
        and (item[0] - y) ** 2 + (item[1] - x) ** 2 <= radius**2
        for item in peaks
    )


def main() -> int:
    parser = argparse.ArgumentParser("Audit Center loss stages with spatial matching")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gt-sidecar", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radii", default="1,2,3,5")
    parser.add_argument("--taxonomy-radius", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    bootstrap(args.experiment_root)
    from rgid.evaluation import load_head_checkpoint

    device = torch.device(args.device)
    head, config, _ = load_head_checkpoint(args.checkpoint, device=device)
    dataset = VariantCacheDataset(args.cache_dir, args.gt_sidecar)
    if args.limit is not None:
        dataset.files = dataset.files[: max(int(args.limit), 0)]
    radii = tuple(float(v) for v in args.radii.split(",") if v.strip())
    aggregate = {
        stage: {radius: Counter() for radius in radii}
        for stage in ("local_max", "threshold", "distance_nms", "marker_cap")
    }
    taxonomy = Counter()
    sample_rows = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            feature = sample["fused_roi"][None].to(device)
            logits = head(feature)["center_logits"][0, 0].float().cpu()
            probability = torch.sigmoid(logits).numpy()
            gt = sample["instance_ids"].numpy()
            parents = binary_cc(sample["parent_mask"].numpy(), min_area=1)
            hp = _head_parents(parents, probability.shape)
            relations, lost = _gt_relations(gt, parents, hp, probability.shape)
            local = _plateau_peaks(
                probability, hp, kernel=config.nms_kernel, threshold=None
            )
            thresholded = _plateau_peaks(
                probability,
                hp,
                kernel=config.nms_kernel,
                threshold=config.peak_threshold,
            )
            distance, capped = _distance_and_cap(
                thresholded,
                min_distance=config.min_peak_distance,
                cap=config.max_markers_per_region,
            )
            stages = {
                "local_max": local,
                "threshold": thresholded,
                "distance_nms": distance,
                "marker_cap": capped,
            }
            for stage, peaks in stages.items():
                for radius in radii:
                    result = _match(relations, peaks, radius)
                    aggregate[stage][radius].update(
                        {key: int(result[key]) for key in ("gt", "pred", "matched", "near_gt_predictions", "duplicate_or_unassigned_near_gt")}
                    )

            for relation in relations:
                if relation[1] < 0:
                    taxonomy["parent_lost_at_head_resolution"] += 1
                elif not _has_near(relation, local, args.taxonomy_radius):
                    taxonomy["no_local_max_within_radius"] += 1
                elif not _has_near(relation, thresholded, args.taxonomy_radius):
                    taxonomy["filtered_by_threshold"] += 1
                elif not _has_near(relation, distance, args.taxonomy_radius):
                    taxonomy["suppressed_by_distance_nms"] += 1
                elif not _has_near(relation, capped, args.taxonomy_radius):
                    taxonomy["truncated_by_marker_cap"] += 1
                else:
                    taxonomy["retained_within_radius"] += 1
            sample_rows.append(
                {
                    "name": sample["name"],
                    "gt_relations": len(relations),
                    "parent_lost": lost,
                    "local_peaks": len(local),
                    "threshold_peaks": len(thresholded),
                    "distance_nms_peaks": len(distance),
                    "marker_cap_peaks": len(capped),
                }
            )
            print(f"[{index + 1}/{len(dataset)}] {sample['name']}", flush=True)

    metrics = {}
    for stage, by_radius in aggregate.items():
        metrics[stage] = {}
        for radius, counts in by_radius.items():
            matched = int(counts["matched"])
            metrics[stage][str(radius)] = {
                **dict(counts),
                "recall": matched / max(int(counts["gt"]), 1),
                "precision": matched / max(int(counts["pred"]), 1),
            }
    report = {
        "protocol": "rgid_center_stage_spatial_audit_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "gt_sidecar": str(Path(args.gt_sidecar).resolve()) if args.gt_sidecar else None,
        "gt_variant": "damage_class_cc" if args.gt_sidecar else "binary_cc",
        "radii_head_pixels": list(radii),
        "taxonomy_radius_head_pixels": args.taxonomy_radius,
        "config": {
            "peak_threshold": config.peak_threshold,
            "nms_kernel": config.nms_kernel,
            "min_peak_distance": config.min_peak_distance,
            "max_markers_per_region": config.max_markers_per_region,
        },
        "loss_taxonomy": dict(taxonomy),
        "stage_metrics": metrics,
        "samples": sample_rows,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    atomic_json(report, output_dir / "center_stage_audit.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "center_stage_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    print(json.dumps(report["loss_taxonomy"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
