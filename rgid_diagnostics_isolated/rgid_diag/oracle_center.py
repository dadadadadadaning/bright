from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .common import as_numpy_2d, binary_cc, edt_center, iter_cache, write_csv, write_json
from .parent_stats import select_gt_labels


def _map_coordinate(value: int, src_size: int, dst_size: int) -> int:
    mapped = round(((float(value) + 0.5) * dst_size / src_size) - 0.5)
    return int(np.clip(mapped, 0, dst_size - 1))


def _resize_nearest(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(labels.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=size, mode="nearest")[0, 0].numpy().astype(np.int32)


def build_oracle_center_map(
    gt: np.ndarray,
    parent_mask: np.ndarray,
    head_size: tuple[int, int],
    *,
    min_overlap_pixels: int = 1,
    min_gt_fraction_in_parent: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Place one marker per GT child inside each predicted parent.

    A center is computed on GT∩parent. It is snapped to a valid cell in the
    nearest-neighbour downsampled parent. Collisions are moved to the nearest
    unused cell, but the unchanged decoder may still suppress close peaks.
    """
    parents = binary_cc(parent_mask, min_area=1)
    head_parents = _resize_nearest(parents, head_size)
    output = np.zeros(head_size, dtype=np.float32)
    stats = Counter()
    src_h, src_w = gt.shape
    out_h, out_w = head_size

    for parent_id in range(1, int(parents.max()) + 1):
        parent = parents == parent_id
        used: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int, int]] = []
        for gt_id in (int(v) for v in np.unique(gt[parent]) if int(v) > 0):
            instance = gt == gt_id
            intersection = instance & parent
            overlap = int(intersection.sum())
            fraction = overlap / max(int(instance.sum()), 1)
            if overlap < int(min_overlap_pixels) or fraction < float(min_gt_fraction_in_parent):
                stats["omitted_by_overlap_rule"] += 1
                continue
            y, x = edt_center(intersection)
            candidates.append((gt_id, y, x))
        # Highest score first and a tiny deterministic score difference prevents
        # equal-valued adjacent oracle markers from becoming one plateau.
        for rank, (_, y, x) in enumerate(candidates):
            hy = _map_coordinate(y, src_h, out_h)
            hx = _map_coordinate(x, src_w, out_w)
            valid_cells = np.argwhere(head_parents == parent_id)
            if valid_cells.size == 0:
                stats["omitted_parent_lost_at_head_resolution"] += 1
                continue
            order = np.argsort(
                np.square(valid_cells[:, 0] - hy) + np.square(valid_cells[:, 1] - hx),
                kind="stable",
            )
            selected = None
            for index in order:
                cell = (int(valid_cells[index, 0]), int(valid_cells[index, 1]))
                if cell not in used:
                    selected = cell
                    break
            if selected is None:
                stats["omitted_no_free_head_cell"] += 1
                continue
            if selected != (hy, hx):
                stats["snapped_or_collision_shifted"] += 1
            used.add(selected)
            output[selected] = max(0.90, 1.0 - rank * 1e-4)
            stats["oracle_markers_injected"] += 1
    return output, dict(stats)


def _accumulate_decode(target: Counter, stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        if isinstance(value, (int, float)) and key != "markers_per_split_region":
            target[key] += value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                target[f"{key}.{sub_key}"] += sub_value


def evaluate_oracle_centers(
    *,
    cache_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    gt_source: str = "cached",
    dataset_root: str | Path | None = None,
    damage_template: str = "target/{name}_building_damage.tif",
    true_instance_root: str | Path | None = None,
    true_instance_template: str = "{name}.tif",
    min_area: int = 32,
    min_overlap_pixels: int = 1,
    min_gt_fraction_in_parent: float = 0.0,
    device: str = "cuda",
    boundary_gate_threshold: float | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    from rgid.evaluation import load_head_checkpoint
    from rgid.geometry import RGIDDecoder
    from rgid.metrics import aggregate_metric_dicts, evaluate_instance_labels
    from rgid.targets_losses import binary_mask_to_instances

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    head, config, checkpoint_payload = load_head_checkpoint(checkpoint, device=torch_device)
    calibration = checkpoint_payload.get("boundary_calibration", {})
    threshold = (
        float(boundary_gate_threshold)
        if boundary_gate_threshold is not None
        else float(calibration.get("boundary_gate_threshold", config.boundary_gate_threshold))
    )
    decoder = RGIDDecoder(config)
    metric_rows = {key: [] for key in (
        "g0", "learned_g1", "learned_g2", "learned_g3",
        "oracle_g1", "oracle_g2", "oracle_g3",
    )}
    decode_totals = {key: Counter() for key in metric_rows if key != "g0"}
    oracle_marker_totals = Counter()
    sample_rows: list[dict[str, Any]] = []
    head.eval()

    with torch.inference_mode():
        for cache_path, payload in iter_cache(cache_dir, limit=limit):
            if "fused_roi" not in payload:
                raise KeyError(f"{cache_path} has no fused_roi")
            name = str(payload["name"])
            fused = torch.as_tensor(payload["fused_roi"])[None].to(torch_device).float()
            parent = as_numpy_2d(payload["parent_mask"], "parent_mask") > 0
            gt, _ = select_gt_labels(
                payload,
                source=gt_source,
                dataset_root=dataset_root,
                damage_template=damage_template,
                true_instance_root=true_instance_root,
                true_instance_template=true_instance_template,
                min_area=min_area,
            )
            if gt.shape != parent.shape:
                raise ValueError(f"Shape mismatch for {name}: GT {gt.shape}, parent {parent.shape}")
            amp = torch_device.type == "cuda"
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16, enabled=amp):
                predictions = head(fused)
            center_logits = predictions["center_logits"][0, 0].float().cpu()
            boundary_logits = predictions["boundary_logits"][0, 0].float().cpu()
            boundary_probability = torch.sigmoid(boundary_logits)
            oracle_prob, marker_stats = build_oracle_center_map(
                gt,
                parent,
                tuple(center_logits.shape),
                min_overlap_pixels=min_overlap_pixels,
                min_gt_fraction_in_parent=min_gt_fraction_in_parent,
            )
            oracle_marker_totals.update(marker_stats)
            g0 = binary_mask_to_instances(parent, connectivity=1)
            metric_rows["g0"].append(evaluate_instance_labels(
                gt, g0, config.match_iou_threshold, config.merge_split_overlap_threshold
            ))
            row: dict[str, Any] = {"name": name, **marker_stats}
            for family, center, boundary, logits in (
                ("learned", center_logits, boundary_logits, True),
                ("oracle", oracle_prob, boundary_probability, False),
            ):
                for mode in ("g1", "g2", "g3"):
                    result = decoder.decode(
                        center,
                        boundary,
                        parent,
                        mode,
                        inputs_are_logits=logits,
                        boundary_gate_threshold=threshold,
                    )
                    key = f"{family}_{mode}"
                    metrics = evaluate_instance_labels(
                        gt,
                        result.instance_labels,
                        config.match_iou_threshold,
                        config.merge_split_overlap_threshold,
                    )
                    metric_rows[key].append(metrics)
                    _accumulate_decode(decode_totals[key], result.stats.to_dict())
                    row[f"{key}_f1"] = metrics["f1"]
                    row[f"{key}_pred_instances"] = metrics["pred_instances"]
            sample_rows.append(row)

    methods: dict[str, Any] = {}
    for key, values in metric_rows.items():
        methods[key] = dict(aggregate_metric_dicts(values))
        if key != "g0":
            methods[key]["decode_totals"] = dict(decode_totals[key])
    gains = {
        mode: float(methods[f"oracle_{mode}"]["f1"] - methods[f"learned_{mode}"]["f1"])
        for mode in ("g1", "g2", "g3")
    }
    report = {
        "protocol": "oracle_center_fixed_watershed_gate_rollback_v1",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "gt_source": gt_source,
        "boundary_gate_threshold": threshold,
        "oracle_definition": (
            "One EDT marker per qualifying GT-instance intersection inside each predicted "
            "parent; the original RGID NMS, marker cap, watershed, boundary gate and "
            "geometry rollback are unchanged. Boundary predictions remain learned."
        ),
        "oracle_marker_totals": dict(oracle_marker_totals),
        "oracle_minus_learned_f1": gains,
        "methods": methods,
        "samples": sample_rows,
    }
    output = Path(output_dir)
    write_json(output / "oracle_center_evaluation.json", report)
    write_csv(output / "oracle_center_samples.csv", sample_rows)
    return report
