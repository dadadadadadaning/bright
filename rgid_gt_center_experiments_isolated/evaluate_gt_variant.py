#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from variant_common import VariantCacheDataset, atomic_json, bootstrap, collate_variant


def _add_stats(target: Counter, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, (int, float)) and key != "markers_per_split_region":
            target[key] += value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                target[f"{key}.{subkey}"] += subvalue


def main() -> int:
    parser = argparse.ArgumentParser("Evaluate G0-G3 against a GT sidecar")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--gt-sidecar", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bootstrap(args.experiment_root)
    from rgid.evaluation import load_head_checkpoint
    from rgid.geometry import RGIDDecoder
    from rgid.metrics import aggregate_metric_dicts, evaluate_instance_labels
    from rgid.targets_losses import binary_mask_to_instances

    device = torch.device(args.device)
    head, config, checkpoint = load_head_checkpoint(args.checkpoint, device=device)
    calibration = checkpoint.get("boundary_calibration")
    if not isinstance(calibration, dict):
        raise KeyError("Checkpoint is not calibrated")
    threshold = float(calibration["boundary_gate_threshold"])
    loader = DataLoader(
        VariantCacheDataset(args.val_cache, args.gt_sidecar),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variant,
    )
    decoder = RGIDDecoder(config)
    rows = {mode: [] for mode in ("g0", "g1", "g2", "g3")}
    totals = {mode: Counter() for mode in ("g1", "g2", "g3")}
    with torch.inference_mode():
        for batch in loader:
            features = batch["fused_roi"].to(device).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                predictions = head(features)
            centers = predictions["center_logits"].float().cpu()
            boundaries = predictions["boundary_logits"].float().cpu()
            for index in range(int(features.shape[0])):
                gt = batch["instance_ids"][index].numpy()
                parent = batch["parent_mask"][index].numpy().astype(bool)
                g0 = binary_mask_to_instances(parent, connectivity=1)
                rows["g0"].append(evaluate_instance_labels(gt, g0))
                for mode in ("g1", "g2", "g3"):
                    result = decoder.decode(
                        centers[index, 0],
                        boundaries[index, 0],
                        parent,
                        mode,
                        inputs_are_logits=True,
                        boundary_gate_threshold=threshold,
                    )
                    rows[mode].append(
                        evaluate_instance_labels(
                            gt,
                            result.instance_labels,
                            match_iou_threshold=config.match_iou_threshold,
                            merge_split_overlap_threshold=config.merge_split_overlap_threshold,
                        )
                    )
                    _add_stats(totals[mode], result.stats.to_dict())
    methods = {}
    for mode in ("g0", "g1", "g2", "g3"):
        methods[mode] = dict(aggregate_metric_dicts(rows[mode]))
        if mode != "g0":
            methods[mode]["decode_totals"] = dict(totals[mode])
    output = {
        "protocol": "rgid_gt_variant_g0_g3_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "val_cache": str(Path(args.val_cache).resolve()),
        "gt_sidecar": str(Path(args.gt_sidecar).resolve()) if args.gt_sidecar else None,
        "gt_variant": "damage_class_cc" if args.gt_sidecar else "binary_cc",
        "boundary_gate_threshold": threshold,
        "methods": methods,
    }
    atomic_json(output, args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
