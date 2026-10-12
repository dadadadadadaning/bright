#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from rgid.cache_dataset import RGIDCacheDataset, collate_cached_samples
from rgid.evaluation import load_head_checkpoint
from rgid.geometry import RGIDDecoder
from rgid.metrics import aggregate_metric_dicts, evaluate_instance_labels
from rgid.targets_losses import binary_mask_to_instances, build_rgid_targets


EXTRA_COUNT_KEYS = (
    "center_candidates_total",
    "centers_retained_total",
    "valid_center_targets_total",
    "split_candidate_regions",
    "split_accepted_regions",
    "boundary_gate_accepted_regions",
    "boundary_gate_rejected_regions",
    "rollback_candidate_regions",
    "rollback_accepted_regions",
    "rollback_rejected_regions",
)


def _new_decode_totals() -> Dict[str, object]:
    return {
        **{key: 0 for key in EXTRA_COUNT_KEYS},
        "rollback_reason_counts": {},
    }


def _add_decode_stats(
    totals: Dict[str, object],
    stats: Mapping[str, object],
    valid_center_targets: int,
) -> None:
    for key in EXTRA_COUNT_KEYS:
        if key == "valid_center_targets_total":
            continue
        totals[key] = int(totals[key]) + int(stats.get(key, 0))
    totals["valid_center_targets_total"] = (
        int(totals["valid_center_targets_total"]) + int(valid_center_targets)
    )
    reasons = stats.get("rollback_reason_counts", {})
    if isinstance(reasons, Mapping):
        target = totals["rollback_reason_counts"]
        for reason, count in reasons.items():
            target[str(reason)] = target.get(str(reason), 0) + int(count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate cached RGID G0/G1/G2/G3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    val_cache = Path(args.val_cache).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    head, config, checkpoint = load_head_checkpoint(checkpoint_path, device=device)
    calibration = checkpoint.get("boundary_calibration")
    if not isinstance(calibration, Mapping):
        raise KeyError(
            "checkpoint has no frozen boundary_calibration; run "
            "calibrate_cached_rgid.py on the training cache first"
        )
    if "boundary_gate_threshold" not in calibration:
        raise KeyError("boundary_calibration has no boundary_gate_threshold")
    boundary_gate_threshold = float(calibration["boundary_gate_threshold"])

    loader = DataLoader(
        RGIDCacheDataset(val_cache),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_cached_samples,
    )
    decoder = RGIDDecoder(config)
    metric_rows: Dict[str, List[Mapping[str, float | int]]] = {
        mode: [] for mode in ("g0", "g1", "g2", "g3")
    }
    decode_totals = {mode: _new_decode_totals() for mode in ("g1", "g2", "g3")}

    head.eval()
    with torch.inference_mode():
        for batch in loader:
            fused_roi = batch["fused_roi"].to(device).float()
            amp_enabled = device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                predictions = head(fused_roi)
            center_logits = predictions["center_logits"].float().cpu()
            boundary_logits = predictions["boundary_logits"].float().cpu()
            instances = batch["instance_ids"].cpu()
            parents = batch["parent_mask"].cpu()
            _, target_stats = build_rgid_targets(
                instances,
                center_logits.shape[-2:],
                config,
            )

            for index in range(int(fused_roi.shape[0])):
                gt = instances[index].numpy()
                parent = parents[index].numpy().astype(bool)
                g0_labels = binary_mask_to_instances(parent, connectivity=1)
                metric_rows["g0"].append(
                    evaluate_instance_labels(
                        gt,
                        g0_labels,
                        match_iou_threshold=config.match_iou_threshold,
                        merge_split_overlap_threshold=(
                            config.merge_split_overlap_threshold
                        ),
                    )
                )
                valid_targets = int(target_stats[index].valid_instances)
                for mode in ("g1", "g2", "g3"):
                    decoded = decoder.decode(
                        center_logits[index, 0],
                        boundary_logits[index, 0],
                        parent,
                        mode,
                        inputs_are_logits=True,
                        boundary_gate_threshold=boundary_gate_threshold,
                    )
                    metric_rows[mode].append(
                        evaluate_instance_labels(
                            gt,
                            decoded.instance_labels,
                            match_iou_threshold=config.match_iou_threshold,
                            merge_split_overlap_threshold=(
                                config.merge_split_overlap_threshold
                            ),
                        )
                    )
                    _add_decode_stats(
                        decode_totals[mode],
                        decoded.stats.to_dict(),
                        valid_targets,
                    )

    methods: Dict[str, Dict[str, object]] = {}
    for mode in ("g0", "g1", "g2", "g3"):
        aggregate = dict(aggregate_metric_dicts(metric_rows[mode]))
        if mode != "g0":
            aggregate.update(decode_totals[mode])
        methods[mode] = aggregate
    output = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "val_cache": os.path.abspath(val_cache),
        "boundary_gate_threshold": boundary_gate_threshold,
        **methods,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
