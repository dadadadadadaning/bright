#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from rgid.cache_dataset import RGIDCacheDataset, collate_cached_samples
from rgid.calibration import BoundaryThresholdCalibrator
from rgid.evaluation import load_head_checkpoint
from rgid.targets_losses import build_rgid_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Calibrate RGID boundary gate on train cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
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
    train_cache = Path(args.train_cache).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    head, config, checkpoint = load_head_checkpoint(checkpoint_path, device=device)
    declared_train_cache = checkpoint.get("train_cache")
    if not declared_train_cache:
        raise KeyError(
            "checkpoint has no train_cache provenance; validation cache use "
            "cannot be excluded"
        )
    declared_path = Path(str(declared_train_cache)).expanduser().resolve()
    if declared_path != train_cache:
        raise RuntimeError(
            "calibration must use the exact training cache recorded by the "
            f"checkpoint: expected={declared_path}, received={train_cache}"
        )

    loader = DataLoader(
        RGIDCacheDataset(train_cache),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_cached_samples,
    )
    calibrator = BoundaryThresholdCalibrator()
    head.eval()
    with torch.inference_mode():
        for batch in loader:
            fused_roi = batch["fused_roi"].to(device).float()
            predictions = head(fused_roi)
            targets, _ = build_rgid_targets(
                batch["instance_ids"],
                predictions["boundary_logits"].shape[-2:],
                config,
                device=device,
            )
            calibrator.update(
                predictions["boundary_logits"].float(),
                targets["boundary_target"].float(),
                building_mask=batch["parent_mask"].to(device),
            )
    calibration = asdict(calibrator.compute())
    ordering_valid = bool(
        calibration["boundary_positive_median"]
        > calibration["boundary_interior_median"]
    )
    calibration["boundary_ordering_valid"] = ordering_valid
    if not ordering_valid:
        warnings.warn(
            "boundary_positive_median is not greater than "
            "boundary_interior_median; the frozen threshold is diagnostic only",
            RuntimeWarning,
        )

    output_payload = dict(checkpoint)
    output_payload["boundary_calibration"] = calibration
    output_payload["calibration_train_cache"] = os.path.abspath(train_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "output": str(output_path),
                "train_cache": str(train_cache),
                "boundary_calibration": calibration,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
