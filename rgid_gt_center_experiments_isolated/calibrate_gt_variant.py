#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from variant_common import VariantCacheDataset, bootstrap, collate_variant, atomic_torch_save


def main() -> int:
    parser = argparse.ArgumentParser("Calibrate boundary gate with variant GT")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--gt-sidecar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bootstrap(args.experiment_root)
    from rgid.calibration import BoundaryThresholdCalibrator
    from rgid.evaluation import load_head_checkpoint
    from rgid.targets_losses import build_rgid_targets

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    train_cache = Path(args.train_cache).expanduser().resolve()
    sidecar = Path(args.gt_sidecar).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    head, config, checkpoint = load_head_checkpoint(checkpoint_path, device=device)
    if Path(checkpoint.get("train_cache", "")).resolve() != train_cache:
        raise RuntimeError("Checkpoint train_cache provenance does not match")
    if Path(checkpoint.get("gt_sidecar_dir", "")).resolve() != sidecar:
        raise RuntimeError("Checkpoint gt_sidecar_dir provenance does not match")
    loader = DataLoader(
        VariantCacheDataset(train_cache, sidecar),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variant,
    )
    calibrator = BoundaryThresholdCalibrator()
    with torch.inference_mode():
        for batch in loader:
            features = batch["fused_roi"].to(device).float()
            predictions = head(features)
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
    calibration["boundary_ordering_valid"] = bool(
        calibration["boundary_positive_median"]
        > calibration["boundary_interior_median"]
    )
    payload = dict(checkpoint)
    payload["boundary_calibration"] = calibration
    payload["calibration_train_cache"] = os.path.abspath(train_cache)
    payload["calibration_gt_sidecar_dir"] = os.path.abspath(sidecar)
    atomic_torch_save(payload, output)
    print(calibration, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
