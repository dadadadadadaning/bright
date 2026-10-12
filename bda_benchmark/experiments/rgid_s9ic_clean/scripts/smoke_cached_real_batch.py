#!/usr/bin/env python3
"""Run the complete cached pipeline on two samples from ``real_batch.pt``."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.calibrate_cached_rgid import main as calibrate_main
from scripts.compare_rgid_results import main as compare_main
from scripts.eval_cached_rgid import main as evaluate_main
from scripts.train_cached_rgid import main as train_main


def _load(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("real batch must contain a mapping")
    missing = [
        key
        for key in ("fused_roi", "instance_ids", "parent_mask")
        if key not in payload
    ]
    if missing:
        raise KeyError(f"real batch is missing {missing}")
    return payload


def _two_samples(payload: Mapping[str, object]):
    fused = torch.as_tensor(payload["fused_roi"]).float()
    instances = torch.as_tensor(payload["instance_ids"]).long()
    parents = torch.as_tensor(payload["parent_mask"]).bool()
    if fused.ndim != 4 or fused.shape[1] != 256:
        raise ValueError("real batch fused_roi must be [B,256,Hh,Wh]")
    if instances.ndim != 3 or parents.ndim != 3:
        raise ValueError("real batch instance_ids/parent_mask must be [B,H,W]")
    if fused.shape[0] != instances.shape[0] or instances.shape != parents.shape:
        raise ValueError("real batch tensor shapes are inconsistent")
    if fused.shape[0] == 0:
        raise ValueError("real batch contains no samples")
    indices = [0, 1 if fused.shape[0] > 1 else 0]
    return [
        {
            "fused_roi": fused[index].contiguous(),
            "instance_ids": instances[index].contiguous(),
            "parent_mask": parents[index].contiguous(),
            "name": f"real_batch_{sample_index}",
        }
        for sample_index, index in enumerate(indices)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Two-sample cached RGID real-batch smoke")
    parser.add_argument(
        "--real-batch",
        default="/root/shared-nvme/BRIGHT-checkpoints/S9Ic_RGID/debug/real_batch.pt",
    )
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    real_batch = Path(args.real_batch).expanduser().resolve()
    if not real_batch.is_file():
        raise FileNotFoundError(real_batch)
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="rgid_cached_real_batch_")).resolve()
    )
    train_cache = work_dir / "train_cache"
    val_cache = work_dir / "val_cache"
    output_dir = work_dir / "training"
    train_cache.mkdir(parents=True, exist_ok=False)
    val_cache.mkdir(parents=True, exist_ok=False)
    samples = _two_samples(_load(real_batch))
    for index, sample in enumerate(samples):
        torch.save(sample, train_cache / f"{index:06d}.pt")
        torch.save(sample, val_cache / f"{index:06d}.pt")

    train_main(
        [
            "--train-cache",
            str(train_cache),
            "--output-dir",
            str(output_dir),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--accumulation",
            "1",
            "--device",
            args.device,
            "--checkpoint-steps",
            "2",
            "--num-workers",
            "0",
        ]
    )
    trained = output_dir / "rgid_step_2.pth"
    calibrated = output_dir / "rgid_step_2_calibrated.pth"
    results = output_dir / "eval_g0123.json"
    comparison = output_dir / "comparison.csv"
    calibrate_main(
        [
            "--checkpoint",
            str(trained),
            "--train-cache",
            str(train_cache),
            "--output",
            str(calibrated),
            "--batch-size",
            "2",
            "--device",
            args.device,
        ]
    )
    evaluate_main(
        [
            "--checkpoint",
            str(calibrated),
            "--val-cache",
            str(val_cache),
            "--output",
            str(results),
            "--batch-size",
            "2",
            "--device",
            args.device,
        ]
    )
    compare_main(["--results", str(results), "--output-csv", str(comparison)])
    summary = {
        "real_batch": str(real_batch),
        "work_dir": str(work_dir),
        "trained_checkpoint": str(trained),
        "calibrated_checkpoint": str(calibrated),
        "evaluation": str(results),
        "comparison_csv": str(comparison),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Cached real-batch integration smoke passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
