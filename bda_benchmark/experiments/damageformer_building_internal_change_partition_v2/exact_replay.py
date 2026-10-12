#!/usr/bin/env python3
"""Verify identity-safe A1 replay before any V2 training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset

from .core import load_a1_parent, seed_worker, write_json
from .model import DamageFormerBuildingInternalChangePartitionV2
from .train import evaluate, format_metrics, read_list


def main() -> None:
    p = argparse.ArgumentParser("V2 exact A1 replay")
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_list", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--variant", default="p_sc_residual")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target_region_area", type=int, default=16)
    p.add_argument("--min_region_area", type=int, default=4)
    p.add_argument("--max_regions_per_building", type=int, default=16)
    args = p.parse_args()

    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingInternalChangePartitionV2(
        args.variant,
        target_region_area=args.target_region_area,
        min_region_area=args.min_region_area,
        max_regions_per_building=args.max_regions_per_building,
    ).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    model.configure_stage("structure")
    val_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), args.crop_size, type="val"
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    metrics = evaluate(model, loader, device, 255)
    report = {
        "parent_report": parent_report,
        "metrics": metrics,
        "change_gain": float(torch.tanh(model.biscp.change_gain).detach()),
        "feedback_gain": float(torch.tanh(model.biscp.feedback_gain).detach()),
        "stage": model.stage,
        "expected_identity_reason": "Stage-0/1 never writes region/LCA evidence into A1 parent feature.",
        "test_accessed": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_json), report)
    print("Exact replay | " + format_metrics(metrics))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
