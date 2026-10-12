#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation/test evaluation for a trained S8-P4 A0...A4 checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.make_data_loader_p4_native_sar import (  # noqa: E402
    VALID_VARIANTS as P4_DATASET_VARIANTS,
    build_p4_dataset,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import (  # noqa: E402
    DamageFormer_S8_P4_NativeSAR_MICB,
)
from train_S5_S8_GPU import seed_worker, setup_runtime  # noqa: E402
from train_S8_P4_native_sar_micb import evaluate, format_metrics  # noqa: E402


def load_trained_checkpoint(model: torch.nn.Module, path: str) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("checkpoint contains no model state")
    cleaned = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate S8-P4 NS-MICB")
    parser.add_argument("--variant", choices=P4_DATASET_VARIANTS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--sar_stats_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ignore_index", type=int, default=255)
    args = parser.parse_args()
    if args.variant != "a0" and not args.sar_stats_json:
        raise ValueError("A1/A2/A3/A4 require --sar_stats_json")

    device, distributed, rank, world_size, _ = setup_runtime(args.device)
    if distributed or rank != 0 or world_size != 1:
        raise RuntimeError("S8-P4 evaluation intentionally uses one GPU")
    model = DamageFormer_S8_P4_NativeSAR_MICB(
        variant=args.variant,
        roi_alpha=0.1,
        detach_building_prior=True,
        token_dim=128,
        mil_topk=8,
        max_tokens_per_bag=256,
        mil_residual_scale=0.0,
        use_asymmetric_regions=False,
    ).to(device)
    checkpoint = load_trained_checkpoint(model, args.checkpoint)
    dataset = build_p4_dataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        args.split,
        args.variant,
        args.sar_stats_json,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    metrics = evaluate(model, loader, device, args.ignore_index)
    payload = {
        "protocol": "S8-P4_official_pixel_protocol",
        "split": args.split,
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "strict_load": True,
        "metrics": metrics,
        "student_metadata": model.experiment_metadata(),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(format_metrics(metrics), flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
