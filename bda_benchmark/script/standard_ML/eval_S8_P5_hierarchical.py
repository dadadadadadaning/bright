#!/usr/bin/env python3
"""Frozen validation/test evaluation for S8-P5-H checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.make_data_loader_p4_native_sar import build_p4_dataset  # noqa: E402
from model.DamageFormer_S8_P5_Hierarchical import (  # noqa: E402
    DamageFormer_S8_P5_Hierarchical,
)
from train_S8_P4_native_sar_micb import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate frozen S8-P5-H")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("expected a P5 training checkpoint with model state")
    saved_args = checkpoint.get("args", {})
    variant = str(checkpoint.get("variant", saved_args.get("variant", "c2")))
    model = DamageFormer_S8_P5_Hierarchical(
        variant=variant,
        roi_alpha=0.1,
        detach_building_prior=True,
        token_dim=128,
        mil_topk=8,
        max_tokens_per_bag=256,
        mil_residual_scale=0.0,
        use_asymmetric_regions=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = build_p4_dataset(
        args.dataset_path, args.data_list_path, 1024, args.split, "a2",
        args.sar_stats_json,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False,
    )
    metrics = evaluate(model, loader, device, 255, 0)
    payload = {
        "protocol": "S8_P5_H_FROZEN_EVAL_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_selector": "validation_selected" if args.split == "test" else "validation",
        "variant": variant,
        "split": args.split,
        "images": len(dataset),
        "metrics": metrics,
        "training_split_repartitioned": False,
        "test_used_for_model_selection": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
