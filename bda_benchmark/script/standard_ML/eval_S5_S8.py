#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone full-validation evaluator for S5_HCDM and S8_IA_MLA.

Reports:
  localization F1/IoU
  damage OA
  4-class mIoU
  valid_mIoU over intact/damaged/destroyed
  per-class IoU/F1
  damaged->intact and destroyed->intact error rates
  GT damaged/destroyed prediction-row percentages
  HCDM gamma and stage diagnostics
  S8 class-wise Stage3/4 cosine distance and scale-weight usage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for p in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the exact dataset, metric, and model-loading logic from training.
from train_S5_S8 import (
    build_dataset,
    clean_state_dict,
    evaluate_model,
    extract_state_dict,
    format_metrics,
    import_model,
)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate S5/S8 DamageFormer experiments")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", default="auto", choices=["auto", "S5", "S8", "s5", "s8"])
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--image_size", "--crop_size", dest="image_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = full validation set")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    experiment = args.experiment.upper()
    if experiment == "AUTO":
        if not isinstance(ckpt, dict) or "experiment" not in ckpt:
            raise ValueError("Checkpoint has no experiment metadata; pass --experiment S5 or S8.")
        experiment = str(ckpt["experiment"]).upper()
    if experiment not in ("S5", "S8"):
        raise ValueError(f"Unsupported experiment {experiment}")

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    ModelClass = import_model(experiment)
    model_kwargs = {
        "diff_channels": int(ckpt_args.get("diff_channels", 64)),
        "gamma_init": float(ckpt_args.get("gamma_init", 0.0)),
        "use_pretrained_swin": False,
        "return_dict": True,
    }
    if experiment == "S8":
        model_kwargs.update(
            align_dim=int(ckpt_args.get("align_dim", 128)),
            roi_alpha=float(ckpt_args.get("roi_alpha", 0.1)),
        )
    model = ModelClass(**model_kwargs).to(device)
    state = clean_state_dict(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
        if missing:
            print("First missing:", missing[:12], flush=True)
        if unexpected:
            print("First unexpected:", unexpected[:12], flush=True)

    val_set = build_dataset(
        args.val_dataset_path, args.val_data_list_path, args.image_size, "val"
    )
    loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    metrics = evaluate_model(
        model, loader, device, experiment, args.ignore_index, args.max_batches
    )
    print("=" * 100)
    print(f"Experiment: {experiment}")
    print(f"Checkpoint: {args.checkpoint}")
    print(format_metrics(metrics))
    print("Per-class F1:")
    print(
        f"  intact={100*metrics['intact_f1']:.4f} | "
        f"damaged={100*metrics['damaged_f1']:.4f} | "
        f"destroyed={100*metrics['destroyed_f1']:.4f}"
    )
    print("GT damaged prediction row (%):")
    print(
        f"  background={metrics['gt_damaged_pred_background_pct']:.4f}, "
        f"intact={metrics['gt_damaged_pred_intact_pct']:.4f}, "
        f"damaged={metrics['gt_damaged_pred_damaged_pct']:.4f}, "
        f"destroyed={metrics['gt_damaged_pred_destroyed_pct']:.4f}"
    )
    print("GT destroyed prediction row (%):")
    print(
        f"  background={metrics['gt_destroyed_pred_background_pct']:.4f}, "
        f"intact={metrics['gt_destroyed_pred_intact_pct']:.4f}, "
        f"damaged={metrics['gt_destroyed_pred_damaged_pct']:.4f}, "
        f"destroyed={metrics['gt_destroyed_pred_destroyed_pct']:.4f}"
    )
    print("HCDM diagnostics:")
    print(
        f"  gamma={metrics['gamma_hcdm']:.8f} | "
        + " | ".join(
            f"S{i}_abs={metrics.get(f'hcdm_stage{i}_abs_mean', float('nan')):.6f}"
            for i in range(1, 5)
        )
        + f" | fused_abs={metrics.get('hcdm_fused_abs_mean', float('nan')):.6f}"
    )
    if experiment == "S8":
        print("IA-MLA class-wise cosine distance (Stage3+Stage4 pooled):")
        print(
            f"  intact={metrics['ia_intact_distance']:.6f} | "
            f"damaged={metrics['ia_damaged_distance']:.6f} | "
            f"destroyed={metrics['ia_destroyed_distance']:.6f}"
        )
        print("Class-wise adaptive scale weights:")
        for name in ("intact", "damaged", "destroyed"):
            print(
                f"  {name}: "
                + ", ".join(
                    f"S{i}={metrics[f'scale_w{i}_{name}']:.6f}" for i in range(1, 5)
                )
            )
    print("=" * 100)

    output_json = args.output_json or str(Path(args.checkpoint).with_suffix(".eval.json"))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics JSON: {output_json}")


if __name__ == "__main__":
    main()
