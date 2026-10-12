#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate S5/S6/S7/S8 hierarchical RGB-SAR DamageFormer checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_Hierarchical_RGBSAR import (  # noqa: E402
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


def infer_experiment(ckpt, requested: str) -> str:
    requested = requested.upper()
    if requested != "AUTO":
        return requested
    if not isinstance(ckpt, dict):
        raise ValueError("Raw state_dict checkpoint has no experiment metadata; pass --experiment S5/S6/S7/S8")
    for key in ("experiment", "variant"):
        if key in ckpt:
            return str(ckpt[key]).upper()
    args = ckpt.get("args", {})
    if isinstance(args, dict) and "experiment" in args:
        return str(args["experiment"]).upper()
    raise ValueError("Checkpoint has no experiment metadata; pass --experiment S5/S6/S7/S8")


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate hierarchical RGB-SAR DamageFormer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--experiment",
        default="auto",
        choices=["auto", "S5", "S6", "S7", "S8", "s5", "s6", "s7", "s8"],
    )
    parser.add_argument("--dataset_path", "--val_dataset_path", dest="dataset_path", required=True)
    parser.add_argument("--data_list_path", "--val_data_list_path", dest="data_list_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--image_size", "--crop_size", dest="image_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full split")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--allow_non_strict", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    experiment = infer_experiment(ckpt, args.experiment)
    if experiment not in ("S5", "S6", "S7", "S8"):
        raise ValueError(f"Unsupported experiment {experiment}")

    ckpt_args: Dict = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    ModelClass = import_model(experiment)
    model = ModelClass(
        variant=experiment,
        diff_channels=int(ckpt_args.get("diff_channels", 64)),
        gamma_init=float(ckpt_args.get("gamma_init", 0.0)),
        roi_alpha=float(ckpt_args.get("roi_alpha", 0.1)),
        align_dim=int(ckpt_args.get("align_dim", 128)),
        detach_building_prior=not bool(ckpt_args.get("no_detach_building_prior", False)),
        use_pretrained_swin=False,
        return_dict=True,
    ).to(device)

    state = clean_state_dict(extract_state_dict(ckpt))
    if args.allow_non_strict:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"NON-STRICT CHECKPOINT LOAD: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("First missing keys:", missing[:20])
        if unexpected:
            print("First unexpected keys:", unexpected[:20])
    else:
        model.load_state_dict(state, strict=True)
        print("STRICT CHECKPOINT LOAD: PASS")

    dataset = build_dataset(args.dataset_path, args.data_list_path, args.image_size, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        experiment=experiment,
        ignore_index=args.ignore_index,
        max_batches=args.max_batches,
    )

    print("=" * 100)
    print(f"Experiment : {experiment}")
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(format_metrics(metrics))
    print(
        "F1(intact/damaged/destroyed)="
        f"{100*metrics['intact_f1']:.4f}/"
        f"{100*metrics['damaged_f1']:.4f}/"
        f"{100*metrics['destroyed_f1']:.4f}"
    )
    print("GT damaged row (%):", {
        key.replace("gt_damaged_pred_", ""): round(value, 4)
        for key, value in metrics.items() if key.startswith("gt_damaged_pred_")
    })
    print("GT destroyed row (%):", {
        key.replace("gt_destroyed_pred_", ""): round(value, 4)
        for key, value in metrics.items() if key.startswith("gt_destroyed_pred_")
    })
    print(
        "HCDM:",
        f"gamma={metrics['gamma_hcdm']:.8f}",
        *(f"S{i}_abs={metrics.get(f'hcdm_stage{i}_abs_mean', float('nan')):.6f}" for i in range(1, 5)),
        f"fused_abs={metrics.get('hcdm_fused_abs_mean', float('nan')):.6f}",
    )
    if experiment in ("S7", "S8"):
        print("Class-wise adaptive scale weights:")
        for name in ("intact", "damaged", "destroyed"):
            print(" ", name, ", ".join(
                f"S{i}={metrics[f'scale_w{i}_{name}']:.6f}" for i in range(1, 5)
            ))
    if experiment == "S8":
        print(
            "IA cosine distance:",
            f"intact={metrics['ia_intact_distance']:.6f}",
            f"damaged={metrics['ia_damaged_distance']:.6f}",
            f"destroyed={metrics['ia_destroyed_distance']:.6f}",
        )
    print("=" * 100)

    output_json = args.output_json or str(Path(args.checkpoint).with_suffix(f".{args.split}.eval.json"))
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"experiment": experiment, "split": args.split, **metrics}, file, ensure_ascii=False, indent=2)
    print(f"Saved metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
