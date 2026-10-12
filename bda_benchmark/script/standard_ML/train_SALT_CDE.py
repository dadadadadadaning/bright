# -*- coding: utf-8 -*-
"""Train the isolated D0/D0N/D1--D3 SALT-CDE ablation family."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_SALT_CDE import (
    SALT_VARIANTS,
    VALID_SALT_VARIANTS,
    build_salt_damageformer,
    counterfactual_transport_loss,
    make_counterfactual_sar,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import (
    best_weights_payload,
    checkpoint_payload,
    checkpoint_selection_key,
    evaluate,
    load_resume,
    metrics_line,
    segmentation_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SALT-CDE controlled ablation")
    parser.add_argument("--variant", choices=VALID_SALT_VARIANTS, required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=8000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--loc_ce_weight", type=float, default=1.0)
    parser.add_argument("--loc_lovasz_weight", type=float, default=1.0)
    parser.add_argument("--damage_ce_weight", type=float, default=1.0)
    parser.add_argument("--damage_lovasz_weight", type=float, default=1.0)
    parser.add_argument("--base_aux_weight", type=float, default=0.0)
    parser.add_argument("--sar_aux_weight", type=float, default=0.0)
    parser.add_argument("--counterfactual_weight", type=float, default=0.20)
    parser.add_argument("--counterfactual_mode", choices=("batch_shuffle", "shift_x8", "zero"), default="batch_shuffle")
    parser.add_argument("--gate_margin", type=float, default=0.10)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_pretrained_rgb", action="store_true")
    parser.add_argument("--skip_initial_eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.variant != "d0" and not args.sar_stats_path:
        raise ValueError("D0N/D1/D2/D3 require train-only SAR statistics")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_salt_damageformer(
        args.variant, pretrained_rgb=not args.no_pretrained_rgb
    ).to(device)
    optimizer = build_optimizer(model, args)
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and args.amp == "fp16"
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and args.amp == "fp16"
        )
    start_step, best_metrics = load_resume(
        args.resume, model, optimizer, scaler, require_optimizer=bool(args.resume)
    )
    best_step = int(start_step) if best_metrics else None

    dataset_variant = "h0" if args.variant == "d0" else "h1"
    effective_batch = args.train_batch_size * args.accumulation_steps
    total_crops = args.optimizer_steps * effective_batch
    train_dataset = HeterogeneousRGBSARDataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        max_iters=max(args.train_batch_size, total_crops),
        split="train",
        variant=dataset_variant,
        sar_stats_path=args.sar_stats_path,
    )
    val_dataset = HeterogeneousRGBSARDataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        split="val",
        variant=dataset_variant,
        sar_stats_path=args.sar_stats_path,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print("=" * 88, flush=True)
    print(f"Experiment      : {SALT_VARIANTS[args.variant].name}", flush=True)
    print(f"Variant         : {args.variant}", flush=True)
    print(f"Dataset view    : {dataset_variant}", flush=True)
    print(f"Steps/EBS/crops : {args.optimizer_steps}/{effective_batch}/{total_crops}", flush=True)
    print(f"Optimizer       : AdamW lr={args.learning_rate} wd={args.weight_decay}", flush=True)
    print("Test access     : FORBIDDEN", flush=True)
    print(f"Output          : {output_dir}", flush=True)
    print("=" * 88, flush=True)

    if not args.skip_initial_eval:
        print(metrics_line("Initial validation", evaluate(
            model, val_loader, device, args.ignore_index, args.val_max_batches
        )), flush=True)

    step = start_step
    micro_step = 0
    running: Dict[str, float] = {}
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    window_start = time.time()
    while step < args.optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        rgb, sar, location, damage, _names = batch
        rgb = rgb.float().to(device, non_blocking=True)
        sar = sar.float().to(device, non_blocking=True)
        location = location.long().to(device, non_blocking=True)
        damage = damage.long().to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            wrong = None
            if args.variant in ("d3", "d3v2") and args.counterfactual_weight > 0:
                wrong_sar = make_counterfactual_sar(sar, args.counterfactual_mode)
                outputs, wrong = model.forward_counterfactual(rgb, sar, wrong_sar)
            else:
                outputs = model(rgb, sar, return_dict=True)
            loc_loss, loc_ce, loc_lovasz = segmentation_loss(
                outputs["output_loc"], location, args.ignore_index,
                args.loc_ce_weight, args.loc_lovasz_weight,
            )
            damage_loss, damage_ce, damage_lovasz = segmentation_loss(
                outputs["output_dam"], damage, args.ignore_index,
                args.damage_ce_weight, args.damage_lovasz_weight,
            )
            zero = outputs["output_dam"].float().sum() * 0.0
            base_aux = zero
            if "base_damage_logits" in outputs:
                base_aux, _, _ = segmentation_loss(
                    outputs["base_damage_logits"], damage, args.ignore_index, 1.0, 1.0
                )
            sar_aux = zero
            if args.sar_aux_weight > 0 and "sar_aux_logits" in outputs:
                sar_aux = F.cross_entropy(
                    outputs["sar_aux_logits"].float(), damage, ignore_index=args.ignore_index
                )
            cf = {
                "counterfactual_loss": zero,
                "counterfactual_gate_rank": zero,
                "counterfactual_gate_bce": zero,
                "counterfactual_fallback_kl": zero,
                "transport_nonharm_loss": zero,
            }
            if wrong is not None:
                cf = counterfactual_transport_loss(
                    outputs, wrong, damage,
                    ignore_index=args.ignore_index,
                    gate_margin=args.gate_margin,
                    explicit_gate_bce=args.variant == "d3v2",
                )
            total = (
                loc_loss + damage_loss
                + args.base_aux_weight * base_aux
                + args.sar_aux_weight * sar_aux
                + args.counterfactual_weight * cf["counterfactual_loss"]
            )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss before step {step + 1}")
        scaler.scale(total / args.accumulation_steps).backward()
        micro_step += 1
        values = {
            "total_loss": total, "loc_loss": loc_loss, "damage_loss": damage_loss,
            "loc_ce": loc_ce, "loc_lovasz": loc_lovasz,
            "damage_ce": damage_ce, "damage_lovasz": damage_lovasz,
            "base_aux_loss": base_aux, "sar_aux_loss": sar_aux, **cf,
        }
        for key, value in values.items():
            running[key] = running.get(key, 0.0) + float(value.detach())
        if micro_step % args.accumulation_steps:
            continue
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            denominator = args.log_every * args.accumulation_steps
            record = {
                "step": step,
                **{key: value / denominator for key, value in running.items()},
                "optimizer_steps_per_second": args.log_every / max(time.time() - window_start, 1e-6),
            }
            print(json.dumps(record, allow_nan=False), flush=True)
            running = {}
            window_start = time.time()
        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics = evaluate(model, val_loader, device, args.ignore_index, args.val_max_batches)
            print(metrics_line("Validation", metrics), flush=True)
            if not best_metrics or checkpoint_selection_key(metrics, "valid_miou") > checkpoint_selection_key(best_metrics, "valid_miou"):
                best_metrics = metrics
                best_step = step
                atomic_save(best_weights_payload(model, step, best_metrics, args), output_dir / "best_val_miou.pth")
                print(f"New best valid_mIoU={100*metrics['valid_miou']:.4f}", flush=True)
            (output_dir / "latest_validation.json").write_text(
                json.dumps({"step": step, "metrics": metrics}, indent=2) + "\n", encoding="utf-8"
            )
        if step % args.save_every == 0 or step == args.optimizer_steps:
            atomic_save(
                checkpoint_payload(model, optimizer, scaler, step, best_metrics, args),
                output_dir / "latest_resume.pth",
            )
    summary = {
        "protocol": "SALT_CDE_TRAIN_v1",
        "variant": args.variant,
        "completed_step": step,
        "best_step": best_step,
        "best_valid_miou": best_metrics.get("valid_miou"),
        "best_metrics": best_metrics,
        "effective_batch_size": effective_batch,
        "total_training_crops": total_crops,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Training complete: step={step}, best_step={best_step}, "
        f"best valid_mIoU={100*best_metrics.get('valid_miou', 0.0):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
