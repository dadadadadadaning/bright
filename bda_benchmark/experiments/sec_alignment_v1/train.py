# -*- coding: utf-8 -*-
"""Train SEC variants without repeating the completed clean D1 experiment."""
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
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.experiments.sec_alignment_v1.model import build_sec_model
from bda_benchmark.experiments.sec_alignment_v1.objectives import (
    SEC_VARIANTS,
    building_prediction_consistency_loss,
    linear_warmup,
    reference_validity_for_shift,
    relative_offset_equivariance_loss,
    sample_axis_shift,
    select_alignment_support,
    shift_without_wrap,
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


def configure_trainable_parameters(model: nn.Module, variant: str):
    """Freeze D1 for S2R and expose only stage-2/3 matching projections."""
    if not SEC_VARIANTS[variant].alignment_only:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return list(model.parameters()), ["all"]

    allowed = (
        "joint_stages.1.query.",
        "joint_stages.1.key.",
        "joint_stages.2.query.",
        "joint_stages.2.key.",
    )
    selected, names = [], []
    for name, parameter in model.named_parameters():
        trainable = any(token in name for token in allowed)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
            names.append(name)
    if not selected:
        raise RuntimeError(
            "S2R found no query/key parameters; inspect the SALT-D1 module names"
        )
    return selected, names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SEC shift-equivariant RGB--SAR training")
    parser.add_argument("--variant", choices=tuple(SEC_VARIANTS), required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Completed SALT-D1 best weights used as the common SEC start.",
    )
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
    parser.add_argument("--offset_delta_weight", type=float, default=0.05)
    parser.add_argument("--consistency_weight", type=float, default=0.10)
    parser.add_argument("--objective_warmup_steps", type=int, default=500)
    parser.add_argument("--shift_amounts", type=int, nargs="+", default=(4, 8))
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_sec_model(
        args.variant, pretrained_rgb=not args.no_pretrained_rgb
    ).to(device)
    if args.init_checkpoint and not args.resume:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if str(initial.get("variant", "")).lower() != "d1":
            raise RuntimeError("SEC initialization must be a completed SALT-D1 checkpoint")
        model.load_state_dict(initial["model"], strict=True)
        print(f"Initialized from frozen D1: {args.init_checkpoint}", flush=True)
    trainable_parameters, trainable_names = configure_trainable_parameters(
        model, args.variant
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
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
    # ``best_metrics`` in a resume checkpoint does not record when the best
    # weights were produced.  Recover that step from the actual best artifact
    # instead of incorrectly labelling the resume step as the best step.
    existing_best = output_dir / "best_val_miou.pth"
    if args.resume and existing_best.is_file():
        best_record = torch.load(
            existing_best, map_location="cpu", weights_only=False
        )
        best_step = int(best_record.get("step", best_step or 0))

    effective_batch = args.train_batch_size * args.accumulation_steps
    total_crops = args.optimizer_steps * effective_batch
    train_dataset = HeterogeneousRGBSARDataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        max_iters=max(args.train_batch_size, total_crops),
        split="train",
        variant="h1",
        sar_stats_path=args.sar_stats_path,
    )
    val_dataset = HeterogeneousRGBSARDataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        split="val",
        variant="h1",
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
    print(f"Experiment       : {SEC_VARIANTS[args.variant].name}", flush=True)
    architecture = (
        "SALT-D1 coarse-to-fine transport"
        if SEC_VARIANTS[args.variant].coarse_to_fine
        else "exact SALT-D1"
    )
    print(f"Architecture     : {architecture}", flush=True)
    print(f"Steps/EBS/crops  : {args.optimizer_steps}/{effective_batch}/{total_crops}", flush=True)
    print(f"Paired views     : clean + one shifted SAR view", flush=True)
    print(f"Shift amounts    : {tuple(args.shift_amounts)} pixels, one axis", flush=True)
    print(
        f"Training policy  : {'query/key alignment only' if SEC_VARIANTS[args.variant].alignment_only else 'full model'}",
        flush=True,
    )
    print(
        f"Building support : {SEC_VARIANTS[args.variant].support_mode}",
        flush=True,
    )
    print(
        f"Match schedule   : {'stage3 coarse -> stage2 residual' if SEC_VARIANTS[args.variant].coarse_to_fine else 'independent stage2/stage3'}",
        flush=True,
    )
    print(
        f"Trainable params : {sum(p.numel() for p in trainable_parameters):,} "
        f"in {len(trainable_names)} tensors",
        flush=True,
    )
    print(f"Test/instances   : FORBIDDEN / NEVER", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print("=" * 88, flush=True)
    if not args.skip_initial_eval:
        initial = evaluate(
            model, val_loader, device, args.ignore_index, args.val_max_batches
        )
        print(metrics_line("Initial validation", initial), flush=True)
        if not best_metrics:
            best_metrics = initial
            best_step = int(start_step)
            atomic_save(
                best_weights_payload(model, start_step, best_metrics, args),
                output_dir / "best_val_miou.pth",
            )
            print("Registered the common D1 initialization as step-0 best.", flush=True)

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
        draw_index = step * args.accumulation_steps + (
            micro_step % args.accumulation_steps
        )
        shift_rng = random.Random(args.seed + 104729 * draw_index)
        dx, dy = sample_axis_shift(args.shift_amounts, shift_rng)
        shifted_sar, _ = shift_without_wrap(sar, dx=dx, dy=dy)
        validity = reference_validity_for_shift(sar, dx=dx, dy=dy)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            clean, shifted = model.forward_counterfactual(rgb, sar, shifted_sar)
            building = select_alignment_support(
                clean, location, SEC_VARIANTS[args.variant].support_mode
            )
            loc_loss, loc_ce, loc_lovasz = segmentation_loss(
                clean["output_loc"], location, args.ignore_index,
                args.loc_ce_weight, args.loc_lovasz_weight,
            )
            clean_damage, clean_ce, clean_lovasz = segmentation_loss(
                clean["output_dam"], damage, args.ignore_index,
                args.damage_ce_weight, args.damage_lovasz_weight,
            )
            shifted_damage, shifted_ce, shifted_lovasz = segmentation_loss(
                shifted["output_dam"], damage, args.ignore_index,
                args.damage_ce_weight, args.damage_lovasz_weight,
            )
            damage_loss = 0.5 * (clean_damage + shifted_damage)
            zero = clean["output_dam"].float().sum() * 0.0
            offset_values = {"offset_delta_loss": zero}
            if SEC_VARIANTS[args.variant].offset_delta:
                offset_values = relative_offset_equivariance_loss(
                    clean, shifted, building, validity, dx=dx, dy=dy
                )
            consistency = zero
            if SEC_VARIANTS[args.variant].prediction_consistency:
                consistency = building_prediction_consistency_loss(
                    clean["output_dam"], shifted["output_dam"], building, validity
                )
            objective_scale = linear_warmup(
                step + 1, args.objective_warmup_steps
            )
            if SEC_VARIANTS[args.variant].alignment_only:
                # Do not let shifted-SAR segmentation labels rewrite D1's
                # damage representation.  S2R answers one question only:
                # can the existing matcher learn the known displacement?
                total = offset_values["offset_delta_loss"]
            else:
                total = (
                    loc_loss
                    + damage_loss
                    + objective_scale * args.offset_delta_weight
                    * offset_values["offset_delta_loss"]
                    + objective_scale * args.consistency_weight * consistency
                )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite SEC loss before step {step + 1}")
        scaler.scale(total / args.accumulation_steps).backward()
        micro_step += 1
        values = {
            "total_loss": total,
            "loc_loss": loc_loss,
            "loc_ce": loc_ce,
            "loc_lovasz": loc_lovasz,
            "damage_loss": damage_loss,
            "clean_damage_ce": clean_ce,
            "clean_damage_lovasz": clean_lovasz,
            "shifted_damage_ce": shifted_ce,
            "shifted_damage_lovasz": shifted_lovasz,
            "offset_delta_loss": offset_values["offset_delta_loss"],
            "prediction_consistency_loss": consistency,
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
                "last_shift": {"dx": dx, "dy": dy},
                "objective_scale": linear_warmup(step, args.objective_warmup_steps),
                "optimizer_steps_per_second": args.log_every
                / max(time.time() - window_start, 1e-6),
            }
            print(json.dumps(record, allow_nan=False), flush=True)
            running = {}
            window_start = time.time()
        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics = evaluate(
                model, val_loader, device, args.ignore_index, args.val_max_batches
            )
            print(metrics_line("Clean validation", metrics), flush=True)
            if not best_metrics or checkpoint_selection_key(
                metrics, "valid_miou"
            ) > checkpoint_selection_key(best_metrics, "valid_miou"):
                best_metrics = metrics
                best_step = step
                atomic_save(
                    best_weights_payload(model, step, best_metrics, args),
                    output_dir / "best_val_miou.pth",
                )
                print(f"New best clean valid_mIoU={100*metrics['valid_miou']:.4f}", flush=True)
            (output_dir / "latest_validation.json").write_text(
                json.dumps({"step": step, "metrics": metrics}, indent=2) + "\n",
                encoding="utf-8",
            )
        if step % args.save_every == 0 or step == args.optimizer_steps:
            atomic_save(
                checkpoint_payload(model, optimizer, scaler, step, best_metrics, args),
                output_dir / "latest_resume.pth",
            )

    summary = {
        "protocol": "SEC_TRAIN_v1",
        "variant": args.variant,
        "architecture": (
            "SALT_D1_coarse_to_fine_transport"
            if SEC_VARIANTS[args.variant].coarse_to_fine
            else "SALT_D1_exact"
        ),
        "completed_step": step,
        "best_step": best_step,
        "best_valid_miou": best_metrics.get("valid_miou"),
        "best_metrics": best_metrics,
        "effective_batch_size": effective_batch,
        "unique_training_crops": total_crops,
        "views_per_crop": 2,
        "training_policy": (
            "query_key_alignment_only"
            if SEC_VARIANTS[args.variant].alignment_only
            else "full_model"
        ),
        "building_support_mode": SEC_VARIANTS[args.variant].support_mode,
        "coarse_to_fine": SEC_VARIANTS[args.variant].coarse_to_fine,
        "trainable_parameter_count": sum(
            p.numel() for p in trainable_parameters
        ),
        "trainable_parameter_names": trainable_names,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Training complete: step={step}, best_step={best_step}, "
        f"best clean valid_mIoU={100*best_metrics.get('valid_miou', 0.0):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
