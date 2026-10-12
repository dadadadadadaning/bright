#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train S8-P5 DCMF native-SAR fusion variants on one GPU."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.make_data_loader_p4_native_sar import build_p4_dataset  # noqa: E402
from model.DamageFormer_S8_MICB_DSMIL import BuildingBagSpec  # noqa: E402
from model.DamageFormer_S8_P4_NativeSAR_MICB import (  # noqa: E402
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from model.DamageFormer_S8_P5_DCMF import (  # noqa: E402
    DamageFormer_S8_P5_DCMF,
    P5_VARIANTS,
    configure_p5_trainable,
    decur_redundancy_losses,
    get_p5_variant_spec,
    load_p3_checkpoint_for_p5,
    lock_p5_frozen_modules,
)
from model.DamageFormer_S8_P5_C2_Recovery import (  # noqa: E402
    C2RecoveryHeads,
    c2_recovery_losses,
    get_c2_recovery_spec,
)
from train_S5_S8_GPU import (  # noqa: E402
    fast_hist,
    metrics_from_hist,
    segmentation_loss,
    set_random_seed,
    setup_runtime,
    seed_worker,
)
from train_S8_P4_native_sar_micb import (  # noqa: E402
    _atomic_save as _p4_atomic_save,
    augment_metrics,
    evaluate,
    file_sha256,
    format_metrics,
    p3_bag_losses,
    parse_p4_batch,
    select_bags,
    spatial_kl,
)


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    """P5-local atomic save that removes an incomplete temporary archive."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        _p4_atomic_save(payload, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _update_latest_checkpoint(source: Path, destination: Path) -> None:
    """Point latest_resume at an existing periodic checkpoint without duplication."""

    temporary = destination.with_name(destination.name + ".link.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            # Some network filesystems do not support hard links.
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_optimizer(
    model: DamageFormer_S8_P5_DCMF,
    sar_learning_rate: float,
    fusion_learning_rate: float,
    core_learning_rate: float,
    weight_decay: float,
    recovery_heads: C2RecoveryHeads | None = None,
    recovery_learning_rate: float = 1e-4,
) -> torch.optim.Optimizer:
    groups: Dict[str, list] = {"sar": [], "fusion": [], "core": []}
    for name, parameter in model.named_parameters():
        if name.startswith(("encoder_2.", "fusion_layer_2.")):
            groups["sar"].append(parameter)
        elif name.startswith("fusion_layer_3."):
            groups["fusion"].append(parameter)
        elif name.startswith(("clf_2.", "micb_dsmil.")):
            groups["core"].append(parameter)
    if any(not values for values in groups.values()):
        raise RuntimeError(
            "invalid P5 optimizer groups: "
            + ", ".join(f"{key}={len(value)}" for key, value in groups.items())
        )
    parameter_groups = [
            {
                "params": groups["sar"],
                "lr": float(sar_learning_rate),
                "name": "native_sar_encoder",
            },
            {
                "params": groups["fusion"],
                "lr": float(fusion_learning_rate),
                "name": "dcmf",
            },
            {
                "params": groups["core"],
                "lr": float(core_learning_rate),
                "name": "p3_damage_instance_heads",
            },
        ]
    if recovery_heads is not None:
        parameter_groups.append(
            {
                "params": list(recovery_heads.parameters()),
                "lr": float(recovery_learning_rate),
                "name": "c2_recovery_heads",
            }
        )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(weight_decay),
    )


def configure_recovery_trainable(
    model: DamageFormer_S8_P5_DCMF,
    heads: C2RecoveryHeads,
    warmup: bool,
) -> Dict[str, int]:
    """Warm up only recovery heads, then unlock the declared C2 modules."""

    configure_p5_trainable(model, warmup=False)
    if warmup:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for parameter in heads.parameters():
        parameter.requires_grad_(True)
    return {
        "total": int(
            sum(parameter.numel() for parameter in model.parameters())
            + sum(parameter.numel() for parameter in heads.parameters())
        ),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in tuple(model.parameters()) + tuple(heads.parameters())
                if parameter.requires_grad
            )
        ),
    }


def load_c2_weights(model: DamageFormer_S8_P5_DCMF, path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("C2 checkpoint must be a mapping")
    saved_variant = str(checkpoint.get("variant", "c2")).lower()
    if saved_variant != "c2":
        raise RuntimeError(f"expected a C2 checkpoint, got variant={saved_variant}")
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("C2 checkpoint has no model state")
    cleaned = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return {
        "checkpoint": os.path.abspath(path),
        "source_step": int(checkpoint.get("step", -1)),
        "source_best_metrics": dict(checkpoint.get("best_metrics", {})),
        "strict_load": True,
    }


def p5_distillation_losses(
    student_outputs,
    teacher_outputs,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    student_loc, student_damage, student_aux = student_outputs
    teacher_loc, teacher_damage, teacher_aux = teacher_outputs
    return {
        "loss_kd_damage": spatial_kl(
            student_damage, teacher_damage.detach(), temperature
        ),
        "loss_kd_location": spatial_kl(
            student_loc, teacher_loc.detach(), temperature
        ),
        "loss_kd_fused": F.smooth_l1_loss(
            student_aux["fused_roi"].float(),
            teacher_aux["fused_roi"].detach().float(),
        ),
        "loss_kd_sar": F.smooth_l1_loss(
            student_aux["sar_feature"].float(),
            teacher_aux["sar_feature"].detach().float(),
        ),
    }


@torch.no_grad()
def evaluate_recovery(
    student: DamageFormer_S8_P5_DCMF,
    teacher: DamageFormer_S8_P4_NativeSAR_MICB,
    recovery_heads: C2RecoveryHeads,
    recovery_variant: str,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Full validation for a P3-anchored C2 recovery model."""

    spec = get_c2_recovery_spec(recovery_variant)
    student.eval()
    teacher.eval()
    recovery_heads.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_damage_hist: Dict[str, torch.Tensor] = {}
    processed = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        pre, sar, teacher_sar, location, damage, names = parse_p4_batch(batch, device)
        c2_location, c2_damage = student(pre, sar)
        p3_location, p3_damage = teacher(pre, teacher_sar)
        output_loc, output_damage = recovery_heads.anchor_outputs(
            c2_location,
            c2_damage,
            p3_location,
            p3_damage,
            enabled=spec.p3_anchor,
        )
        loc_prediction = output_loc.argmax(1)
        damage_prediction = output_damage.argmax(1)
        loc_hist += fast_hist(loc_prediction, location, 2, ignore_index)
        damage_hist += fast_hist(damage_prediction, damage, 4, ignore_index)
        for sample_index in range(int(pre.shape[0])):
            sample_name = str(names[sample_index])
            event_name = re.sub(r"_\d+$", "", sample_name)
            sample_hist = fast_hist(
                damage_prediction[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1],
                4,
                ignore_index,
            )
            if event_name not in event_damage_hist:
                event_damage_hist[event_name] = torch.zeros_like(sample_hist)
            event_damage_hist[event_name] += sample_hist
        processed += int(pre.shape[0])
    metrics = augment_metrics(metrics_from_hist(loc_hist, damage_hist), loc_hist)
    intact_total = damage_hist[1].sum().clamp_min(1.0)
    metrics["intact_to_damaged"] = float(
        (damage_hist[1, 2] / intact_total).item()
    )
    metrics["intact_to_destroyed"] = float(
        (damage_hist[1, 3] / intact_total).item()
    )
    metrics["processed_images"] = processed
    event_mious = []
    for hist in event_damage_hist.values():
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - intersection
        event_mious.append(float((intersection / union.clamp_min(1.0)).mean().item()))
    metrics["event_count"] = len(event_mious)
    metrics["event_mean_official_style_4class_miou"] = (
        sum(event_mious) / len(event_mious) if event_mious else 0.0
    )
    return metrics


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_metrics: Mapping[str, float],
    args,
    early_stopped: bool = False,
    recovery_heads: C2RecoveryHeads | None = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "experiment": "S8_P5_DCMF",
        "variant": str(args.variant),
        "best_metrics": dict(best_metrics),
        "args": vars(args),
        "early_stopped": bool(early_stopped),
        "recovery_variant": str(args.recovery_variant),
    }
    if recovery_heads is not None:
        payload["recovery_heads"] = recovery_heads.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    return payload


def load_resume(
    model: nn.Module,
    path: str,
    recovery_heads: C2RecoveryHeads | None = None,
    expected_recovery_variant: str = "none",
):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resume checkpoint must be a mapping")
    saved_variant = str(checkpoint.get("variant", "")).lower()
    if saved_variant and saved_variant != str(model.variant).lower():
        raise RuntimeError(
            f"resume variant mismatch: checkpoint={saved_variant}, "
            f"model={model.variant}"
        )
    saved_recovery = str(checkpoint.get("recovery_variant", "none")).lower()
    if saved_recovery != str(expected_recovery_variant).lower():
        raise RuntimeError(
            f"resume recovery variant mismatch: checkpoint={saved_recovery}, "
            f"requested={expected_recovery_variant}"
        )
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("resume checkpoint has no model state")
    cleaned = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    if recovery_heads is not None:
        recovery_state = checkpoint.get("recovery_heads")
        if not isinstance(recovery_state, Mapping):
            raise KeyError("recovery resume checkpoint has no recovery_heads state")
        recovery_heads.load_state_dict(recovery_state, strict=True)
    return checkpoint, int(checkpoint.get("step", 0)), dict(
        checkpoint.get("best_metrics", {})
    )


def main() -> None:
    parser = argparse.ArgumentParser("Train S8-P5 DCMF")
    parser.add_argument("--variant", choices=tuple(P5_VARIANTS), required=True)
    parser.add_argument(
        "--recovery_variant",
        choices=("none", "r0", "r1", "r2", "r3"),
        default="none",
    )
    parser.add_argument("--c2_checkpoint", default="")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--sar_learning_rate", type=float, default=1e-4)
    parser.add_argument("--fusion_learning_rate", type=float, default=1e-4)
    parser.add_argument("--core_learning_rate", type=float, default=1e-6)
    parser.add_argument("--recovery_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--use_lovasz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument("--min_instance_area", type=int, default=32)
    parser.add_argument("--max_bags_per_class_per_batch", type=int, default=16)
    parser.add_argument("--mil_warmup_steps", type=int, default=1000)
    parser.add_argument("--lambda_mil", type=float, default=0.25)
    parser.add_argument("--lambda_exist_aux", type=float, default=0.25)
    parser.add_argument("--lambda_severity_aux", type=float, default=0.25)
    parser.add_argument("--lambda_intact_guard", type=float, default=0.05)

    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--lambda_kd_damage", type=float, default=1.0)
    parser.add_argument("--lambda_kd_location", type=float, default=0.25)
    parser.add_argument("--lambda_kd_fused", type=float, default=0.10)
    parser.add_argument("--lambda_kd_sar_warmup", type=float, default=1.0)
    parser.add_argument("--lambda_kd_sar_late", type=float, default=0.25)
    parser.add_argument("--lambda_decur_common", type=float, default=0.05)
    parser.add_argument("--lambda_decur_unique", type=float, default=0.02)
    parser.add_argument("--error_flow_margin", type=float, default=0.3)
    parser.add_argument("--lambda_error_flow", type=float, default=0.10)
    parser.add_argument("--supcon_temperature", type=float, default=0.10)
    parser.add_argument("--lambda_supcon", type=float, default=0.05)
    parser.add_argument("--recovery_loss_warmup_steps", type=int, default=500)
    parser.add_argument("--anchor_init_tolerance_pp", type=float, default=0.05)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument(
        "--validation_only_no_checkpoint",
        action="store_true",
        help="Run validation/guard without writing large checkpoint archives.",
    )
    parser.add_argument("--expected_a0_valid_miou", type=float, default=60.5019)
    parser.add_argument("--guard_start_step", type=int, default=3000)
    parser.add_argument("--early_stop_drop_pp", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    args = parser.parse_args()

    if args.core_learning_rate > 1e-6:
        raise ValueError("P5 core learning rate must remain <= 1e-6")
    recovery_enabled = args.recovery_variant != "none"
    if recovery_enabled and args.variant != "c2":
        raise ValueError("C2 recovery requires --variant c2")
    if recovery_enabled and not os.path.isfile(args.c2_checkpoint):
        raise FileNotFoundError(
            "C2 recovery requires --c2_checkpoint, got "
            f"{args.c2_checkpoint!r}"
        )
    if args.guard_start_step <= args.warmup_steps:
        raise ValueError("guard_start_step must be later than warmup_steps")
    if not os.path.isfile(args.sar_stats_json):
        raise FileNotFoundError(args.sar_stats_json)

    device, distributed, rank, world_size, _ = setup_runtime(args.device)
    if distributed or rank != 0 or world_size != 1:
        raise RuntimeError("S8-P5 controlled experiments intentionally use one GPU")
    generator = set_random_seed(args.seed, 0, args.deterministic)
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and args.amp_dtype == "fp16"
    )

    model_kwargs = {
        "roi_alpha": 0.1,
        "detach_building_prior": True,
        "token_dim": 128,
        "mil_topk": 8,
        "max_tokens_per_bag": 256,
        "mil_residual_scale": 0.0,
        "use_asymmetric_regions": False,
    }
    student = DamageFormer_S8_P5_DCMF(
        variant=args.variant, **model_kwargs
    ).to(device)
    recovery_heads = C2RecoveryHeads().to(device) if recovery_enabled else None
    recovery_spec = (
        get_c2_recovery_spec(args.recovery_variant) if recovery_enabled else None
    )
    teacher = DamageFormer_S8_P4_NativeSAR_MICB(
        variant="a0", **model_kwargs
    ).to(device)
    teacher_report = load_p3_checkpoint(teacher, args.base_ckpt)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint, start_step, best_metrics = load_resume(
            student,
            args.resume,
            recovery_heads=recovery_heads,
            expected_recovery_variant=args.recovery_variant,
        )
        migration_report = {"resume": args.resume, "teacher": teacher_report}
    elif recovery_enabled:
        migration_report = {
            "student": load_c2_weights(student, args.c2_checkpoint),
            "teacher": teacher_report,
            "recovery_variant": args.recovery_variant,
            "recovery_spec": recovery_spec.__dict__,
        }
        start_step, best_metrics = 0, {}
    else:
        migration_report = {
            "student": load_p3_checkpoint_for_p5(student, args.base_ckpt),
            "teacher": teacher_report,
        }
        start_step, best_metrics = 0, {}

    warmup = start_step < args.warmup_steps
    parameter_counts = (
        configure_recovery_trainable(student, recovery_heads, warmup=warmup)
        if recovery_enabled and recovery_heads is not None
        else configure_p5_trainable(student, warmup=warmup)
    )
    optimizer = build_optimizer(
        student,
        args.sar_learning_rate,
        args.fusion_learning_rate,
        args.core_learning_rate,
        args.weight_decay,
        recovery_heads=recovery_heads,
        recovery_learning_rate=args.recovery_learning_rate,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if scaler.is_enabled() and "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])

    # P5 always consumes the native one-channel train-statistics view.  The
    # same loader also supplies the frozen A0 teacher view.
    train_set = build_p4_dataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        "train",
        "a2",
        args.sar_stats_json,
    )
    val_set = build_p4_dataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        "val",
        "a2",
        args.sar_stats_json,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    if len(train_loader) == 0:
        raise RuntimeError("training loader is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (
        ("config.json", vars(args)),
        ("migration_report.json", migration_report),
        (
            "runtime_metadata.json",
            {
                "created_at": datetime.now().isoformat(),
                "device": str(device),
                "cuda_device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "parameter_counts": parameter_counts,
                "student_metadata": student.experiment_metadata(),
                "teacher_frozen": True,
                "recovery_variant": args.recovery_variant,
                "recovery_spec": (
                    None if recovery_spec is None else recovery_spec.__dict__
                ),
            },
        ),
        (
            "checkpoint_identity.json",
            {
                "base_checkpoint": os.path.abspath(args.base_ckpt),
                "base_checkpoint_sha256": file_sha256(args.base_ckpt),
                "expected_experiment": "S8_MICB_DSMIL_P3",
            },
        ),
    ):
        with open(output_dir / filename, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)

    spec = get_p5_variant_spec(args.variant)
    print("=" * 80, flush=True)
    print(f"Experiment       : {spec.name}", flush=True)
    print(f"C2 recovery      : {args.recovery_variant}", flush=True)
    if recovery_enabled:
        print(f"Recovery method  : {recovery_spec.name}", flush=True)
        print(f"C2 checkpoint    : {args.c2_checkpoint}", flush=True)
    print(f"Base/teacher     : {args.base_ckpt}", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print("SAR input        : native 1-channel, train-only statistics", flush=True)
    print(
        "MSFA/Cross/DeCUR : "
        f"{spec.filter_augmentation}/"
        f"{spec.cross_attention}/"
        f"{spec.common_unique_decoupling}",
        flush=True,
    )
    print("RGB/Loc/BN       : FROZEN / FROZEN / FROZEN", flush=True)
    print(
        f"LR sar/fusion/core: {args.sar_learning_rate}/"
        f"{args.fusion_learning_rate}/{args.core_learning_rate}",
        flush=True,
    )
    print(
        f"Steps/EBS        : {args.optimizer_steps}/"
        f"{args.train_batch_size * args.accumulation_steps}",
        flush=True,
    )
    print(
        f"Initial trainable: {parameter_counts['trainable']:,}/"
        f"{parameter_counts['total']:,}",
        flush=True,
    )
    print("=" * 80, flush=True)

    location_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    damage_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    best_valid_miou = float(best_metrics.get("valid_miou", -1.0))
    best_step = int(start_step)
    bad_validation_count = 0

    if start_step == 0 and not args.skip_initial_eval:
        initial = (
            evaluate_recovery(
                student,
                teacher,
                recovery_heads,
                args.recovery_variant,
                val_loader,
                device,
                args.ignore_index,
                args.val_max_batches,
            )
            if recovery_enabled and recovery_heads is not None
            else evaluate(
                student, val_loader, device, args.ignore_index, args.val_max_batches
            )
        )
        initial_text = format_metrics(initial)
        if recovery_enabled:
            initial_text += (
                f" | Int->Dmg={100*initial['intact_to_damaged']:.2f}%"
                f" | Int->Des={100*initial['intact_to_destroyed']:.2f}%"
            )
        print("Initial validation | " + initial_text, flush=True)
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps({"step": 0, "phase": "initial", **initial}) + "\n")
        if (
            recovery_enabled
            and recovery_spec is not None
            and recovery_spec.p3_anchor
            and args.val_max_batches == 0
            and abs(100.0 * initial["valid_miou"] - args.expected_a0_valid_miou)
            > args.anchor_init_tolerance_pp
        ):
            raise RuntimeError(
                "P3-anchored C2 recovery does not reproduce P3 at initialization: "
                f"observed={100.0 * initial['valid_miou']:.4f}, "
                f"expected={args.expected_a0_valid_miou:.4f} +/- "
                f"{args.anchor_init_tolerance_pp:.4f}"
            )
        best_valid_miou = float(initial["valid_miou"])
        best_metrics = dict(initial)
        best_step = 0
        if not args.validation_only_no_checkpoint:
            _atomic_save(
                checkpoint_payload(
                    student,
                    optimizer,
                    scaler,
                    0,
                    best_metrics,
                    args,
                    recovery_heads=recovery_heads,
                ),
                output_dir / "best_val_miou.pth",
            )

    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    window_start = time.perf_counter()
    window_step = step
    early_stopped = False
    lock_p5_frozen_modules(student, warmup=warmup)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < args.optimizer_steps and not early_stopped:
        for batch in train_loader:
            if warmup and step >= args.warmup_steps:
                warmup = False
                parameter_counts = (
                    configure_recovery_trainable(
                        student, recovery_heads, warmup=False
                    )
                    if recovery_enabled and recovery_heads is not None
                    else configure_p5_trainable(student, warmup=False)
                )
                print(
                    f"Unlocked clf_2 + MICB-DSMIL at step {step}; "
                    f"trainable={parameter_counts['trainable']:,}",
                    flush=True,
                )
            lock_p5_frozen_modules(student, warmup=warmup)
            if recovery_heads is not None:
                recovery_heads.train()
            pre, sar, teacher_sar, location, damage, _ = parse_p4_batch(
                batch, device
            )
            bags: Sequence[BuildingBagSpec] = []
            if step + 1 > args.mil_warmup_steps:
                bags = select_bags(
                    location,
                    damage,
                    args.min_instance_area,
                    args.max_bags_per_class_per_batch,
                )

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                teacher_outputs = teacher(pre, teacher_sar, return_aux=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                raw_student_outputs = student(
                    pre, sar, bag_specs=bags, return_aux=True
                )
                c2_location, c2_damage, student_aux = raw_student_outputs
                if recovery_enabled and recovery_heads is not None:
                    p3_location, p3_damage, _ = teacher_outputs
                    output_loc, output_damage = recovery_heads.anchor_outputs(
                        c2_location,
                        c2_damage,
                        p3_location,
                        p3_damage,
                        enabled=recovery_spec.p3_anchor,
                    )
                    student_outputs = (
                        output_loc,
                        output_damage,
                        student_aux,
                    )
                else:
                    output_loc, output_damage = c2_location, c2_damage
                    student_outputs = raw_student_outputs
                loss_loc, ce_loc, lovasz_loc = segmentation_loss(
                    output_loc,
                    location,
                    location_ce,
                    args.use_lovasz,
                    args.ce_weight,
                    args.lovasz_weight,
                    args.ignore_index,
                )
                loss_damage, ce_damage, lovasz_damage = segmentation_loss(
                    output_damage,
                    damage,
                    damage_ce,
                    args.use_lovasz,
                    args.ce_weight,
                    args.lovasz_weight,
                    args.ignore_index,
                )
                bag_losses = p3_bag_losses(
                    student_aux["bag_result"], output_damage
                )
                kd_losses = p5_distillation_losses(
                    student_outputs, teacher_outputs, args.kd_temperature
                )
                decur_losses = decur_redundancy_losses(student_aux)
                recovery_losses = (
                    c2_recovery_losses(
                        recovery_heads,
                        recovery_spec,
                        output_damage,
                        damage,
                        student_aux["fused_roi"],
                        bags,
                        args.error_flow_margin,
                        args.supcon_temperature,
                        args.ignore_index,
                    )
                    if recovery_enabled and recovery_heads is not None
                    else {
                        "error_flow": {
                            "loss": output_damage.float().sum() * 0.0,
                            "loss_damaged_to_intact": output_damage.float().sum()
                            * 0.0,
                            "loss_destroyed_to_intact": output_damage.float().sum()
                            * 0.0,
                        },
                        "supcon": {
                            "loss": output_damage.float().sum() * 0.0,
                            "anchor_count": 0,
                            "embedding_count": 0,
                        },
                    }
                )
                mil_scale = 0.0 if step + 1 <= args.mil_warmup_steps else 1.0
                sar_kd_weight = (
                    args.lambda_kd_sar_warmup
                    if warmup
                    else args.lambda_kd_sar_late
                )
                loss_bag = (
                    args.lambda_mil * bag_losses["loss_mil"]
                    + args.lambda_exist_aux * bag_losses["loss_exist_aux"]
                    + args.lambda_severity_aux * bag_losses["loss_severity_aux"]
                    + args.lambda_intact_guard * bag_losses["loss_intact_guard"]
                )
                loss_kd = (
                    args.lambda_kd_damage * kd_losses["loss_kd_damage"]
                    + args.lambda_kd_location * kd_losses["loss_kd_location"]
                    + args.lambda_kd_fused * kd_losses["loss_kd_fused"]
                    + sar_kd_weight * kd_losses["loss_kd_sar"]
                )
                loss_decur = (
                    args.lambda_decur_common * decur_losses["loss_common"]
                    + args.lambda_decur_unique * decur_losses["loss_unique"]
                )
                recovery_scale = min(
                    1.0,
                    float(step + 1)
                    / float(max(args.recovery_loss_warmup_steps, 1)),
                )
                loss_recovery = recovery_scale * (
                    args.lambda_error_flow * recovery_losses["error_flow"]["loss"]
                    + args.lambda_supcon * recovery_losses["supcon"]["loss"]
                )
                # Localization is intentionally excluded: its branch is frozen.
                total_loss = (
                    loss_damage
                    + mil_scale * loss_bag
                    + loss_kd
                    + loss_decur
                    + loss_recovery
                )

            if not bool(torch.isfinite(total_loss.detach()).item()):
                raise FloatingPointError(
                    f"non-finite P5 loss at step={step}: "
                    f"damage={loss_damage.item()}, bag={loss_bag.item()}, "
                    f"kd={loss_kd.item()}, decur={loss_decur.item()}, "
                    f"recovery={loss_recovery.item()}"
                )
            scaled = total_loss / float(args.accumulation_steps)
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                continue
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            step += 1

            if step % args.log_every == 0:
                elapsed = max(time.perf_counter() - window_start, 1e-6)
                completed = max(step - window_step, 1)
                message = {
                    "step": step,
                    "phase": "sar_alignment" if warmup else "joint_damage_adaptation",
                    "total_loss": float(total_loss.detach().item()),
                    "loc_loss_monitor_only": float(loss_loc.detach().item()),
                    "cls_loss": float(loss_damage.detach().item()),
                    "cls_ce_loss": float(ce_damage.detach().item()),
                    "cls_lovasz_loss": float(lovasz_damage.detach().item()),
                    "mil_loss": float(bag_losses["loss_mil"].detach().item()),
                    "kd_damage_loss": float(kd_losses["loss_kd_damage"].detach().item()),
                    "kd_sar_loss": float(kd_losses["loss_kd_sar"].detach().item()),
                    "kd_fused_loss": float(kd_losses["loss_kd_fused"].detach().item()),
                    "decur_common_loss": float(
                        decur_losses["loss_common"].detach().item()
                    ),
                    "decur_unique_loss": float(
                        decur_losses["loss_unique"].detach().item()
                    ),
                    "error_flow_loss": float(
                        recovery_losses["error_flow"]["loss"].detach().item()
                    ),
                    "error_flow_damaged": float(
                        recovery_losses["error_flow"][
                            "loss_damaged_to_intact"
                        ].detach().item()
                    ),
                    "error_flow_destroyed": float(
                        recovery_losses["error_flow"][
                            "loss_destroyed_to_intact"
                        ].detach().item()
                    ),
                    "supcon_loss": float(
                        recovery_losses["supcon"]["loss"].detach().item()
                    ),
                    "supcon_anchor_count": int(
                        recovery_losses["supcon"]["anchor_count"]
                    ),
                    "recovery_scale": recovery_scale,
                    "recovery_gate": (
                        float(recovery_heads.gate_value.detach().item())
                        if recovery_heads is not None
                        else 0.0
                    ),
                    "common_diag": float(decur_losses["common_diag"].item()),
                    "unique_correlation": float(
                        decur_losses["unique_correlation"].item()
                    ),
                    "bag_count": int(student_aux["bag_count"]),
                    "lr_sar": optimizer.param_groups[0]["lr"],
                    "lr_fusion": optimizer.param_groups[1]["lr"],
                    "lr_core": optimizer.param_groups[2]["lr"],
                    "optimizer_steps_per_second": completed / elapsed,
                }
                if device.type == "cuda":
                    message["peak_memory_gib"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                print(json.dumps(message, ensure_ascii=False), flush=True)
                window_start, window_step = time.perf_counter(), step
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

            if step % args.eval_every == 0 or step == args.optimizer_steps:
                metrics = (
                    evaluate_recovery(
                        student,
                        teacher,
                        recovery_heads,
                        args.recovery_variant,
                        val_loader,
                        device,
                        args.ignore_index,
                        args.val_max_batches,
                    )
                    if recovery_enabled and recovery_heads is not None
                    else evaluate(
                        student,
                        val_loader,
                        device,
                        args.ignore_index,
                        args.val_max_batches,
                    )
                )
                metrics_text = format_metrics(metrics)
                if recovery_enabled:
                    metrics_text += (
                        f" | Int->Dmg={100*metrics['intact_to_damaged']:.2f}%"
                        f" | Int->Des={100*metrics['intact_to_destroyed']:.2f}%"
                    )
                print("Validation | " + metrics_text, flush=True)
                with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
                    file.write(json.dumps({"step": step, **metrics}) + "\n")
                if metrics["valid_miou"] > best_valid_miou:
                    best_valid_miou = metrics["valid_miou"]
                    best_metrics = dict(metrics)
                    best_step = step
                    _atomic_save(
                        checkpoint_payload(
                            student,
                            optimizer,
                            scaler,
                            step,
                            best_metrics,
                            args,
                            recovery_heads=recovery_heads,
                        ),
                        output_dir / "best_val_miou.pth",
                    )
                    print(
                        f"New best valid_mIoU={100*best_valid_miou:.4f}",
                        flush=True,
                    )
                below_guard = (
                    args.val_max_batches == 0
                    and step >= args.guard_start_step
                    and 100.0 * metrics["valid_miou"]
                    < args.expected_a0_valid_miou - args.early_stop_drop_pp
                )
                bad_validation_count = (
                    bad_validation_count + 1 if below_guard else 0
                )
                if (
                    args.early_stop_patience > 0
                    and bad_validation_count >= args.early_stop_patience
                ):
                    early_stopped = True
                    print(
                        "Early stop: P5 remained below the delayed baseline "
                        f"guard for {bad_validation_count} validations.",
                        flush=True,
                    )

            if step % args.save_every == 0:
                payload = checkpoint_payload(
                    student,
                    optimizer,
                    scaler,
                    step,
                    best_metrics,
                    args,
                    early_stopped,
                    recovery_heads=recovery_heads,
                )
                checkpoint_path = output_dir / f"checkpoint_step_{step}.pth"
                _atomic_save(payload, checkpoint_path)
                _update_latest_checkpoint(
                    checkpoint_path, output_dir / "latest_resume.pth"
                )
            if step >= args.optimizer_steps or early_stopped:
                break

    final_payload = checkpoint_payload(
        student,
        optimizer,
        scaler,
        step,
        best_metrics,
        args,
        early_stopped,
        recovery_heads=recovery_heads,
    )
    if not args.validation_only_no_checkpoint:
        _atomic_save(final_payload, output_dir / "final.pth")
    summary = {
        "variant": args.variant,
        "completed_step": step,
        "target_optimizer_steps": args.optimizer_steps,
        "early_stopped": early_stopped,
        "best_valid_miou": best_valid_miou,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "recovery_variant": args.recovery_variant,
        "recovery_gate": (
            float(recovery_heads.gate_value.detach().item())
            if recovery_heads is not None
            else 0.0
        ),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(
        f"Training complete: step={step}, early_stopped={early_stopped}, "
        f"best valid_mIoU={100*best_valid_miou:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
