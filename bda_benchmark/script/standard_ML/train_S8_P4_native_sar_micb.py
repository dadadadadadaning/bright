#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the controlled S8-P4 A0...A4 native-SAR ablations on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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

from dataset.make_data_loader_p4_native_sar import (  # noqa: E402
    VALID_VARIANTS as P4_DATASET_VARIANTS,
    build_p4_dataset,
    load_sar_stats,
)
from model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    BuildingBagSpec,
    build_training_bag_specs,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import (  # noqa: E402
    DamageFormer_S8_P4_NativeSAR_MICB,
    P4_LATE_TRAINABLE_PREFIXES,
    SAR_INPUT_PREFIX,
    configure_p4_trainable,
    get_p4_variant_spec,
    load_p3_checkpoint,
    lock_frozen_batch_norm,
)
from train_S5_S8_GPU import (  # noqa: E402
    fast_hist,
    metrics_from_hist,
    segmentation_loss,
    set_random_seed,
    setup_runtime,
    seed_worker,
)


def parse_p4_batch(batch: Mapping[str, object], device: torch.device):
    required = (
        "pre_img",
        "post_img",
        "teacher_post_img",
        "label_loc",
        "label_dam",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"P4 batch is missing {missing}")
    pre = batch["pre_img"].to(device, non_blocking=True).float()
    student_sar = batch["post_img"].to(device, non_blocking=True).float()
    teacher_sar = batch["teacher_post_img"].to(device, non_blocking=True).float()
    location = batch["label_loc"].to(device, non_blocking=True).long()
    damage = batch["label_dam"].to(device, non_blocking=True).long()
    names = batch.get("name", [""] * int(pre.shape[0]))
    return pre, student_sar, teacher_sar, location, damage, names


def select_bags(
    location: torch.Tensor,
    damage: torch.Tensor,
    min_area: int,
    max_per_class: int,
) -> List[BuildingBagSpec]:
    bags = build_training_bag_specs(location, damage, min_area=min_area)
    if max_per_class <= 0:
        return bags
    selected: List[BuildingBagSpec] = []
    for label in (1, 2, 3):
        class_bags = [bag for bag in bags if bag.label == label]
        if len(class_bags) > max_per_class:
            class_bags = random.sample(class_bags, max_per_class)
        selected.extend(class_bags)
    return selected


def p3_bag_losses(
    bag_result: Mapping[str, object] | None,
    reference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    zero = reference.float().sum() * 0.0
    result = {
        "loss_mil": zero,
        "loss_exist_aux": zero,
        "loss_severity_aux": zero,
        "loss_intact_guard": zero,
        "loss_coverage": zero,
    }
    if not bag_result or int(bag_result.get("bag_count", 0)) == 0:
        return result

    labels = bag_result["labels"]
    log_probs3 = bag_result["log_probs3"].float()
    exist_logits = bag_result["exist_logits"].float()
    severity_logits = bag_result["severity_logits"].float()
    if not isinstance(labels, torch.Tensor):
        raise TypeError("bag labels must be a tensor")
    labels = labels.long()
    result["loss_mil"] = F.nll_loss(log_probs3, labels - 1)
    exist_target = (labels > 1).long()
    result["loss_exist_aux"] = F.cross_entropy(exist_logits, exist_target)
    severity_mask = labels > 1
    if bool(severity_mask.any().item()):
        result["loss_severity_aux"] = F.cross_entropy(
            severity_logits[severity_mask],
            labels[severity_mask] - 2,
        )
    guard = bag_result["intact_guard_logits"]
    if isinstance(guard, torch.Tensor) and guard.numel() > 0:
        result["loss_intact_guard"] = F.binary_cross_entropy_with_logits(
            guard.float(), torch.zeros_like(guard, dtype=torch.float32)
        )
    coverage = bag_result["coverage"]
    if isinstance(coverage, torch.Tensor) and coverage.numel() > 0:
        result["loss_coverage"] = F.mse_loss(
            coverage.float(), exist_target.float()
        )
    return result


def mil_supervision_scale(step: int, warmup_steps: int, ramp_steps: int) -> float:
    """Backward-compatible 0/linear/1 schedule for building supervision."""

    if step < 0 or warmup_steps < 0 or ramp_steps < 0:
        raise ValueError("MIL schedule steps must be non-negative")
    completed = int(step) + 1
    if completed <= int(warmup_steps):
        return 0.0
    if int(ramp_steps) == 0:
        return 1.0
    return min(
        1.0,
        (completed - int(warmup_steps)) / float(ramp_steps),
    )


def spatial_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"KD logits differ: {student_logits.shape} vs {teacher_logits.shape}"
        )
    t = float(temperature)
    student = F.log_softmax(student_logits.float() / t, dim=1)
    teacher = F.softmax(teacher_logits.float() / t, dim=1)
    # Sum classes, average pixels and batch. This keeps the loss independent
    # of crop area, unlike KL's batchmean reduction on dense maps.
    return F.kl_div(student, teacher, reduction="none").sum(dim=1).mean() * (t * t)


def distillation_losses(
    student_outputs: Tuple[torch.Tensor, torch.Tensor, Mapping[str, object]],
    teacher_outputs: Tuple[torch.Tensor, torch.Tensor, Mapping[str, object]],
    temperature: float,
) -> Dict[str, torch.Tensor]:
    student_loc, student_damage, student_aux = student_outputs
    teacher_loc, teacher_damage, teacher_aux = teacher_outputs
    student_feature = student_aux["fused_roi"]
    teacher_feature = teacher_aux["fused_roi"]
    if not isinstance(student_feature, torch.Tensor) or not isinstance(
        teacher_feature, torch.Tensor
    ):
        raise TypeError("fused_roi features must be tensors")
    return {
        "loss_kd_damage": spatial_kl(
            student_damage, teacher_damage.detach(), temperature
        ),
        "loss_kd_location": spatial_kl(
            student_loc, teacher_loc.detach(), temperature
        ),
        "loss_kd_feature": F.mse_loss(
            student_feature.float(), teacher_feature.detach().float()
        ),
    }


def _module_for_prefix(model: nn.Module, prefix: str) -> nn.Module:
    path = prefix.rstrip(".").split(".")
    module = model
    for name in path:
        if name.isdigit():
            module = module[int(name)]
        else:
            module = getattr(module, name)
    return module


def set_p4_training_mode(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
    input_only: bool,
) -> None:
    # Start from fully deterministic frozen inference behavior, then enable
    # training behavior only for the explicitly trainable modules.
    model.eval()
    _module_for_prefix(model, SAR_INPUT_PREFIX).train()
    if not input_only:
        for prefix in P4_LATE_TRAINABLE_PREFIXES:
            _module_for_prefix(model, prefix).train()
    lock_frozen_batch_norm(model)


def build_optimizer(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
    core_learning_rate: float,
    input_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    input_parameters = []
    core_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(SAR_INPUT_PREFIX):
            input_parameters.append(parameter)
        elif name.startswith(P4_LATE_TRAINABLE_PREFIXES):
            core_parameters.append(parameter)
    if not input_parameters or not core_parameters:
        raise RuntimeError(
            f"invalid P4 optimizer groups: input={len(input_parameters)}, "
            f"core={len(core_parameters)}"
        )
    return torch.optim.AdamW(
        [
            {
                "params": input_parameters,
                "lr": float(input_learning_rate),
                "name": "sar_input_head",
            },
            {
                "params": core_parameters,
                "lr": float(core_learning_rate),
                "name": "fusion3_clf2",
            },
        ],
        weight_decay=float(weight_decay),
    )


def augment_metrics(
    metrics: Dict[str, float],
    loc_hist: torch.Tensor,
) -> Dict[str, float]:
    tp = loc_hist[1, 1].double()
    precision = tp / loc_hist[:, 1].sum().clamp_min(1.0)
    recall = tp / loc_hist[1].sum().clamp_min(1.0)
    metrics["loc_precision"] = float(precision.item())
    metrics["loc_recall"] = float(recall.item())
    metrics["balanced_score"] = metrics["damage_score"]
    return metrics


@torch.no_grad()
def evaluate(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_damage_hist: Dict[str, torch.Tensor] = {}
    processed = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        pre, sar, _, location, damage, names = parse_p4_batch(batch, device)
        output_loc, output_damage = model(pre, sar)
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
    metrics = metrics_from_hist(loc_hist, damage_hist)
    metrics = augment_metrics(metrics, loc_hist)
    # Occurrence diagnostics use the declared BRIGHT class order:
    # 0=background, 1=intact, 2=damaged, 3=destroyed.  Gate-miss is broader
    # than Dmg->Int because it also counts damaged pixels predicted background.
    affected_total = damage_hist[2:4].sum().clamp_min(1.0)
    affected_true_positive = damage_hist[2:4, 2:4].sum()
    damaged_total = damage_hist[2].sum().clamp_min(1.0)
    metrics["affected_recall"] = float(
        (affected_true_positive / affected_total).item()
    )
    metrics["damaged_gate_miss"] = float(
        (damage_hist[2, 0:2].sum() / damaged_total).item()
    )
    metrics["processed_images"] = processed
    event_mious: List[float] = []
    event_miou_by_name: Dict[str, float] = {}
    for event_name, hist in event_damage_hist.items():
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - intersection
        event_iou = intersection / union.clamp_min(1.0)
        event_miou = float(event_iou.mean().item())
        event_mious.append(event_miou)
        event_miou_by_name[event_name] = event_miou
    metrics["event_count"] = len(event_mious)
    metrics["event_mean_official_style_4class_miou"] = (
        sum(event_mious) / len(event_mious) if event_mious else 0.0
    )
    metrics["event_worst_official_style_4class_miou"] = (
        min(event_mious) if event_mious else 0.0
    )
    metrics["event_miou_by_name"] = event_miou_by_name
    return metrics


def format_metrics(metrics: Mapping[str, float]) -> str:
    return (
        f"valid_mIoU={100*metrics['valid_miou']:.4f} | "
        f"mIoU4={100*metrics['miou4']:.4f} | "
        f"Loc P/R/F1={100*metrics['loc_precision']:.2f}/"
        f"{100*metrics['loc_recall']:.2f}/{100*metrics['loc_f1']:.2f} | "
        f"IoU(int/dmg/des)={100*metrics['intact_iou']:.2f}/"
        f"{100*metrics['damaged_iou']:.2f}/"
        f"{100*metrics['destroyed_iou']:.2f} | "
        f"Dmg->Int={100*metrics['damaged_to_intact']:.2f}% | "
        f"Des->Int={100*metrics['destroyed_to_intact']:.2f}%"
        f" | Affected R={100*metrics['affected_recall']:.2f}%"
        f" | Dmg gate-miss={100*metrics['damaged_gate_miss']:.2f}%"
        f" | Event mean/worst="
        f"{100*metrics['event_mean_official_style_4class_miou']:.2f}/"
        f"{100*metrics['event_worst_official_style_4class_miou']:.2f}"
    )


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_metrics: Mapping[str, float],
    args,
    early_stopped: bool = False,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "experiment": "S8_P4_NSMICB",
        "variant": str(args.variant),
        "best_metrics": dict(best_metrics),
        "args": vars(args),
        "early_stopped": bool(early_stopped),
    }
    metadata_fn = getattr(model, "experiment_metadata", None)
    if callable(metadata_fn):
        payload["model_metadata"] = metadata_fn()
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    return payload


def _load_resume_model(
    model: nn.Module,
    path: str,
) -> Tuple[Mapping[str, object], int, Dict[str, float]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resume checkpoint must be a mapping")
    saved_variant = str(checkpoint.get("variant", "")).lower()
    model_variant = str(getattr(model, "variant", "")).lower()
    if saved_variant and saved_variant != model_variant:
        raise RuntimeError(
            f"resume variant mismatch: checkpoint={saved_variant}, model={model_variant}"
        )
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("resume checkpoint has no model state")
    cleaned = {
        (key[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return checkpoint, int(checkpoint.get("step", 0)), dict(
        checkpoint.get("best_metrics", {})
    )


def main() -> None:
    parser = argparse.ArgumentParser("Train S8-P4 NS-MICB A0...A4")
    parser.add_argument("--variant", choices=P4_DATASET_VARIANTS, required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_json", default="")
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--expected_resume_step",
        type=int,
        default=-1,
        help="Fail fast unless the resume checkpoint is from this optimizer step.",
    )
    parser.add_argument(
        "--admit_resume_checkpoint",
        action="store_true",
        help=(
            "Copy the validated resume state into the new experiment as the "
            "safe selectable baseline before further optimisation."
        ),
    )
    parser.add_argument(
        "--expected_resume_valid_miou_percent", type=float, default=-1.0
    )
    parser.add_argument(
        "--resume_metric_tolerance_pp", type=float, default=0.02
    )
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--core_learning_rate", type=float, default=1e-6)
    parser.add_argument("--input_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--input_only_steps", type=int, default=1000)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--use_lovasz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument("--min_instance_area", type=int, default=32)
    parser.add_argument("--max_bags_per_class_per_batch", type=int, default=16)
    parser.add_argument("--mil_warmup_steps", type=int, default=1000)
    parser.add_argument(
        "--mil_ramp_steps",
        type=int,
        default=0,
        help=(
            "Linearly ramp building-level supervision after mil_warmup_steps. "
            "Zero preserves the original hard 0-to-1 switch."
        ),
    )
    parser.add_argument("--lambda_mil", type=float, default=0.25)
    parser.add_argument("--lambda_exist_aux", type=float, default=0.25)
    parser.add_argument("--lambda_severity_aux", type=float, default=0.25)
    parser.add_argument("--lambda_intact_guard", type=float, default=0.05)
    parser.add_argument("--lambda_coverage", type=float, default=0.0)

    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--lambda_kd_damage", type=float, default=1.0)
    parser.add_argument("--lambda_kd_location", type=float, default=0.25)
    parser.add_argument("--lambda_kd_feature", type=float, default=0.1)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument("--expected_a0_valid_miou", type=float, default=60.5019)
    parser.add_argument("--a0_tolerance", type=float, default=0.10)
    parser.add_argument("--early_stop_drop_pp", type=float, default=0.3)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    args = parser.parse_args()

    if args.variant != "a0" and not args.sar_stats_json:
        raise ValueError(
            "A1/A2/A3/A4 require --sar_stats_json generated from train only"
        )
    if args.core_learning_rate > 1e-6:
        raise ValueError("P4 core_learning_rate is locked to <= 1e-6")
    if args.input_learning_rate != 1e-4:
        raise ValueError("P4 input_learning_rate is locked to 1e-4")

    device, distributed, rank, world_size, _ = setup_runtime(args.device)
    if distributed or rank != 0 or world_size != 1:
        raise RuntimeError("S8-P4 controlled ablations intentionally use one GPU")
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
    student = DamageFormer_S8_P4_NativeSAR_MICB(
        variant=args.variant, **model_kwargs
    ).to(device)
    teacher = DamageFormer_S8_P4_NativeSAR_MICB(
        variant="a0", **model_kwargs
    ).to(device)
    teacher_report = load_p3_checkpoint(teacher, args.base_ckpt)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    resume_checkpoint: Mapping[str, object] | None = None
    if args.resume:
        resume_checkpoint, start_step, best_metrics = _load_resume_model(
            student, args.resume
        )
        if (
            args.expected_resume_step >= 0
            and start_step != args.expected_resume_step
        ):
            raise RuntimeError(
                "resume step mismatch: "
                f"observed={start_step}, expected={args.expected_resume_step}"
            )
        if args.expected_resume_valid_miou_percent >= 0:
            observed = 100.0 * float(best_metrics.get("valid_miou", -1.0))
            if (
                abs(observed - args.expected_resume_valid_miou_percent)
                > args.resume_metric_tolerance_pp
            ):
                raise RuntimeError(
                    "resume validation metric mismatch: "
                    f"observed={observed:.4f}, "
                    f"expected={args.expected_resume_valid_miou_percent:.4f} "
                    f"+/- {args.resume_metric_tolerance_pp:.4f} pp"
                )
        migration_report: Dict[str, object] = {
            "resume": args.resume,
            "teacher": teacher_report,
        }
    else:
        student_stats = (
            load_sar_stats(args.sar_stats_json)
            if args.variant == "a4"
            else None
        )
        migration_report = {
            "student": load_p3_checkpoint(
                student,
                args.base_ckpt,
                sar_stats=student_stats,
            ),
            "teacher": teacher_report,
        }
        start_step, best_metrics = 0, {}

    input_only = start_step < args.input_only_steps
    parameter_counts = configure_p4_trainable(student, input_only=input_only)
    optimizer = build_optimizer(
        student,
        args.core_learning_rate,
        args.input_learning_rate,
        args.weight_decay,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if scaler.is_enabled() and "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])

    train_set = build_p4_dataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        "train",
        args.variant,
        args.sar_stats_json,
    )
    val_set = build_p4_dataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        "val",
        args.variant,
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
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
    with open(output_dir / "migration_report.json", "w", encoding="utf-8") as file:
        json.dump(migration_report, file, ensure_ascii=False, indent=2)
    checkpoint_identity = {
        "base_checkpoint": os.path.abspath(args.base_ckpt),
        "base_checkpoint_sha256": file_sha256(args.base_ckpt),
        "expected_experiment": "S8_MICB_DSMIL_P3",
        "accepted_file_step": 4000,
        "accepted_val_miou4_percent": 69.57721843922928,
        "accepted_valid_miou_percent": args.expected_a0_valid_miou,
    }
    with open(output_dir / "checkpoint_identity.json", "w", encoding="utf-8") as file:
        json.dump(checkpoint_identity, file, ensure_ascii=False, indent=2)
    with open(
        output_dir / "base_checkpoint_sha256.txt", "w", encoding="utf-8"
    ) as file:
        file.write(checkpoint_identity["base_checkpoint_sha256"] + "\n")
    if args.sar_stats_json and os.path.isfile(args.sar_stats_json):
        destination = output_dir / "sar_train_stats.json"
        if os.path.abspath(args.sar_stats_json) != os.path.abspath(destination):
            shutil.copy2(args.sar_stats_json, destination)
    runtime = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "parameter_counts": parameter_counts,
        "student_metadata": student.experiment_metadata(),
        "teacher_frozen": True,
    }
    with open(output_dir / "runtime_metadata.json", "w", encoding="utf-8") as file:
        json.dump(runtime, file, ensure_ascii=False, indent=2)

    spec = get_p4_variant_spec(args.variant)
    print("=" * 80, flush=True)
    print(f"Experiment       : {spec.name}", flush=True)
    print(f"Base P3          : {args.base_ckpt}", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print(f"SAR channels     : {spec.sar_channels}", flush=True)
    print(f"SAR normalization: {spec.sar_normalization}", flush=True)
    print(f"SAR input head   : {spec.sar_input_head}", flush=True)
    print("R1/R4/reliability: DISABLED / DISABLED / DISABLED", flush=True)
    print(
        f"LR input/core    : {args.input_learning_rate}/{args.core_learning_rate}",
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
    bad_validation_count = 0

    if args.admit_resume_checkpoint:
        if resume_checkpoint is None:
            raise ValueError("--admit_resume_checkpoint requires --resume")
        if not best_metrics or "valid_miou" not in best_metrics:
            raise RuntimeError(
                "resume checkpoint has no validated best_metrics; refusing to "
                "admit an unvalidated safe state"
            )
        _atomic_save(
            checkpoint_payload(
                student, optimizer, scaler, start_step, best_metrics, args
            ),
            output_dir / "best_val_miou.pth",
        )
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "step": start_step,
                        "phase": "resume_safe_state",
                        "source_checkpoint": os.path.abspath(args.resume),
                        **best_metrics,
                    }
                )
                + "\n"
            )
        print(
            "Resume safe state admitted to selection | "
            f"step={start_step} | valid_mIoU={100*best_valid_miou:.4f}",
            flush=True,
        )

    if start_step == 0 and not args.skip_initial_eval:
        initial = evaluate(
            student, val_loader, device, args.ignore_index, args.val_max_batches
        )
        print("Initial validation | " + format_metrics(initial), flush=True)
        if (
            args.variant in ("a0", "a4")
            and args.val_max_batches == 0
            and abs(100.0 * initial["valid_miou"] - args.expected_a0_valid_miou)
            > args.a0_tolerance
        ):
            raise RuntimeError(
                "A0 does not reproduce the accepted P3 baseline: "
                f"observed={100*initial['valid_miou']:.4f}, "
                f"expected={args.expected_a0_valid_miou:.4f} +/- "
                f"{args.a0_tolerance:.4f}"
            )
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
            file.write(
                json.dumps({"step": 0, "phase": "initial", **initial}) + "\n"
            )
        if initial["valid_miou"] > best_valid_miou:
            best_valid_miou = initial["valid_miou"]
            best_metrics = dict(initial)
            _atomic_save(
                checkpoint_payload(
                    student, optimizer, scaler, 0, best_metrics, args
                ),
                output_dir / "best_val_miou.pth",
            )

    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    window_start = time.perf_counter()
    window_step = step
    early_stopped = False
    set_p4_training_mode(student, input_only)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < args.optimizer_steps and not early_stopped:
        for batch in train_loader:
            if input_only and step >= args.input_only_steps:
                input_only = False
                parameter_counts = configure_p4_trainable(student, input_only=False)
                set_p4_training_mode(student, input_only=False)
                print(
                    f"Unlocked fusion_layer_3 + clf_2 at step {step}; "
                    f"trainable={parameter_counts['trainable']:,}",
                    flush=True,
                )
            else:
                set_p4_training_mode(student, input_only)

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
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    teacher_outputs = teacher(
                        pre, teacher_sar, return_aux=True
                    )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                student_outputs = student(
                    pre, sar, bag_specs=bags, return_aux=True
                )

            output_loc, output_damage, student_aux = student_outputs
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
            bag_losses = p3_bag_losses(student_aux["bag_result"], output_damage)
            kd_losses = distillation_losses(
                student_outputs, teacher_outputs, args.kd_temperature
            )
            mil_scale = mil_supervision_scale(
                step, args.mil_warmup_steps, args.mil_ramp_steps
            )
            loss_bag = (
                args.lambda_mil * bag_losses["loss_mil"]
                + args.lambda_exist_aux * bag_losses["loss_exist_aux"]
                + args.lambda_severity_aux * bag_losses["loss_severity_aux"]
                + args.lambda_intact_guard * bag_losses["loss_intact_guard"]
                + args.lambda_coverage * bag_losses["loss_coverage"]
            )
            loss_kd = (
                args.lambda_kd_damage * kd_losses["loss_kd_damage"]
                + args.lambda_kd_location * kd_losses["loss_kd_location"]
                + args.lambda_kd_feature * kd_losses["loss_kd_feature"]
            )
            total_loss = loss_loc + loss_damage + mil_scale * loss_bag + loss_kd
            if not bool(torch.isfinite(total_loss.detach()).item()):
                raise FloatingPointError(
                    f"non-finite loss at step={step}: loc={loss_loc.item()}, "
                    f"damage={loss_damage.item()}, bag={loss_bag.item()}, "
                    f"kd={loss_kd.item()}"
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
                    "optimizer_steps": step,
                    "crops_seen": (
                        step
                        * args.train_batch_size
                        * args.accumulation_steps
                    ),
                    "phase": "input_only" if input_only else "selective_unfreeze",
                    "total_loss": float(total_loss.detach().item()),
                    "loc_loss": float(loss_loc.detach().item()),
                    "cls_loss": float(loss_damage.detach().item()),
                    "ce_loss": float(
                        (ce_loc + ce_damage).detach().item()
                    ),
                    "lovasz_loss": float(
                        (lovasz_loc + lovasz_damage).detach().item()
                    ),
                    "loc_ce_loss": float(ce_loc.detach().item()),
                    "loc_lovasz_loss": float(lovasz_loc.detach().item()),
                    "cls_ce_loss": float(ce_damage.detach().item()),
                    "cls_lovasz_loss": float(lovasz_damage.detach().item()),
                    "mil_loss": float(bag_losses["loss_mil"].detach().item()),
                    "exist_loss": float(
                        bag_losses["loss_exist_aux"].detach().item()
                    ),
                    "severity_loss": float(
                        bag_losses["loss_severity_aux"].detach().item()
                    ),
                    "intact_guard_loss": float(
                        bag_losses["loss_intact_guard"].detach().item()
                    ),
                    "kd_damage_loss": float(
                        kd_losses["loss_kd_damage"].detach().item()
                    ),
                    "kd_location_loss": float(
                        kd_losses["loss_kd_location"].detach().item()
                    ),
                    "kd_feature_loss": float(
                        kd_losses["loss_kd_feature"].detach().item()
                    ),
                    "bag_count": int(student_aux["bag_count"]),
                    "mil_scale": mil_scale,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_input": optimizer.param_groups[0]["lr"],
                    "lr_core": optimizer.param_groups[1]["lr"],
                    "optimizer_steps_per_second": completed / elapsed,
                    "seconds_per_optimizer_step": elapsed / completed,
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
                metrics = evaluate(
                    student,
                    val_loader,
                    device,
                    args.ignore_index,
                    args.val_max_batches,
                )
                print("Validation | " + format_metrics(metrics), flush=True)
                with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
                    file.write(json.dumps({"step": step, **metrics}) + "\n")
                if metrics["valid_miou"] > best_valid_miou:
                    best_valid_miou = metrics["valid_miou"]
                    best_metrics = dict(metrics)
                    _atomic_save(
                        checkpoint_payload(
                            student, optimizer, scaler, step, best_metrics, args
                        ),
                        output_dir / "best_val_miou.pth",
                    )
                    print(
                        f"New best valid_mIoU={100*best_valid_miou:.4f}",
                        flush=True,
                    )

                below_guard = (
                    args.val_max_batches == 0
                    and step > args.input_only_steps
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
                        "Early stop: validation remained more than "
                        f"{args.early_stop_drop_pp:.3f} pp below the accepted "
                        f"baseline for {bad_validation_count} validations.",
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
                )
                _atomic_save(payload, output_dir / f"checkpoint_step_{step}.pth")
                _atomic_save(payload, output_dir / "latest_resume.pth")
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
    )
    _atomic_save(final_payload, output_dir / "final.pth")
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "variant": args.variant,
                "completed_step": step,
                "target_optimizer_steps": args.optimizer_steps,
                "crops_seen": (
                    step * args.train_batch_size * args.accumulation_steps
                ),
                "early_stopped": early_stopped,
                "best_valid_miou": best_valid_miou,
                "best_metrics": best_metrics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"Training complete: step={step}, early_stopped={early_stopped}, "
        f"best valid_mIoU={100*best_valid_miou:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
