#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-O1: isolated 500-step direct Occurrence supervision screen.

This experiment starts from the accepted P3 checkpoint and trains only the
last damage fusion block and four-class classifier.  It uses natural Train
images and ordinary labels; no Memory cache, retrieval key, pair graph,
projector, validation-selected threshold, or Test sample is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import build_p4_dataset
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
    lock_frozen_batch_norm,
)
from script.standard_ML.p3_direct_occurrence_runtime import (
    balanced_occurrence_bce,
    o1_validation_checks,
)
from script.standard_ML.p3_hard_occurrence_runtime import protected_spatial_kl
from script.standard_ML.train_S8_P4_native_sar_micb import (
    evaluate,
    format_metrics,
    parse_p4_batch,
)
from train_S5_S8_GPU import segmentation_loss, seed_worker, set_random_seed


O1_TRAINABLE_PREFIXES = ("fusion_layer_3.", "clf_2.")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def o1_model_delta(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(O1_TRAINABLE_PREFIXES)
    }


def load_o1_model_delta(
    model: nn.Module,
    delta: Mapping[str, torch.Tensor],
) -> None:
    expected = {
        name
        for name in model.state_dict()
        if name.startswith(O1_TRAINABLE_PREFIXES)
    }
    actual = set(delta)
    if actual != expected:
        raise RuntimeError(
            "O1 model-delta contract mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    incompatible = model.load_state_dict(dict(delta), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected O1 delta keys: {incompatible.unexpected_keys}"
        )


def configure_o1_trainable(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(O1_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    lock_frozen_batch_norm(model)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def set_o1_train_mode(model: DamageFormer_S8_P4_NativeSAR_MICB) -> None:
    model.eval()
    model.fusion_layer_3.train()
    model.clf_2.train()
    lock_frozen_batch_norm(model)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "protocol": "P3_O1_direct_occurrence_delta_checkpoint_v1",
        "base_p3_checkpoint": str(args.p3_checkpoint),
        "base_p3_checkpoint_sha256": sha256(args.p3_checkpoint),
        "model_delta": o1_model_delta(model),
        "model_delta_prefixes": list(O1_TRAINABLE_PREFIXES),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "metrics": dict(metrics),
        "config": vars(args),
        "inference_uses_memory": False,
        "validation_selects_loss_weight": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser("P3-O1 direct Occurrence screen")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--optimizer_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--occurrence_weight", type=float, default=0.05)
    parser.add_argument("--protect_weight", type=float, default=0.5)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--expected_valid_miou", type=float, default=60.5019)
    parser.add_argument("--baseline_tolerance_pp", type=float, default=0.10)
    parser.add_argument("--minimum_damaged_gain_pp", type=float, default=0.30)
    parser.add_argument("--maximum_guard_drop_pp", type=float, default=0.20)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    args = parser.parse_args()

    if args.optimizer_steps != 500:
        raise ValueError("P3-O1 screen is locked to 500 optimizer steps")
    if args.eval_every != 250:
        raise ValueError("P3-O1 eval_every is locked to 250")
    if args.occurrence_weight <= 0:
        raise ValueError("occurrence_weight must be positive")
    if args.protect_weight < 0:
        raise ValueError("protect_weight cannot be negative")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = set_random_seed(args.seed, 0, False)
    # Match the accepted P3/H1 replay runtime and avoid batch-dependent cuDNN
    # algorithm switching at numerically sensitive localization boundaries.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    student = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    teacher = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(student, args.p3_checkpoint)
    load_p3_checkpoint(teacher, args.p3_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    parameter_counts = configure_o1_trainable(student)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_set = build_p4_dataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        "train",
        "a0",
        "",
    )
    val_set = build_p4_dataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        "val",
        "a0",
        "",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline = evaluate(student, val_loader, device, args.ignore_index, 0)
    print("Initial validation | " + format_metrics(baseline), flush=True)
    if (
        abs(100.0 * baseline["valid_miou"] - args.expected_valid_miou)
        > args.baseline_tolerance_pp
    ):
        raise RuntimeError("P3-O1 Step-0 does not reproduce accepted P3")

    baseline_checks, baseline_diagnostics = o1_validation_checks(
        baseline,
        baseline,
        minimum_damaged_gain_pp=args.minimum_damaged_gain_pp,
        maximum_guard_drop_pp=args.maximum_guard_drop_pp,
        minimum_nonnegative_events=args.minimum_nonnegative_events,
    )
    history = [
        {
            "step": 0,
            "metrics": baseline,
            "checks": baseline_checks,
            "diagnostics": baseline_diagnostics,
            "passed": False,
        }
    ]
    checkpoint_hash = sha256(args.p3_checkpoint)
    (output_dir / "safe_step0.json").write_text(
        json.dumps(
            {
                "protocol": "P3_O1_safe_step0_reference_v1",
                "step": 0,
                "base_p3_checkpoint": str(args.p3_checkpoint),
                "base_p3_checkpoint_sha256": checkpoint_hash,
                "metrics": baseline,
                "inference_uses_memory": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16
    train_iterator = iter(train_loader)
    best_admitted = None
    start_time = time.perf_counter()

    for step in range(1, args.optimizer_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        set_o1_train_mode(student)
        optimizer.zero_grad(set_to_none=True)
        pre, sar, teacher_sar, _, damage, _ = parse_p4_batch(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            _, teacher_damage, _ = teacher(
                pre, teacher_sar, return_aux=True
            )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            _, student_damage, _ = student(pre, sar, return_aux=True)
            damage_loss, _, _ = segmentation_loss(
                student_damage,
                damage,
                ce,
                True,
                args.ce_weight,
                args.lovasz_weight,
                args.ignore_index,
            )
            occurrence_loss, occurrence_diagnostics = balanced_occurrence_bce(
                student_damage,
                damage,
                ignore_index=args.ignore_index,
            )
            protect_loss = protected_spatial_kl(
                student_damage,
                teacher_damage,
                damage,
                temperature=args.kd_temperature,
                protected_classes=(1, 3),
                ignore_index=args.ignore_index,
            )
            total_loss = (
                damage_loss
                + args.occurrence_weight * occurrence_loss
                + args.protect_weight * protect_loss
            )
        total_loss.backward()

        grad_norms = {
            prefix: sum(
                float(parameter.grad.detach().abs().sum())
                for name, parameter in student.named_parameters()
                if name.startswith(prefix + ".") and parameter.grad is not None
            )
            for prefix in ("fusion_layer_3", "clf_2")
        }
        if step == 1 and any(value <= 0.0 for value in grad_norms.values()):
            raise RuntimeError(f"P3-O1 gradient contract failed: {grad_norms}")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            5.0,
        )
        optimizer.step()

        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "total_loss": float(total_loss.detach()),
                        "damage_loss": float(damage_loss.detach()),
                        "occurrence_loss": float(occurrence_loss.detach()),
                        "protect_loss": float(protect_loss.detach()),
                        "occurrence": occurrence_diagnostics,
                        "grad_norms": grad_norms,
                        "steps_per_second": step
                        / max(time.perf_counter() - start_time, 1e-6),
                    }
                ),
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics = evaluate(
                student, val_loader, device, args.ignore_index, 0
            )
            checks, diagnostics = o1_validation_checks(
                metrics,
                baseline,
                minimum_damaged_gain_pp=args.minimum_damaged_gain_pp,
                maximum_guard_drop_pp=args.maximum_guard_drop_pp,
                minimum_nonnegative_events=args.minimum_nonnegative_events,
            )
            result = {
                "step": step,
                "metrics": metrics,
                "checks": checks,
                "diagnostics": diagnostics,
                "passed": all(checks.values()),
            }
            history.append(result)
            print("Validation | " + format_metrics(metrics), flush=True)
            print(f"P3-O1 checks: {checks}", flush=True)
            atomic_save(
                checkpoint_payload(
                    student,
                    optimizer,
                    step=step,
                    metrics=metrics,
                    args=args,
                ),
                output_dir / "latest_resume.pth",
            )
            if result["passed"] and (
                best_admitted is None
                or diagnostics["damaged_gain_pp"]
                > best_admitted["diagnostics"]["damaged_gain_pp"]
            ):
                best_admitted = result
                atomic_save(
                    checkpoint_payload(
                        student,
                        optimizer,
                        step=step,
                        metrics=metrics,
                        args=args,
                    ),
                    output_dir / "best_admitted.pth",
                )

    decision = (
        "ADMIT_P3_O1_TO_1500_STEP_EXTENSION_NO_TEST"
        if best_admitted is not None
        else "STOP_P3_O1_DIRECT_OCCURRENCE_SCREEN_DO_NOT_TEST"
    )
    summary = {
        "protocol": "P3_O1_direct_occurrence_screen_summary_v1",
        "baseline": baseline,
        "history": history,
        "best_admitted": best_admitted,
        "selected_checkpoint": (
            str(output_dir / "best_admitted.pth")
            if best_admitted
            else str(args.p3_checkpoint)
        ),
        "parameter_counts": parameter_counts,
        "decision": decision,
        "inference_uses_memory": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Decision: {decision}")
    print(f"Selected: {summary['selected_checkpoint']}")
    print(f"Saved: {output_dir}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
