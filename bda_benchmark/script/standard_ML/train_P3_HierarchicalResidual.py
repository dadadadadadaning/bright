#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-HR0/HR1/HR2/HR3-O isolated hierarchical residual experiment.

HR0 performs a full frozen-P3 validation guard.  HR1 trains only deep-feature
Occurrence and Severity residual heads.  HR2 changes the Occurrence input to a
Stage2+Stage3 explicit RGB--SAR relation pyramid.  Test and Memory are absent.
HR3-O keeps the HR2 evidence path, disables Severity, and trains a region-level
Occurrence repair/protection objective.
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
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import build_p4_dataset
from model.DamageFormer_P3_HierarchicalResidual import (
    DamageFormerP3HierarchicalResidual,
    VALID_P3_HR_VARIANTS,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import load_p3_checkpoint
from script.standard_ML.p3_hierarchical_residual_runtime import (
    hierarchical_residual_loss,
    hr3o_region_occurrence_loss,
    hr_validation_checks,
    severity_validation_checks,
)
from script.standard_ML.train_S8_P4_native_sar_micb import (
    evaluate,
    format_metrics,
    parse_p4_batch,
)
from train_S5_S8_GPU import seed_worker, set_random_seed


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def checkpoint_payload(
    model: DamageFormerP3HierarchicalResidual,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    consumed_batches: int,
    metrics: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "protocol": "P3_HR_hierarchical_residual_delta_v2",
        "variant": model.variant,
        "base_p3_checkpoint": str(args.p3_checkpoint),
        "base_p3_checkpoint_sha256": sha256(args.p3_checkpoint),
        "hierarchical_modules": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "consumed_batches": int(consumed_batches),
        "metrics": dict(metrics),
        "config": vars(args),
        "p3_frozen": True,
        "inference_uses_memory": False,
        "validation_accessed": True,
        "test_accessed": False,
    }


def build_loaders(args: argparse.Namespace, device: torch.device):
    generator = set_random_seed(args.seed, 0, False)
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
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser("P3-HR hierarchical residual experiment")
    parser.add_argument("--variant", choices=VALID_P3_HR_VARIANTS, required=True)
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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hard_fraction", type=float, default=0.25)
    parser.add_argument("--minimum_hard_pixels", type=int, default=64)
    parser.add_argument("--minimum_component_area", type=int, default=16)
    parser.add_argument("--repairable_min_margin", type=float, default=-2.0)
    parser.add_argument("--repairable_max_margin", type=float, default=0.0)
    parser.add_argument("--required_margin", type=float, default=0.0)
    parser.add_argument("--four_class_weight", type=float, default=0.25)
    parser.add_argument("--occurrence_weight", type=float, default=1.0)
    parser.add_argument("--severity_weight", type=float, default=0.5)
    parser.add_argument("--protection_weight", type=float, default=0.10)
    parser.add_argument("--maximum_occurrence_delta", type=float, default=2.0)
    parser.add_argument("--maximum_severity_delta", type=float, default=2.0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--expected_valid_miou", type=float, default=60.5019)
    parser.add_argument("--baseline_tolerance_pp", type=float, default=0.10)
    parser.add_argument("--minimum_damaged_gain_pp", type=float, default=0.30)
    parser.add_argument("--minimum_gate_miss_gain_pp", type=float, default=0.50)
    parser.add_argument("--minimum_affected_recall_gain_pp", type=float, default=0.50)
    parser.add_argument("--maximum_guard_drop_pp", type=float, default=0.20)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    args = parser.parse_args()

    if args.variant == "hr0" and args.optimizer_steps != 0:
        raise ValueError("HR0 is a zero-step exact P3 guard")
    if args.variant != "hr0" and args.optimizer_steps != 500:
        raise ValueError("HR1/HR2/HR3-O screens are locked to 500 optimiser steps")
    if args.variant != "hr0" and args.eval_every != 250:
        raise ValueError("HR1/HR2/HR3-O eval_every is locked to 250")
    if args.variant == "hr3o":
        if args.repairable_min_margin < -args.maximum_occurrence_delta:
            raise ValueError(
                "HR3-O repairable_min_margin is outside the bounded "
                "Occurrence residual's reachable range"
            )
        if args.protection_weight <= 0.0:
            raise ValueError("HR3-O requires positive I/D/X protection weight")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    model = DamageFormerP3HierarchicalResidual(
        args.variant,
        maximum_occurrence_delta=args.maximum_occurrence_delta,
        maximum_severity_delta=args.maximum_severity_delta,
    ).to(device)
    load_p3_checkpoint(model.p3, args.p3_checkpoint)
    model.p3.eval()
    for parameter in model.p3.parameters():
        parameter.requires_grad_(False)

    train_loader, val_loader = build_loaders(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Exact online guard on a real batch, before any optimisation.
    guard_batch = next(iter(train_loader))
    guard_pre, guard_sar, _, _, _, _ = parse_p4_batch(guard_batch, device)
    model.eval()
    with torch.no_grad():
        _, guard_output, guard_aux = model(guard_pre, guard_sar, return_aux=True)
    if not torch.equal(guard_output, guard_aux["base_damage"]):
        maximum = float((guard_output - guard_aux["base_damage"]).abs().max())
        raise RuntimeError(f"P3-HR Step-0 is not exact P3; max_delta={maximum}")

    baseline = evaluate(model, val_loader, device, args.ignore_index, 0)
    print("Initial validation | " + format_metrics(baseline), flush=True)
    if (
        abs(100.0 * float(baseline["valid_miou"]) - args.expected_valid_miou)
        > args.baseline_tolerance_pp
    ):
        raise RuntimeError("P3-HR Step-0 does not reproduce accepted P3 metrics")

    if args.variant == "hr0":
        summary = {
            "protocol": "P3_HR0_exact_guard_v1",
            "variant": args.variant,
            "baseline": baseline,
            "exact_online_guard": True,
            "decision": "ADMIT_TO_P3_HR1_500_STEP_SCREEN",
            "selected_checkpoint": str(args.p3_checkpoint),
            "p3_frozen": True,
            "validation_accessed": True,
            "test_accessed": False,
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Decision: ADMIT_TO_P3_HR1_500_STEP_SCREEN")
        print(f"Saved: {output_dir}")
        print("Test accessed: False")
        return

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("P3-HR screen has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = [{"step": 0, "metrics": baseline, "passed": False}]
    best_admitted = None
    train_iterator = iter(train_loader)
    successful_step = 0
    consumed_batches = 0
    skipped_batches = 0
    start_time = time.perf_counter()
    amp_enabled = device.type == "cuda"

    while successful_step < args.optimizer_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        consumed_batches += 1
        pre, sar, _, _, target, _ = parse_p4_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            _, output, aux = model(pre, sar, return_aux=True)
            if args.variant == "hr3o":
                loss, diagnostics = hr3o_region_occurrence_loss(
                    aux["base_damage"],
                    aux["new_occurrence"],
                    aux["base_occurrence"],
                    aux["occurrence_delta"],
                    aux["severity_delta"],
                    target,
                    repairable_min_margin=args.repairable_min_margin,
                    repairable_max_margin=args.repairable_max_margin,
                    required_margin=args.required_margin,
                    minimum_component_area=args.minimum_component_area,
                    occurrence_weight=args.occurrence_weight,
                    protection_weight=args.protection_weight,
                    ignore_index=args.ignore_index,
                )
            else:
                loss, diagnostics = hierarchical_residual_loss(
                    output,
                    aux["base_damage"],
                    aux["new_occurrence"],
                    aux["base_occurrence"],
                    aux["new_severity"],
                    aux["base_severity"],
                    aux["occurrence_delta"],
                    aux["severity_delta"],
                    target,
                    hard_fraction=args.hard_fraction,
                    minimum_hard_pixels=args.minimum_hard_pixels,
                    required_margin=args.required_margin,
                    four_class_weight=args.four_class_weight,
                    occurrence_weight=args.occurrence_weight,
                    severity_weight=args.severity_weight,
                    protection_weight=args.protection_weight,
                    ignore_index=args.ignore_index,
                )
        if loss is None:
            skipped_batches += 1
            if consumed_batches > 1000 and successful_step == 0:
                raise RuntimeError("no usable Damaged+Intact batch in 1000 attempts")
            continue
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite P3-HR loss: {float(loss.detach())}")
        loss.backward()
        gradients = [
            parameter.grad.detach().abs().sum()
            for parameter in trainable
            if parameter.grad is not None
        ]
        grad_norm = float(torch.stack(gradients).sum()) if gradients else 0.0
        if grad_norm <= 0.0:
            raise RuntimeError("P3-HR trainable modules received no gradient")
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        successful_step += 1

        if successful_step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        "step": successful_step,
                        "consumed_batches": consumed_batches,
                        "skipped_batches": skipped_batches,
                        "loss": float(loss.detach()),
                        "grad_norm_l1": grad_norm,
                        "hierarchical": diagnostics,
                        "steps_per_second": successful_step
                        / max(time.perf_counter() - start_time, 1e-6),
                    }
                ),
                flush=True,
            )

        if successful_step % args.eval_every == 0:
            model.eval()
            metrics = evaluate(model, val_loader, device, args.ignore_index, 0)
            if args.variant == "hr1s":
                checks, gate_diagnostics = severity_validation_checks(
                    metrics,
                    baseline,
                    maximum_guard_drop_pp=args.maximum_guard_drop_pp,
                    minimum_nonnegative_events=args.minimum_nonnegative_events,
                )
            else:
                checks, gate_diagnostics = hr_validation_checks(
                    metrics,
                    baseline,
                    minimum_damaged_gain_pp=args.minimum_damaged_gain_pp,
                    minimum_gate_miss_gain_pp=args.minimum_gate_miss_gain_pp,
                    minimum_affected_recall_gain_pp=args.minimum_affected_recall_gain_pp,
                    maximum_guard_drop_pp=args.maximum_guard_drop_pp,
                    minimum_nonnegative_events=args.minimum_nonnegative_events,
                )
            result = {
                "step": successful_step,
                "metrics": metrics,
                "checks": checks,
                "diagnostics": gate_diagnostics,
                "passed": all(checks.values()),
            }
            history.append(result)
            print("Validation | " + format_metrics(metrics), flush=True)
            print(f"P3-{args.variant.upper()} checks: {checks}", flush=True)
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    step=successful_step,
                    consumed_batches=consumed_batches,
                    metrics=metrics,
                    args=args,
                ),
                output_dir / "latest_delta.pth",
            )
            if result["passed"] and (
                best_admitted is None
                or gate_diagnostics["damaged_gain_pp"]
                > best_admitted["diagnostics"]["damaged_gain_pp"]
            ):
                best_admitted = result
                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        step=successful_step,
                        consumed_batches=consumed_batches,
                        metrics=metrics,
                        args=args,
                    ),
                    output_dir / "best_admitted.pth",
                )

    if args.variant == "hr1" and best_admitted is None:
        # HR1 is the mandatory no-new-evidence ablation.  HR2 remains a
        # predeclared independent test because it adds the Stage2+3 evidence
        # whose absence may be exactly why HR1 fails.
        decision = "HR1_COMPLETE_NO_GAIN_PROCEED_TO_PREDECLARED_HR2_NO_TEST"
        selected = str(args.p3_checkpoint)
    elif args.variant == "hr1":
        decision = "HR1_PASSED_PROCEED_TO_PREDECLARED_HR2_NO_TEST"
        selected = str(output_dir / "best_admitted.pth")
    elif args.variant in ("hr1o", "hr1s") and best_admitted is None:
        decision = f"STOP_P3_{args.variant.upper()}_SCREEN_KEEP_P3_DO_NOT_TEST"
        selected = str(args.p3_checkpoint)
    elif args.variant in ("hr1o", "hr1s"):
        decision = f"ADMIT_P3_{args.variant.upper()}_ISOLATED_COMPONENT_NO_TEST"
        selected = str(output_dir / "best_admitted.pth")
    elif args.variant == "hr2" and best_admitted is None:
        decision = "STOP_P3_HR2_SCREEN_KEEP_P3_DO_NOT_TEST"
        selected = str(args.p3_checkpoint)
    elif args.variant == "hr2":
        decision = "ADMIT_P3_HR2_TO_1500_STEP_MULTISEED_NO_TEST"
        selected = str(output_dir / "best_admitted.pth")
    elif best_admitted is None:
        decision = "STOP_P3_HR3O_REGION_OCCURRENCE_KEEP_P3_DO_NOT_TEST"
        selected = str(args.p3_checkpoint)
    else:
        decision = "ADMIT_P3_HR3O_TO_1500_STEP_MULTISEED_NO_TEST"
        selected = str(output_dir / "best_admitted.pth")
    summary = {
        "protocol": "P3_HR_hierarchical_residual_screen_v2",
        "variant": args.variant,
        "baseline": baseline,
        "history": history,
        "best_admitted": best_admitted,
        "selected_checkpoint": selected,
        "successful_steps": successful_step,
        "consumed_batches": consumed_batches,
        "skipped_batches": skipped_batches,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "p3_frozen": True,
        "decision": decision,
        "inference_uses_memory": False,
        "validation_accessed": True,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Decision: {decision}")
    print(f"Selected: {selected}")
    print(f"Saved: {output_dir}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
