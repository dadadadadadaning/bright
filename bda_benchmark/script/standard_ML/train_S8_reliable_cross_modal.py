#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the controlled S8-R0...R5 reliable cross-modal ablations.

The accepted S8 checkpoint is used only to migrate the common dual-Swin,
multi-scale fusion, localization head, and four-class damage head.  Every new
mechanism is isolated behind a variant switch and receives explicit auxiliary
supervision only when the variant enables it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.DamageFormer_S8_ReliableCrossModal import (  # noqa: E402
    DamageFormer_S8_ReliableCrossModal,
    get_variant_spec,
    load_s8_core_checkpoint,
)
from model.reliable_cross_modal import (  # noqa: E402
    build_hierarchical_targets,
    reliable_auxiliary_losses,
)
from train_S5_S8_GPU import (  # noqa: E402
    build_dataset,
    fast_hist,
    metrics_from_hist,
    parse_batch,
    segmentation_loss,
    set_random_seed,
    setup_runtime,
    seed_worker,
)


CORE_PREFIXES = (
    "encoder_1.",
    "encoder_2.",
    "fusion_layer_1.",
    "fusion_layer_2.",
    "fusion_layer_3.",
    "clf_1.",
    "clf_2.",
)


def _is_core_parameter(name: str) -> bool:
    return name.startswith(CORE_PREFIXES)


def set_core_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if _is_core_parameter(name):
            parameter.requires_grad_(bool(trainable))
    for name in (
        "encoder_1",
        "encoder_2",
        "fusion_layer_1",
        "fusion_layer_2",
        "fusion_layer_3",
        "clf_1",
        "clf_2",
    ):
        module = getattr(model, name)
        module.train(bool(trainable))


def build_optimizer(
    model: nn.Module,
    base_learning_rate: float,
    new_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    core_parameters = []
    new_parameters = []
    for name, parameter in model.named_parameters():
        if _is_core_parameter(name):
            core_parameters.append(parameter)
        else:
            new_parameters.append(parameter)
    groups = [
        {
            "params": core_parameters,
            "lr": float(base_learning_rate),
            "name": "s8_core",
        }
    ]
    if new_parameters:
        groups.append(
            {
                "params": new_parameters,
                "lr": float(new_learning_rate),
                "name": "reliable_new",
            }
        )
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def schedule_scale(step: int, warmup: int, ramp: int) -> float:
    if step <= warmup:
        return 0.0
    if ramp > 0 and step < warmup + ramp:
        return float(step - warmup) / float(ramp)
    return 1.0


def _binary_hist(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    if target.shape[-2:] != logits.shape[-2:]:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(),
            size=logits.shape[-2:],
            mode="nearest",
        )[:, 0].long()
    prediction = logits.argmax(dim=1)
    return fast_hist(prediction, target, 2, ignore_index)


def _iou_and_recall(hist: torch.Tensor) -> Tuple[float, float]:
    hist = hist.double()
    tp = hist[1, 1]
    union = hist[1].sum() + hist[:, 1].sum() - tp
    recall = tp / hist[1].sum().clamp_min(1.0)
    return float((tp / union.clamp_min(1.0)).item()), float(recall.item())


def _damage_harmonic_f1(hist: torch.Tensor) -> float:
    """Harmonic mean of Intact/Damaged/Destroyed pixel F1."""

    hist = hist.double()
    tp = torch.diag(hist)
    precision = tp / hist.sum(0).clamp_min(1.0)
    recall = tp / hist.sum(1).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    damage_f1 = f1[1:4]
    if bool((damage_f1 <= 0).any().item()):
        return 0.0
    return float((damage_f1.numel() / (1.0 / damage_f1).sum()).item())


@torch.no_grad()
def evaluate(
    model: DamageFormer_S8_ReliableCrossModal,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    dam_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    change_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    relation_change_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    exist_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    severity_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    diagnostic_sum: Dict[str, float] = {}
    diagnostic_count: Dict[str, int] = {}
    diagnostic_names = (
        "relation_s2_mean",
        "relation_s3_mean",
        "reliability_alpha",
        "reliability_multiplier_min",
        "reliability_multiplier_max",
        "rgb_quality_mean",
        "sar_quality_mean",
        "sar_adapter_delta_mean",
        "rectifier_alpha",
        "rectifier_safe_fraction",
        "rectifier_rgb_gate_mean",
        "rectifier_sar_gate_mean",
    )

    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        pre, post, loc, damage, _ = parse_batch(batch, device)
        outputs = model(pre, post, return_dict=True)
        loc_hist += fast_hist(outputs["output_loc"].argmax(1), loc, 2, ignore_index)
        dam_hist += fast_hist(outputs["output_dam"].argmax(1), damage, 4, ignore_index)
        exist_target, severity_target = build_hierarchical_targets(
            damage, ignore_index=ignore_index
        )
        if "change_logits_feature" in outputs:
            relation_change_hist += _binary_hist(
                outputs["change_logits_feature"], exist_target, ignore_index
            )
        if "change_attention_logits" in outputs:
            attention_logits = outputs["change_attention_logits"]
            attention_logits_2class = torch.cat(
                (torch.zeros_like(attention_logits), attention_logits), dim=1
            )
            change_hist += _binary_hist(
                attention_logits_2class, exist_target, ignore_index
            )
        elif "change_logits_feature" in outputs:
            change_hist += _binary_hist(
                outputs["change_logits_feature"], exist_target, ignore_index
            )
        if "pixel_exist_logits_feature" in outputs:
            exist_hist += _binary_hist(
                outputs["pixel_exist_logits_feature"], exist_target, ignore_index
            )
        if "pixel_severity_logits_feature" in outputs:
            severity_hist += _binary_hist(
                outputs["pixel_severity_logits_feature"], severity_target, ignore_index
            )
        for name in diagnostic_names:
            if name not in outputs:
                continue
            value = outputs[name]
            if isinstance(value, torch.Tensor):
                diagnostic_sum[name] = diagnostic_sum.get(name, 0.0) + float(
                    value.detach().float().mean().item()
                )
                diagnostic_count[name] = diagnostic_count.get(name, 0) + 1

    metrics = metrics_from_hist(loc_hist, dam_hist)
    metrics["damage_hmean_f1"] = _damage_harmonic_f1(dam_hist)
    if change_hist.sum() > 0:
        metrics["change_iou"], metrics["change_recall"] = _iou_and_recall(change_hist)
    if relation_change_hist.sum() > 0:
        (
            metrics["relation_change_iou"],
            metrics["relation_change_recall"],
        ) = _iou_and_recall(relation_change_hist)
    if exist_hist.sum() > 0:
        metrics["pixel_exist_iou"], metrics["pixel_exist_recall"] = _iou_and_recall(
            exist_hist
        )
    if severity_hist.sum() > 0:
        metrics["destroyed_vs_damaged_iou"], metrics["destroyed_recall"] = (
            _iou_and_recall(severity_hist)
        )
    for name, total in diagnostic_sum.items():
        metrics[name] = total / max(1, diagnostic_count[name])
    model.train()
    return metrics


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_metrics: Dict[str, float],
    args,
) -> None:
    payload: Dict[str, object] = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "variant": str(args.variant),
        "variant_spec": vars(get_variant_spec(args.variant)),
        "best_metrics": dict(best_metrics),
        "args": vars(args),
    }
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    _atomic_save(payload, path)


def resume_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    path: str,
) -> Tuple[int, Dict[str, float]]:
    if not path:
        return 0, {}
    checkpoint = torch.load(path, map_location="cpu")
    saved_variant = str(checkpoint.get("variant", "")).lower()
    model_variant = str(getattr(model, "variant", "")).lower()
    if saved_variant and saved_variant != model_variant:
        raise RuntimeError(
            f"resume variant mismatch: checkpoint={saved_variant}, model={model_variant}"
        )
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scaler is not None
        and scaler.is_enabled()
        and "scaler_state_dict" in checkpoint
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint.get("step", 0)), dict(checkpoint.get("best_metrics", {}))


def format_metrics(metrics: Mapping[str, float]) -> str:
    message = (
        f"valid_mIoU={100.0 * metrics['valid_miou']:.4f} | "
        f"IoU(int/dmg/des)="
        f"{100.0 * metrics['intact_iou']:.2f}/"
        f"{100.0 * metrics['damaged_iou']:.2f}/"
        f"{100.0 * metrics['destroyed_iou']:.2f} | "
        f"Dmg->Int={100.0 * metrics['damaged_to_intact']:.2f}% | "
        f"Des->Int={100.0 * metrics['destroyed_to_intact']:.2f}%"
        f" | Damage-HF1={100.0 * metrics['damage_hmean_f1']:.4f}"
    )
    for name in (
        "change_iou",
        "pixel_exist_iou",
        "destroyed_vs_damaged_iou",
        "reliability_alpha",
    ):
        if name in metrics:
            message += f" | {name}={metrics[name]:.5f}"
    return message


def main() -> None:
    parser = argparse.ArgumentParser("Train S8 reliable cross-modal ablations")
    parser.add_argument("--variant", choices=sorted(("r0", "r1", "r2", "r3", "r4", "r5")), default="r1")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--exp_name", default="")

    parser.add_argument("--image_size", "--crop_size", dest="image_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--optimizer_steps", "--max_steps", dest="optimizer_steps", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--base_learning_rate", type=float, default=1e-5)
    parser.add_argument("--new_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_only_steps", type=int, default=500)
    parser.add_argument("--relation_projection_channels", type=int, default=32)
    parser.add_argument("--relation_channels", type=int, default=64)
    parser.add_argument("--gate_alpha_max", type=float, default=0.1)
    parser.add_argument("--rectifier_alpha_max", type=float, default=0.05)
    parser.add_argument("--roi_alpha", type=float, default=0.1)

    parser.add_argument("--lambda_change", type=float, default=0.1)
    parser.add_argument("--lambda_exist", type=float, default=0.1)
    parser.add_argument("--lambda_severity", type=float, default=0.1)
    parser.add_argument("--aux_warmup_steps", type=int, default=0)
    parser.add_argument("--aux_ramp_steps", type=int, default=500)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--use_lovasz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument(
        "--expected_base_val_miou",
        type=float,
        default=-1.0,
        help="Expected full-validation baseline in percent; negative disables the guard.",
    )
    parser.add_argument("--base_miou_tolerance", type=float, default=0.5)
    args = parser.parse_args()
    if args.variant == "r0" and args.head_only_steps > 0:
        print("R0 has no new modules; head_only_steps is forced to 0.", flush=True)
        args.head_only_steps = 0

    device, distributed, rank, world_size, _ = setup_runtime(args.device)
    if distributed or world_size != 1 or rank != 0:
        raise RuntimeError(
            "This controlled ablation trainer intentionally supports one GPU. "
            "Use CUDA_VISIBLE_DEVICES to select the card."
        )
    generator = set_random_seed(args.seed, 0, args.deterministic)
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and args.amp_dtype == "fp16"
    )

    model = DamageFormer_S8_ReliableCrossModal(
        variant=args.variant,
        relation_projection_channels=args.relation_projection_channels,
        relation_channels=args.relation_channels,
        gate_alpha_max=args.gate_alpha_max,
        rectifier_alpha_max=args.rectifier_alpha_max,
        roi_alpha=args.roi_alpha,
        use_pretrained_swin=False,
        return_dict=True,
    ).to(device)
    optimizer = build_optimizer(
        model,
        args.base_learning_rate,
        args.new_learning_rate,
        args.weight_decay,
    )

    if args.resume:
        start_step, best_metrics = resume_checkpoint(
            model, optimizer, scaler, args.resume
        )
        migration_report: Dict[str, object] = {"resume": args.resume}
    else:
        migration_report = load_s8_core_checkpoint(
            model, args.base_ckpt, strict_core=True
        )
        start_step, best_metrics = 0, {}

    train_set = build_dataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.image_size,
        "train",
    )
    val_set = build_dataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.image_size,
        "val",
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"S8_{args.variant.upper()}_seed{args.seed}_{timestamp}"
    output_dir = Path(args.output_root) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
    with open(output_dir / "migration_report.json", "w", encoding="utf-8") as file:
        json.dump(migration_report, file, ensure_ascii=False, indent=2)

    spec = get_variant_spec(args.variant)
    core_frozen = start_step < args.head_only_steps
    set_core_trainable(model, not core_frozen)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    runtime_metadata = {
        "total_parameters": int(total_parameters),
        "initial_trainable_parameters": int(trainable_parameters),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }
    with open(output_dir / "runtime_metadata.json", "w", encoding="utf-8") as file:
        json.dump(runtime_metadata, file, ensure_ascii=False, indent=2)
    print("=" * 80, flush=True)
    print(f"Experiment       : {spec.name}", flush=True)
    print(f"Variant switches : {vars(spec)}", flush=True)
    print(f"Base checkpoint  : {args.base_ckpt}", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print(
        f"LR core/new      : {args.base_learning_rate}/{args.new_learning_rate}",
        flush=True,
    )
    print(
        f"Optimizer steps  : {args.optimizer_steps}; accumulation={args.accumulation_steps}",
        flush=True,
    )
    print(
        f"Parameters       : total={total_parameters:,}; "
        f"initial_trainable={trainable_parameters:,}",
        flush=True,
    )
    print("=" * 80, flush=True)

    loc_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    damage_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    best_valid_miou = float(best_metrics.get("valid_miou", -1.0))

    if start_step == 0 and not args.skip_initial_eval:
        initial_metrics = evaluate(
            model, val_loader, device, args.ignore_index, args.val_max_batches
        )
        print("Initial validation | " + format_metrics(initial_metrics), flush=True)
        if (
            args.expected_base_val_miou >= 0.0
            and args.val_max_batches == 0
            and abs(
                100.0 * initial_metrics["valid_miou"]
                - args.expected_base_val_miou
            )
            > args.base_miou_tolerance
        ):
            raise RuntimeError(
                "Initial full-validation baseline mismatch: "
                f"observed={100.0 * initial_metrics['valid_miou']:.4f}, "
                f"expected={args.expected_base_val_miou:.4f} +/- "
                f"{args.base_miou_tolerance:.4f}. "
                "Stop before training because the checkpoint and model path "
                "are not semantically compatible."
            )
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {"step": 0, "phase": "initial", **initial_metrics},
                    ensure_ascii=False,
                )
                + "\n"
            )
        if initial_metrics["valid_miou"] > best_valid_miou:
            best_valid_miou = initial_metrics["valid_miou"]
            best_metrics = dict(initial_metrics)
            save_checkpoint(
                output_dir / "best_val_miou.pth",
                model,
                optimizer,
                scaler,
                0,
                best_metrics,
                args,
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    log_window_start = time.perf_counter()
    log_window_start_step = step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < args.optimizer_steps:
        for batch in train_loader:
            if core_frozen:
                # evaluate() restores model.train(); re-lock BatchNorm and
                # encoder stochastic layers during the head-only phase.
                set_core_trainable(model, False)
            if core_frozen and step >= args.head_only_steps:
                set_core_trainable(model, True)
                core_frozen = False
                print(f"Unfroze S8 core at optimizer step {step}.", flush=True)

            pre, post, loc, damage, _ = parse_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(pre, post, return_dict=True)
            loss_loc, _, _ = segmentation_loss(
                outputs["output_loc"],
                loc,
                loc_ce,
                args.use_lovasz,
                args.ce_weight,
                args.lovasz_weight,
                args.ignore_index,
            )
            loss_damage, _, _ = segmentation_loss(
                outputs["output_dam"],
                damage,
                damage_ce,
                args.use_lovasz,
                args.ce_weight,
                args.lovasz_weight,
                args.ignore_index,
            )
            auxiliary = reliable_auxiliary_losses(
                outputs,
                damage,
                lambda_change=args.lambda_change,
                lambda_exist=args.lambda_exist,
                lambda_severity=args.lambda_severity,
                ignore_index=args.ignore_index,
            )
            aux_scale = schedule_scale(
                step + 1, args.aux_warmup_steps, args.aux_ramp_steps
            )
            total_loss = (
                loss_loc + loss_damage + aux_scale * auxiliary["loss_aux_total"]
            )
            if not bool(torch.isfinite(total_loss.detach()).item()):
                raise FloatingPointError(
                    f"non-finite loss at step={step}: "
                    f"loc={loss_loc.item()}, damage={loss_damage.item()}, "
                    f"aux={auxiliary['loss_aux_total'].item()}"
                )
            scaled_loss = total_loss / float(args.accumulation_steps)
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
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
                message = {
                    "step": step,
                    "loss": float(total_loss.detach().item()),
                    "loss_loc": float(loss_loc.detach().item()),
                    "loss_damage": float(loss_damage.detach().item()),
                    "loss_change": float(auxiliary["loss_change"].detach().item()),
                    "loss_change_relation": float(
                        auxiliary["loss_change_relation"].detach().item()
                    ),
                    "loss_change_attention": float(
                        auxiliary["loss_change_attention"].detach().item()
                    ),
                    "loss_pixel_exist": float(
                        auxiliary["loss_pixel_exist"].detach().item()
                    ),
                    "loss_pixel_severity": float(
                        auxiliary["loss_pixel_severity"].detach().item()
                    ),
                    "aux_scale": aux_scale,
                    "core_frozen": core_frozen,
                }
                elapsed = max(time.perf_counter() - log_window_start, 1e-6)
                completed = max(step - log_window_start_step, 1)
                message["optimizer_steps_per_second"] = completed / elapsed
                message["seconds_per_optimizer_step"] = elapsed / completed
                if device.type == "cuda":
                    message["peak_memory_gib"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                for name in (
                    "relation_s2_mean",
                    "relation_s3_mean",
                    "reliability_alpha",
                    "rectifier_alpha",
                    "sar_adapter_delta_mean",
                ):
                    value = outputs.get(name)
                    if isinstance(value, torch.Tensor):
                        message[name] = float(value.detach().float().mean().item())
                print(json.dumps(message, ensure_ascii=False), flush=True)
                log_window_start = time.perf_counter()
                log_window_start_step = step
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

            do_eval = step % args.eval_every == 0 or step == args.optimizer_steps
            if do_eval:
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    args.ignore_index,
                    args.val_max_batches,
                )
                print("Validation | " + format_metrics(metrics), flush=True)
                with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as file:
                    file.write(
                        json.dumps({"step": step, **metrics}, ensure_ascii=False)
                        + "\n"
                    )
                if metrics["valid_miou"] > best_valid_miou:
                    best_valid_miou = metrics["valid_miou"]
                    best_metrics = dict(metrics)
                    save_checkpoint(
                        output_dir / "best_val_miou.pth",
                        model,
                        optimizer,
                        scaler,
                        step,
                        best_metrics,
                        args,
                    )
                    print(
                        f"New best valid_mIoU={100.0 * best_valid_miou:.4f}",
                        flush=True,
                    )

            if step % args.save_every == 0:
                save_checkpoint(
                    output_dir / "latest_resume.pth",
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_metrics,
                    args,
                )
            if step >= args.optimizer_steps:
                break

    save_checkpoint(
        output_dir / "final.pth",
        model,
        optimizer,
        scaler,
        step,
        best_metrics,
        args,
    )
    print(f"Training complete. Best valid_mIoU={100.0 * best_valid_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
