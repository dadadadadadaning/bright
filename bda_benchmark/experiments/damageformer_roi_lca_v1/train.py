#!/usr/bin/env python3
"""Fast D0-initialised feasibility training for A2--A5 ROI-LCA variants."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.model.DamageFormer_ROI_LCA import DamageFormerROILCA, LCA_VARIANTS
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import (
    best_weights_payload, checkpoint_payload, evaluate, metrics_line, segmentation_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state(payload) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(payload.get(key), Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise TypeError("D0 checkpoint has no model state")
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in payload.items() if isinstance(value, torch.Tensor)
    }


def load_d0(model: DamageFormerROILCA, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint_state(payload), strict=False)
    bad_missing = [key for key in missing if not key.startswith(model.plugin_prefix)]
    if bad_missing or unexpected:
        raise RuntimeError(f"D0 mismatch: missing={bad_missing[:10]} unexpected={unexpected[:10]}")
    return {
        "path": str(Path(path).resolve()),
        "step": int(payload.get("step", -1)) if isinstance(payload, Mapping) else -1,
        "valid_miou": (payload.get("best_metrics") or {}).get("valid_miou")
        if isinstance(payload, Mapping) else None,
        "plugin_missing_keys": len(missing),
    }


def shift_zero(value: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    result = torch.zeros_like(value)
    height, width = value.shape[-2:]
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    result[..., target_y, target_x] = value[..., source_y, source_x]
    return result


def synthetic_shift(
    value: torch.Tensor, probability: float, max_pixels: int
) -> Tuple[torch.Tensor, int, int]:
    if random.random() >= probability:
        return value, 0, 0
    magnitudes = [4] if max_pixels < 8 else [4, 8]
    amount = random.choice((-1, 1)) * random.choice(magnitudes)
    if random.random() < 0.5:
        return shift_zero(value, amount, 0), amount, 0
    return shift_zero(value, 0, amount), 0, amount


def offset_loss(outputs: Mapping[str, torch.Tensor], dx: int, dy: int) -> torch.Tensor:
    offset = outputs["alignment_offset"].float()
    roi = outputs["alignment_roi"].detach().float()
    # Original DamageFormer fuses at 1/4 input resolution.
    target = torch.empty_like(offset)
    target[:, 0].fill_(dx / 4.0)
    target[:, 1].fill_(dy / 4.0)
    error = F.smooth_l1_loss(offset, target, reduction="none")
    return (error * roi).sum() / (roi.sum().clamp_min(1.0) * 2.0)


def atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser("DamageFormer ROI-LCA fast feasibility")
    parser.add_argument("--variant", choices=tuple(LCA_VARIANTS), required=True)
    parser.add_argument("--d0_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--optimizer_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--shift_probability", type=float, default=0.50)
    parser.add_argument("--offset_loss_weight", type=float, default=0.05)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip_initial_eval", action="store_true")
    args = parser.parse_args()
    if any("test" in Path(path).name.lower() for path in (args.train_list, args.val_list)):
        raise RuntimeError("ROI-LCA training forbids Test lists")
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = DamageFormerROILCA(args.variant, pretrained=False).to(device)
    parent_report = load_d0(model, args.d0_checkpoint)
    model.freeze_parent()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp == "fp16")
    start_step, best_metrics = 0, {}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("variant") != args.variant:
            raise RuntimeError("resume variant mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if scaler.is_enabled() and saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"])
        best_metrics = dict(saved.get("best_metrics", {}))

    effective_batch = args.batch_size * args.accumulation_steps
    total_crops = max(args.batch_size, args.optimizer_steps * effective_batch)
    train_set = HeterogeneousRGBSARDataset(
        args.dataset_path, args.train_list, 640, max_iters=total_crops, split="train", variant="h0"
    )
    val_set = HeterogeneousRGBSARDataset(
        args.dataset_path, args.val_list, 640, split="val", variant="h0"
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        **vars(args), "variant_contract": model.checkpoint_contract(),
        "parent": parent_report, "effective_batch_size": effective_batch,
        "trainable_parameters": sum(p.numel() for p in trainable), "test_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")

    print("=" * 88, flush=True)
    print(f"Variant         : {args.variant} | {LCA_VARIANTS[args.variant].title}", flush=True)
    print(f"D0 parent       : {args.d0_checkpoint}", flush=True)
    print(f"Steps/EBS       : {args.optimizer_steps}/{effective_batch}", flush=True)
    print(f"Shift/offset    : p={args.shift_probability} / w={args.offset_loss_weight}", flush=True)
    print(f"Trainable       : {sum(p.numel() for p in trainable):,}", flush=True)
    print("Test access     : FORBIDDEN", flush=True)
    print("=" * 88, flush=True)
    if not args.skip_initial_eval:
        print(metrics_line("Initial D0 reproduction", evaluate(
            model, val_loader, device, args.ignore_index, args.val_max_batches
        )), flush=True)

    best_value = float(best_metrics.get("valid_miou", -1.0))
    best_step = start_step if best_metrics else None
    step = start_step
    micro = 0
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    running: Dict[str, float] = {}
    window_start = time.time()
    while step < args.optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        rgb, sar, location, damage = [value.to(device, non_blocking=True) for value in batch[:4]]
        rgb, sar = rgb.float(), sar.float()
        location, damage = location.long(), damage.long()
        maximum_train_shift = 4 * (LCA_VARIANTS[args.variant].window_size // 2)
        shifted_sar, dx, dy = synthetic_shift(
            sar, args.shift_probability, maximum_train_shift
        )
        model.set_plugin_training()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(rgb, shifted_sar, return_dict=True)
            damage_loss, damage_ce, damage_lovasz = segmentation_loss(
                outputs["output_dam"], damage, args.ignore_index, 1.0, 1.0
            )
            alignment_loss = offset_loss(outputs, dx, dy)
            total = damage_loss + args.offset_loss_weight * alignment_loss
        scaler.scale(total / args.accumulation_steps).backward()
        micro += 1
        for key, value in {
            "total_loss": total, "damage_loss": damage_loss, "damage_ce": damage_ce,
            "damage_lovasz": damage_lovasz, "offset_loss": alignment_loss,
            "alignment_gain": torch.tanh(model.roi_lca.gain),
            "mean_confidence": outputs["alignment_confidence"].mean(),
        }.items():
            running[key] = running.get(key, 0.0) + float(value.detach())
        if micro % args.accumulation_steps:
            continue
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            denominator = args.log_every * args.accumulation_steps
            row = {"step": step, **{k: v / denominator for k, v in running.items()},
                   "last_shift": {"dx": dx, "dy": dy},
                   "optimizer_steps_per_second": args.log_every / max(time.time() - window_start, 1e-6)}
            print(json.dumps(row), flush=True)
            running.clear(); window_start = time.time()
        checkpoint = checkpoint_payload(model, optimizer, scaler, step, best_metrics, args)
        checkpoint["parent_report"] = parent_report
        if step % args.save_every == 0:
            atomic_save(checkpoint, output_dir / "latest_resume.pth")
        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics = evaluate(model, val_loader, device, args.ignore_index, args.val_max_batches)
            print(metrics_line("Clean validation", metrics), flush=True)
            with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "metrics": metrics}) + "\n")
            if float(metrics["valid_miou"]) > best_value:
                best_value = float(metrics["valid_miou"]); best_metrics = dict(metrics); best_step = step
                atomic_save(best_weights_payload(model, step, best_metrics, args), output_dir / "best_val_miou.pth")
                print(f"New best valid_mIoU={100*best_value:.4f}", flush=True)
            checkpoint["best_metrics"] = best_metrics
            atomic_save(checkpoint, output_dir / "latest_resume.pth")
    (output_dir / "training_summary.json").write_text(json.dumps({
        "protocol": "DAMAGEFORMER_ROI_LCA_V1", "variant": args.variant,
        "completed_step": step, "best_step": best_step, "best_valid_miou": best_value,
        "best_metrics": best_metrics, "parent": parent_report, "test_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Training complete: step={step} best={100*best_value:.4f}", flush=True)


if __name__ == "__main__":
    main()
