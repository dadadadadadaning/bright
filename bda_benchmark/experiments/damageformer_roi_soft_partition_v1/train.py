#!/usr/bin/env python3
"""Train only ROI + the fixed designed soft partition on Train/Validation.

Parent A1 is loaded read-only and frozen. Only ``cmsp.*`` parameters train.
Validation is run every 500 optimizer steps by default. Test is forbidden.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import segmentation_loss

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.core import (
    atomic_torch_save,
    extract_state,
    load_a1_parent,
    parse_batch,
    seed_everything,
    seed_worker,
    write_json,
)
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.losses import region_losses
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.train import (
    evaluate,
    format_metrics,
    read_list,
)
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.variants import get_variant

from .model import DamageFormerROISoftPartition, FIXED_VARIANT


def main() -> None:
    parser = argparse.ArgumentParser("DamageFormer ROI + Cross-modal Damage-aware Soft Partition V1")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--optimizer_steps", type=int, default=10000)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--plugin_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--region_stride", type=int, default=8)
    args = parser.parse_args()

    for value in (args.train_list, args.val_list):
        if "test" in Path(value).name.lower():
            raise RuntimeError("TEST ACCESS IS FORBIDDEN IN THIS PROTOCOL")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_summary.json").exists() and not args.resume:
        raise RuntimeError(f"completed output exists: {output_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = seed_everything(args.seed)

    spec = get_variant(FIXED_VARIANT)
    contract_ok = (
        spec.use_lca
        and spec.partition == "learnable"
        and spec.sar_source == "aligned"
        and spec.use_alignment_confidence
        and not spec.shuffle_sar
        and not spec.use_damage_hint
        and not spec.use_boundary_loss
    )
    if not contract_ok:
        raise RuntimeError(f"fixed partition contract changed unexpectedly: {spec}")

    model = DamageFormerROISoftPartition(region_stride=args.region_stride).to(device)
    parent_report = load_a1_parent(model, args.parent_ckpt)

    model.freeze_parent()
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise RuntimeError("no trainable parameters")
    bad = [name for name in trainable_names if not name.startswith("cmsp.")]
    if bad:
        raise RuntimeError("unexpected trainable parameter outside cmsp.*: " + ", ".join(bad))

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.plugin_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.optimizer_steps)
    )

    start_step, best_metrics = 0, {}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(extract_state(payload), strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        best_metrics = dict(payload.get("best_metrics", {}))

    train_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.train_list), args.crop_size, type="train"
    )
    val_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), args.crop_size, type="val"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
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
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    loc_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    dam_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and args.amp_dtype == "fp16"
    )

    print(json.dumps({
        "protocol": "DAMAGEFORMER_ROI_SOFT_PARTITION_V1",
        "top_level_modules": [
            "A1_soft_RGB_building_ROI",
            "cross_modal_damage_aware_soft_partition",
        ],
        "fixed_variant": spec.to_dict(),
        "parent": parent_report,
        "optimizer_steps": args.optimizer_steps,
        "eval_every": args.eval_every,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "trainable_prefix_ok": True,
        "test_accessed": False,
    }, ensure_ascii=False, indent=2), flush=True)

    best_value = float(best_metrics.get("valid_miou", -1.0))
    step, micro = start_step, 0
    running: Dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    while step < args.optimizer_steps:
        for batch in train_loader:
            model.set_plugin_training()
            pre, post, location, damage, _ = parse_batch(batch, device)

            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                output_loc, output_damage, aux = model(pre, post, return_aux=True)

                loc_loss, _, _ = segmentation_loss(
                    output_loc, location, loc_ce, True, 1.0, 1.0, args.ignore_index
                )
                damage_loss, _, _ = segmentation_loss(
                    output_damage, damage, dam_ce, True, 1.0, 1.0, args.ignore_index
                )
                regions = region_losses(aux, location, use_boundary_loss=False)

                # ROI head is frozen. Final damage loss + partition regularisation train cmsp.*.
                total = damage_loss + regions["total"]
                scaled = total / args.accumulation_steps

            scaler.scale(scaled).backward()
            micro += 1

            values = {
                "total": total,
                "loc_audit": loc_loss,
                "damage": damage_loss,
                **{f"region_{key}": value for key, value in regions.items()},
            }
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + float(value.detach())

            if micro % args.accumulation_steps:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                denom = float(args.log_every * args.accumulation_steps)
                record = {
                    "step": step,
                    **{key: value / denom for key, value in running.items()},
                    "region_gain": float(torch.tanh(model.cmsp.region_gain).detach()),
                    "lca_gain": float(torch.tanh(model.cmsp.lca.gain).detach()),
                    "steps_per_second": step / max(1.0e-6, time.time() - start_time),
                }
                print(json.dumps(record), flush=True)
                with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                running.clear()

            should_eval = step % args.eval_every == 0 or step == args.optimizer_steps
            if should_eval:
                metrics = evaluate(
                    model, val_loader, device, args.ignore_index, args.val_max_batches
                )
                print("Validation | " + format_metrics(metrics), flush=True)

                payload = {
                    "protocol": "DAMAGEFORMER_ROI_SOFT_PARTITION_V1",
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metrics": best_metrics,
                    "contract": model.checkpoint_contract(),
                    "parent_report": parent_report,
                }
                atomic_torch_save(payload, output_dir / "latest.pth")

                if float(metrics["valid_miou"]) > best_value:
                    best_value = float(metrics["valid_miou"])
                    best_metrics = metrics
                    payload["best_metrics"] = best_metrics
                    atomic_torch_save(payload, output_dir / "best_val_miou.pth")

            if step % args.save_every == 0:
                atomic_torch_save({
                    "protocol": "DAMAGEFORMER_ROI_SOFT_PARTITION_V1",
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metrics": best_metrics,
                }, output_dir / f"step_{step:06d}.pth")

            if step >= args.optimizer_steps:
                break

    summary = {
        "protocol": "DAMAGEFORMER_ROI_SOFT_PARTITION_V1",
        "completed_steps": step,
        "best_metrics": best_metrics,
        "parent_report": parent_report,
        "fixed_variant": spec.to_dict(),
        "test_accessed": False,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
