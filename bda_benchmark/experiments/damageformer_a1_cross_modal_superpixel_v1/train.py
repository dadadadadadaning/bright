#!/usr/bin/env python3
"""Train one isolated cross-modal superpixel ablation on train/validation only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist, metrics_from_hist, segmentation_loss

from .core import (
    atomic_torch_save, augment_metrics, event_name, extract_state, load_a1_parent,
    parse_batch, seed_everything, seed_worker, write_json,
)
from .losses import region_losses
from .model import DamageFormerA1CrossModalSuperpixel
from .variants import VARIANTS, get_variant


def read_list(path: str) -> List[str]:
    rows = [row.strip() for row in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        raise RuntimeError(f"empty list: {path}")
    return rows


@torch.no_grad()
def evaluate(model, loader, device, ignore_index: int, max_batches: int = 0):
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    dam_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hists: Dict[str, torch.Tensor] = {}
    residual_sum = confidence_sum = entropy_sum = count = 0.0
    processed = 0
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        pre, post, location, damage, names = parse_batch(batch, device)
        output_loc, output_damage, aux = model(pre, post, return_aux=True)
        pred_loc, pred_damage = output_loc.argmax(1), output_damage.argmax(1)
        loc_hist += fast_hist(pred_loc, location, 2, ignore_index)
        dam_hist += fast_hist(pred_damage, damage, 4, ignore_index)
        for sample_index, name in enumerate(names):
            key = event_name(str(name))
            hist = fast_hist(pred_damage[sample_index:sample_index + 1], damage[sample_index:sample_index + 1], 4, ignore_index)
            event_hists[key] = event_hists.get(key, torch.zeros_like(hist)) + hist
        if "region_update" in aux:
            residual_sum += float((torch.tanh(aux["region_gain"]) * aux["region_update"]).abs().mean())
            confidence_sum += float(aux["region_confidence"].mean())
            entropy_sum += float(aux["alignment_entropy"].mean())
            count += 1.0
        processed += int(pre.shape[0])
    metrics = augment_metrics(metrics_from_hist(loc_hist, dam_hist), loc_hist, dam_hist)
    per_event = {}
    for key, hist in event_hists.items():
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - intersection
        per_event[key] = float((intersection / union.clamp_min(1.0)).mean())
    metrics.update({
        "event_count": len(per_event),
        "event_mean_miou4": sum(per_event.values()) / max(1, len(per_event)),
        "event_worst_miou4": min(per_event.values()) if per_event else 0.0,
        "event_miou4_by_name": per_event,
        "processed_images": processed,
        "region_residual_abs_mean": residual_sum / max(1.0, count),
        "alignment_confidence_mean": confidence_sum / max(1.0, count),
        "alignment_entropy_mean": entropy_sum / max(1.0, count),
    })
    return metrics


def format_metrics(metrics: Mapping[str, float]) -> str:
    return (
        f"valid_mIoU={100*metrics['valid_miou']:.4f} | mIoU4={100*metrics['miou4']:.4f} | "
        f"D-F1={100*metrics['damaged_f1']:.4f} | IoU(I/D/X)="
        f"{100*metrics['intact_iou']:.2f}/{100*metrics['damaged_iou']:.2f}/"
        f"{100*metrics['destroyed_iou']:.2f} | events mean/worst="
        f"{100*metrics['event_mean_miou4']:.2f}/{100*metrics['event_worst_miou4']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser("A1 cross-modal learnable-superpixel ablation")
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--optimizer_steps", type=int, default=5000)
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

    if any("test" in Path(value).name.lower() for value in (args.train_list, args.val_list)):
        raise RuntimeError("this protocol forbids every test split")
    if args.variant == "c0_a1" and args.optimizer_steps > 0:
        raise RuntimeError("c0_a1 is an evaluation-only exact parent reproduction")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_summary.json").exists() and not args.resume:
        raise RuntimeError(f"completed output exists: {output_dir}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = seed_everything(args.seed)
    spec = get_variant(args.variant)
    model = DamageFormerA1CrossModalSuperpixel(args.variant, args.region_stride).to(device)
    parent_report = load_a1_parent(model, args.parent_ckpt)
    model.freeze_parent()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.plugin_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.optimizer_steps))
    start_step, best_metrics = 0, {}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(extract_state(payload), strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        best_metrics = dict(payload.get("best_metrics", {}))

    train_set = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.train_list), args.crop_size, type="train")
    val_set = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.val_list), args.crop_size, type="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=seed_worker,
                              generator=generator, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                            persistent_workers=args.num_workers > 0)
    loc_ce, dam_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index), nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")
    print(json.dumps({"variant": spec.to_dict(), "parent": parent_report,
                      "steps": args.optimizer_steps, "test_accessed": False}, ensure_ascii=False, indent=2), flush=True)
    best_value = float(best_metrics.get("valid_miou", -1.0))
    step, micro = start_step, 0
    running: Dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    while step < args.optimizer_steps:
        for batch in train_loader:
            model.set_plugin_training()
            pre, post, location, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                output_loc, output_damage, aux = model(pre, post, return_aux=True)
                loc_loss, _, _ = segmentation_loss(output_loc, location, loc_ce, True, 1.0, 1.0, args.ignore_index)
                damage_loss, _, _ = segmentation_loss(output_damage, damage, dam_ce, True, 1.0, 1.0, args.ignore_index)
                regions = region_losses(aux, location, spec.use_boundary_loss)
                total = loc_loss + damage_loss + regions["total"]
                scaled = total / args.accumulation_steps
            scaler.scale(scaled).backward()
            micro += 1
            values = {"total": total, "loc": loc_loss, "damage": damage_loss,
                      **{f"region_{key}": value for key, value in regions.items()}}
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
                record = {"step": step, **{key: value / denom for key, value in running.items()},
                          "region_gain": float(torch.tanh(model.cmsp.region_gain).detach()),
                          "lca_gain": float(torch.tanh(model.cmsp.lca.gain).detach()),
                          "steps_per_second": step / max(1.0e-6, time.time() - start_time)}
                print(json.dumps(record), flush=True)
                with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                running.clear()
            should_eval = step % args.eval_every == 0 or step == args.optimizer_steps
            if should_eval:
                metrics = evaluate(model, val_loader, device, args.ignore_index, args.val_max_batches)
                print("Validation | " + format_metrics(metrics), flush=True)
                payload = {"protocol": "DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1", "variant": args.variant,
                           "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "best_metrics": best_metrics,
                           "contract": model.checkpoint_contract(), "parent_report": parent_report}
                atomic_torch_save(payload, output_dir / "latest.pth")
                if float(metrics["valid_miou"]) > best_value:
                    best_value, best_metrics = float(metrics["valid_miou"]), metrics
                    payload["best_metrics"] = best_metrics
                    atomic_torch_save(payload, output_dir / "best_val_miou.pth")
            if step % args.save_every == 0:
                atomic_torch_save({"variant": args.variant, "step": step, "model": model.state_dict(),
                                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                   "best_metrics": best_metrics}, output_dir / f"step_{step:06d}.pth")
            if step >= args.optimizer_steps:
                break
    summary = {"protocol": "DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1", "variant": spec.to_dict(),
               "completed_steps": step, "best_metrics": best_metrics, "parent_report": parent_report,
               "test_accessed": False}
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
