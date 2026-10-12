#!/usr/bin/env python3
"""Stage-wise training for Building-Internal Change-Aware Partition V2."""
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
    atomic_torch_save,
    augment_metrics,
    event_name,
    load_a1_parent,
    load_v2_checkpoint,
    parse_batch,
    seed_everything,
    seed_worker,
    write_json,
)
from .losses import stage_losses, weak_building_bag_loss
from .model import DamageFormerBuildingInternalChangePartitionV2
from .variants import VARIANTS, get_variant


def read_list(path: str) -> List[str]:
    rows = [row.strip() for row in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        raise RuntimeError(f"empty list: {path}")
    return rows


def _region_count_stats(aux: Mapping[str, torch.Tensor]) -> tuple[float, float]:
    owners = aux["token_building_ids"]
    mass = aux["region_mass"]
    counts = []
    ratios = []
    for bi in range(owners.shape[0]):
        ids = torch.unique(owners[bi])
        ids = ids[ids > 0]
        for building_id in ids.tolist():
            idx = (owners[bi] == int(building_id)) & (mass[bi] > 0.5)
            counts.append(float(idx.sum().item()))
            if bool(idx.any().item()):
                ratios.extend(aux["region_area_ratio"][bi, idx].float().detach().cpu().tolist())
    return (
        sum(counts) / max(1, len(counts)),
        sum(ratios) / max(1, len(ratios)),
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    ignore_index: int,
    *,
    val_max_batches: int = 0,
    bag_aggregator: str = "logmeanexp",
    bag_temperature: float = 0.50,
    bag_topk: int = 3,
    bag_label_purity: float = 0.70,
):
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    dam_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hists: Dict[str, torch.Tensor] = {}
    processed = 0
    diag = {
        "confidence": 0.0,
        "align_entropy": 0.0,
        "assoc_entropy": 0.0,
        "change_alpha": 0.0,
        "change_gate": 0.0,
        "utility": 0.0,
        "region_count": 0.0,
        "area_ratio": 0.0,
        "bag_accuracy_num": 0.0,
        "bag_count": 0.0,
        "batches": 0.0,
    }
    for index, batch in enumerate(loader):
        if val_max_batches and index >= val_max_batches:
            break
        pre, post, location, damage, names = parse_batch(batch, device)
        output_loc, output_damage, aux = model(pre, post, return_aux=True)
        pred_loc, pred_damage = output_loc.argmax(1), output_damage.argmax(1)
        loc_hist += fast_hist(pred_loc, location, 2, ignore_index)
        dam_hist += fast_hist(pred_damage, damage, 4, ignore_index)
        for sample_index, name in enumerate(names):
            key = event_name(str(name))
            hist = fast_hist(
                pred_damage[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1],
                4,
                ignore_index,
            )
            event_hists[key] = event_hists.get(key, torch.zeros_like(hist)) + hist
        building = (aux["building_ids"] > 0).reshape(aux["building_ids"].shape[0], -1)
        assoc = aux["association_entropy"]
        if bool(building.any().item()):
            diag["assoc_entropy"] += float(assoc[building].mean())
        diag["confidence"] += float(aux["region_confidence"].mean())
        diag["align_entropy"] += float(aux["region_alignment_entropy"].mean())
        diag["change_alpha"] += float(aux["change_alpha"].abs().mean())
        diag["change_gate"] += float(aux["change_gate_probability"].mean())
        if model.stage == "feedback":
            hard = (aux["building_ids"] > 0).unsqueeze(1)
            if bool(hard.any().item()):
                diag["utility"] += float(aux["utility_gate"][hard].mean())
        rc, ar = _region_count_stats(aux)
        diag["region_count"] += rc
        diag["area_ratio"] += ar
        _, bag_stats = weak_building_bag_loss(
            aux,
            damage,
            ignore_index=ignore_index,
            label_purity_threshold=bag_label_purity,
            aggregator=bag_aggregator,
            bag_temperature=bag_temperature,
            bag_topk=bag_topk,
        )
        diag["bag_accuracy_num"] += bag_stats["bag_accuracy"] * bag_stats["bag_count"]
        diag["bag_count"] += bag_stats["bag_count"]
        diag["batches"] += 1.0
        processed += int(pre.shape[0])

    metrics = augment_metrics(metrics_from_hist(loc_hist, dam_hist), loc_hist, dam_hist)
    per_event = {}
    for key, hist in event_hists.items():
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - intersection
        per_event[key] = float((intersection / union.clamp_min(1.0)).mean())
    c = max(1.0, diag["batches"])
    metrics.update(
        {
            "event_count": len(per_event),
            "event_mean_miou4": sum(per_event.values()) / max(1, len(per_event)),
            "event_worst_miou4": min(per_event.values()) if per_event else 0.0,
            "event_miou4_by_name": per_event,
            "processed_images": processed,
            "alignment_confidence_mean": diag["confidence"] / c,
            "alignment_entropy_mean": diag["align_entropy"] / c,
            "association_entropy_mean": diag["assoc_entropy"] / c,
            "change_alpha_abs_mean": diag["change_alpha"] / c,
            "change_gate_mean": diag["change_gate"] / c,
            "utility_gate_mean": diag["utility"] / c,
            "regions_per_building_mean": diag["region_count"] / c,
            "region_area_ratio_mean": diag["area_ratio"] / c,
            "weak_bag_count": diag["bag_count"],
            "weak_bag_accuracy": diag["bag_accuracy_num"] / max(1.0, diag["bag_count"]),
            "change_gain": float(torch.tanh(model.biscp.change_gain).detach()),
            "feedback_gain": float(torch.tanh(model.biscp.feedback_gain).detach()),
        }
    )
    return metrics


def format_metrics(m: Mapping[str, float]) -> str:
    return (
        f"valid_mIoU={100*m['valid_miou']:.4f} | mIoU4={100*m['miou4']:.4f} | "
        f"D-F1={100*m['damaged_f1']:.4f} | IoU(I/D/X)="
        f"{100*m['intact_iou']:.2f}/{100*m['damaged_iou']:.2f}/{100*m['destroyed_iou']:.2f} | "
        f"events={100*m['event_mean_miou4']:.2f}/{100*m['event_worst_miou4']:.2f} | "
        f"bag={100*m['weak_bag_accuracy']:.2f}%({int(m['weak_bag_count'])}) | "
        f"alpha={m['change_alpha_abs_mean']:.5f} | fb={m['feedback_gain']:.5f}"
    )


def _checkpoint_payload(model, optimizer, scheduler, step, best_metrics, parent_report, args):
    return {
        "protocol": "DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2",
        "variant": args.variant,
        "stage": args.stage,
        "step": int(step),
        # Store only the new plugin; the frozen A1 parent remains the immutable external checkpoint.
        "plugin_model": model.biscp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metrics": best_metrics,
        "parent_report": parent_report,
        "contract": model.checkpoint_contract(),
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Building-Internal Change-Aware Partition V2")
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--stage", required=True, choices=("structure", "change", "feedback"))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--a1_ckpt", required=True)
    parser.add_argument("--init_ckpt", default="", help="Previous-stage V2 checkpoint; plugin weights only.")
    parser.add_argument("--resume", default="", help="Resume the same stage incl. optimizer/scheduler.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--optimizer_steps", type=int, default=3000)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--plugin_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--target_region_area", type=int, default=16)
    parser.add_argument("--min_region_area", type=int, default=4)
    parser.add_argument("--max_regions_per_building", type=int, default=16)
    parser.add_argument("--building_threshold", type=float, default=0.50)
    parser.add_argument("--min_building_area", type=int, default=8)
    parser.add_argument("--change_alpha_max", type=float, default=0.30)
    parser.add_argument("--peak_temperature", type=float, default=0.20)
    parser.add_argument("--bag_aggregator", choices=("logmeanexp", "topk"), default="logmeanexp")
    parser.add_argument("--bag_temperature", type=float, default=0.50)
    parser.add_argument("--bag_topk", type=int, default=3)
    parser.add_argument("--bag_label_purity", type=float, default=0.70)
    args = parser.parse_args()

    if any("test" in Path(value).name.lower() for value in (args.train_list, args.val_list)):
        raise RuntimeError("TEST ACCESS IS FORBIDDEN in this protocol")
    if args.init_ckpt and args.resume:
        raise ValueError("use either --init_ckpt or --resume, not both")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_summary.json").exists() and not args.resume:
        raise RuntimeError(f"completed output exists: {output_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = seed_everything(args.seed)
    spec = get_variant(args.variant)
    model = DamageFormerBuildingInternalChangePartitionV2(
        args.variant,
        target_region_area=args.target_region_area,
        min_region_area=args.min_region_area,
        max_regions_per_building=args.max_regions_per_building,
        building_threshold=args.building_threshold,
        min_building_area=args.min_building_area,
        change_alpha_max=args.change_alpha_max,
        peak_temperature=args.peak_temperature,
    ).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    init_report = None
    if args.init_ckpt:
        init_report = load_v2_checkpoint(model, args.init_ckpt)
    model.configure_stage(args.stage)

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable parameters after stage configuration")
    optimizer = torch.optim.AdamW(parameters, lr=args.plugin_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.optimizer_steps)
    )
    start_step, best_metrics = 0, {}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("stage") != args.stage or payload.get("variant") != args.variant:
            raise RuntimeError("resume checkpoint stage/variant mismatch")
        model.biscp.load_state_dict(payload["plugin_model"], strict=True)
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    print(
        json.dumps(
            {
                "protocol": "DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2",
                "variant": spec.to_dict(),
                "stage": args.stage,
                "parent": parent_report,
                "init": init_report,
                "steps": args.optimizer_steps,
                "trainable_parameter_count": sum(p.numel() for p in parameters),
                "trainable_preview": trainable_names[:80],
                "checkpoint_storage": "plugin-only + optimizer/scheduler; immutable A1 external",
                "gt_used_in_forward_partition": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.stage == "feedback":
        best_value = float(best_metrics.get("valid_miou", -1.0))
    elif args.stage == "change":
        best_value = float(best_metrics.get("weak_bag_accuracy", -1.0))
    else:
        best_value = float(start_step - 1)
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
                loc_loss, _, _ = segmentation_loss(
                    output_loc, location, loc_ce, True, 1.0, 1.0, args.ignore_index
                )
                damage_loss, _, _ = segmentation_loss(
                    output_damage, damage, dam_ce, True, 1.0, 1.0, args.ignore_index
                )
                extra = stage_losses(
                    aux,
                    damage,
                    stage=args.stage,
                    use_bag_supervision=spec.use_bag_supervision,
                    ignore_index=args.ignore_index,
                    bag_aggregator=args.bag_aggregator,
                    bag_temperature=args.bag_temperature,
                    bag_topk=args.bag_topk,
                    bag_label_purity=args.bag_label_purity,
                )
                if args.stage == "feedback":
                    total = damage_loss + extra["total"]
                else:
                    # A1 is frozen and feedback is OFF: pixel damage loss is an audit only.
                    total = extra["total"]
                scaled = total / args.accumulation_steps

            scaler.scale(scaled).backward()
            micro += 1
            record_values = {
                "total": total,
                "loc_audit": loc_loss,
                "damage_audit": damage_loss,
                **{
                    f"v2_{key}": value
                    for key, value in extra.items()
                    if torch.is_tensor(value)
                },
            }
            for key, value in record_values.items():
                running[key] = running.get(key, 0.0) + float(value.detach())
            running["bag_count"] = running.get("bag_count", 0.0) + float(extra.get("bag_count", 0.0))
            running["bag_accuracy_x_count"] = running.get("bag_accuracy_x_count", 0.0) + float(
                extra.get("bag_accuracy", 0.0)
            ) * float(extra.get("bag_count", 0.0))

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
                bag_count = running.get("bag_count", 0.0)
                record = {
                    "step": step,
                    **{
                        key: value / denom
                        for key, value in running.items()
                        if key not in ("bag_count", "bag_accuracy_x_count")
                    },
                    "bag_count_window": bag_count,
                    "bag_accuracy_window": running.get("bag_accuracy_x_count", 0.0) / max(1.0, bag_count),
                    "change_gain": float(torch.tanh(model.biscp.change_gain).detach()),
                    "feedback_gain": float(torch.tanh(model.biscp.feedback_gain).detach()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "steps_per_second": (step - start_step) / max(1.0e-6, time.time() - start_time),
                }
                print(json.dumps(record), flush=True)
                with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                running.clear()

            if step % args.eval_every == 0 or step == args.optimizer_steps:
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    args.ignore_index,
                    val_max_batches=args.val_max_batches,
                    bag_aggregator=args.bag_aggregator,
                    bag_temperature=args.bag_temperature,
                    bag_topk=args.bag_topk,
                    bag_label_purity=args.bag_label_purity,
                )
                print("Validation | " + format_metrics(metrics), flush=True)
                payload = _checkpoint_payload(
                    model, optimizer, scheduler, step, best_metrics, parent_report, args
                )
                atomic_torch_save(payload, output_dir / "latest.pth")
                # Stage-0/1 segmentation is exactly A1 by design, so choose best
                # by weak bag accuracy. Stage-2 returns to valid_mIoU.
                if args.stage == "feedback":
                    selection = float(metrics["valid_miou"])
                elif args.stage == "change":
                    selection = float(metrics["weak_bag_accuracy"])
                else:
                    selection = float(step)
                if selection > best_value:
                    best_value = selection
                    best_metrics = metrics
                    payload["best_metrics"] = best_metrics
                    atomic_torch_save(payload, output_dir / "best.pth")

            if step % args.save_every == 0:
                payload = _checkpoint_payload(
                    model, optimizer, scheduler, step, best_metrics, parent_report, args
                )
                atomic_torch_save(payload, output_dir / f"step_{step:06d}.pth")
            if step >= args.optimizer_steps:
                break

    summary = {
        "protocol": "DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2",
        "variant": spec.to_dict(),
        "stage": args.stage,
        "completed_steps": step,
        "best_metrics": best_metrics,
        "parent_report": parent_report,
        "init_report": init_report,
        "test_accessed": False,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
