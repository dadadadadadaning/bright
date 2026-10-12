#!/usr/bin/env python3
"""Train A0-A3 with one parent checkpoint and one fixed protocol."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
BDA = ROOT / "bda_benchmark"
for value in (str(ROOT), str(BDA), str(HERE), str(BDA / "script/standard_ML")):
    if value not in sys.path:
        sys.path.insert(0, value)

from core import (  # noqa: E402
    atomic_torch_save, augment_metrics, bag_losses, event_name, get_variant,
    load_parent, parse_batch, seed_everything, seed_worker, sha256_file, variant_manifest,
    write_json,
)
from dataset.make_data_loader import MultimodalDamageAssessmentDatset  # noqa: E402
from model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL, build_training_bag_specs,
)
from train_S5_S8_GPU import fast_hist, metrics_from_hist, segmentation_loss  # noqa: E402


def read_list(path: str) -> List[str]:
    rows = [row.strip() for row in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        raise RuntimeError(f"empty data list: {path}")
    return rows


def select_bags(location, damage, min_area: int, max_per_class: int):
    bags = build_training_bag_specs(location, damage, min_area=min_area)
    if max_per_class <= 0:
        return bags
    selected = []
    for label in (1, 2, 3):
        members = [bag for bag in bags if bag.label == label]
        if len(members) > max_per_class:
            members = random.sample(members, max_per_class)
        selected.extend(members)
    return selected


@torch.no_grad()
def evaluate(model, loader, device, ignore_index: int, max_batches: int = 0):
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    dam_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hists: Dict[str, torch.Tensor] = {}
    processed = 0
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        pre, post, location, damage, names = parse_batch(batch, device)
        output_loc, output_damage = model(pre, post)
        pred_loc, pred_damage = output_loc.argmax(1), output_damage.argmax(1)
        loc_hist += fast_hist(pred_loc, location, 2, ignore_index)
        dam_hist += fast_hist(pred_damage, damage, 4, ignore_index)
        for sample_index, name in enumerate(names):
            key = event_name(str(name))
            hist = fast_hist(
                pred_damage[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1], 4, ignore_index,
            )
            event_hists[key] = event_hists.get(key, torch.zeros_like(hist)) + hist
        processed += int(pre.shape[0])
    metrics = augment_metrics(metrics_from_hist(loc_hist, dam_hist), loc_hist, dam_hist)
    per_event = {}
    for key, hist in event_hists.items():
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - intersection
        per_event[key] = float((intersection / union.clamp_min(1.0)).mean().item())
    metrics["event_count"] = len(per_event)
    metrics["event_mean_miou4"] = sum(per_event.values()) / max(1, len(per_event))
    metrics["event_worst_miou4"] = min(per_event.values()) if per_event else 0.0
    metrics["event_miou4_by_name"] = per_event
    metrics["processed_images"] = processed
    return metrics


def format_metrics(metrics: Mapping[str, float]) -> str:
    return (
        f"valid_mIoU={100*metrics['valid_miou']:.4f} | "
        f"mIoU4={100*metrics['miou4']:.4f} | "
        f"D-F1={100*metrics['damaged_f1']:.4f} | "
        f"IoU(I/D/X)={100*metrics['intact_iou']:.2f}/"
        f"{100*metrics['damaged_iou']:.2f}/{100*metrics['destroyed_iou']:.2f} | "
        f"events mean/worst={100*metrics['event_mean_miou4']:.2f}/"
        f"{100*metrics['event_worst_miou4']:.2f}"
    )


def configure_trainable(model, use_bags: bool):
    for name, parameter in model.named_parameters():
        # P3 inherits the old S3 prototype module for checkpoint compatibility,
        # but its forward path explicitly bypasses it.  Never count or optimise
        # those unused parameters in this ablation.
        if name.startswith("damage_proto_memory."):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(use_bags or not name.startswith("micb_dsmil."))
    core = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("micb_dsmil.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("micb_dsmil.")]
    return core, head


def main():
    parser = argparse.ArgumentParser("Controlled DamageFormer-to-P3 ablation")
    parser.add_argument("--variant", required=True, choices=("a0", "a1", "a2", "a3"))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--parent_kind", choices=("damageformer", "s3v2"), default="damageformer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--optimizer_steps", type=int, default=5000)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--core_lr", type=float, default=1e-5)
    parser.add_argument("--bag_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--lambda_mil", type=float, default=0.25)
    parser.add_argument("--lambda_exist", type=float, default=0.25)
    parser.add_argument("--lambda_severity", type=float, default=0.25)
    parser.add_argument("--lambda_guard", type=float, default=0.05)
    parser.add_argument("--min_instance_area", type=int, default=32)
    parser.add_argument("--max_bags_per_class", type=int, default=16)
    parser.add_argument("--mil_warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    spec = get_variant(args.variant)
    if "test" in Path(args.train_list).name.lower() or "test" in Path(args.val_list).name.lower():
        raise RuntimeError("train/validation ablation refuses any list whose filename contains 'test'")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_summary.json").exists() and not args.resume:
        raise RuntimeError(f"completed output exists; refusing to overwrite: {output_dir}")
    if not Path(args.parent_ckpt).is_file():
        raise FileNotFoundError(f"missing common parent checkpoint: {args.parent_ckpt}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = seed_everything(args.seed, args.deterministic)
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")

    model = DamageFormer_S8_MICB_DSMIL(
        roi_alpha=spec.roi_alpha, detach_building_prior=True,
        token_dim=128, mil_topk=8, max_tokens_per_bag=256,
        mil_residual_scale=0.0,
    ).to(device)
    parent_report = load_parent(model, args.parent_ckpt)
    core_params, bag_params = configure_trainable(model, spec.use_bags)
    groups = [{"params": core_params, "lr": args.core_lr, "name": "shared_pixel_path"}]
    if bag_params:
        groups.append({"params": bag_params, "lr": args.bag_lr, "name": "micb_dsmil"})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    start_step, best_metrics = 0, {}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("variant") != args.variant:
            raise RuntimeError("resume checkpoint variant mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if scaler.is_enabled() and "scaler" in saved:
            scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"])
        best_metrics = dict(saved.get("best_metrics", {}))

    train_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.train_list), args.crop_size, type="train"
    )
    val_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), args.crop_size, type="val"
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
    )
    if not len(train_loader):
        raise RuntimeError("empty training loader")
    config = vars(args) | {
        "variant_spec": spec.__dict__, "variant_manifest": variant_manifest(),
        "train_list_sha256": sha256_file(args.train_list),
        "val_list_sha256": sha256_file(args.val_list),
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "test_accessed": False,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "parent_checkpoint.json", parent_report)
    loc_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    dam_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    print("=" * 80, flush=True)
    print(f"Variant          : {args.variant} | {spec.title}", flush=True)
    print(f"Parent checkpoint: {args.parent_ckpt}", flush=True)
    print(f"Parent kind      : {args.parent_kind}", flush=True)
    print(f"ROI / bags / hierarchy: {spec.roi_alpha} / {spec.use_bags} / {spec.use_hierarchy}", flush=True)
    print(f"Steps / EBS      : {args.optimizer_steps} / {args.batch_size * args.accumulation_steps}", flush=True)
    print("TEST             : NEVER", flush=True)
    print("=" * 80, flush=True)

    history_path = output_dir / "history.jsonl"
    best_value = float(best_metrics.get("valid_miou", -1.0))
    step, micro, epoch = start_step, 0, 0
    optimizer.zero_grad(set_to_none=True)
    last_time = time.time()
    while step < args.optimizer_steps:
        for batch in train_loader:
            model.train()
            pre, post, location, damage, _ = parse_batch(batch, device)
            bags = select_bags(
                location, damage, args.min_instance_area, args.max_bags_per_class
            ) if spec.use_bags else []
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                output_loc, output_damage, aux = model(
                    pre, post, bag_specs=bags, return_aux=True
                )
            loss_loc, loc_ce_value, loc_lv = segmentation_loss(
                output_loc, location, loc_ce, True, 1.0, 1.0, args.ignore_index
            )
            loss_damage, dam_ce_value, dam_lv = segmentation_loss(
                output_damage, damage, dam_ce, True, 1.0, 1.0, args.ignore_index
            )
            components = bag_losses(aux["bag_result"], output_damage)
            bag_scale = float(spec.use_bags and (step + 1 > args.mil_warmup_steps))
            auxiliary = args.lambda_mil * components["mil"]
            if spec.use_hierarchy:
                auxiliary = auxiliary + (
                    args.lambda_exist * components["exist"]
                    + args.lambda_severity * components["severity"]
                    + args.lambda_guard * components["guard"]
                )
            total = loss_loc + loss_damage + bag_scale * auxiliary
            scaled = total / args.accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            micro += 1
            if micro < args.accumulation_steps:
                continue
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if step % args.log_every == 0:
                elapsed = max(time.time() - last_time, 1e-6)
                row = {
                    "step": step, "total_loss": float(total.detach()),
                    "loc_loss": float(loss_loc.detach()), "damage_loss": float(loss_damage.detach()),
                    "mil_loss": float(components["mil"].detach()),
                    "exist_loss": float(components["exist"].detach()),
                    "severity_loss": float(components["severity"].detach()),
                    "guard_loss": float(components["guard"].detach()),
                    "bag_count": int(aux["bag_count"]), "bag_scale": bag_scale,
                    "optimizer_steps_per_second": args.log_every / elapsed,
                }
                print(json.dumps(row), flush=True)
                last_time = time.time()
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "variant": args.variant, "best_metrics": best_metrics,
                "parent_checkpoint": parent_report, "args": vars(args),
            }
            if scaler.is_enabled():
                checkpoint["scaler"] = scaler.state_dict()
            if step % args.save_every == 0:
                atomic_torch_save(checkpoint, output_dir / "latest_resume.pth")
            if step % args.eval_every == 0 or step == args.optimizer_steps:
                metrics = evaluate(model, val_loader, device, args.ignore_index, args.val_max_batches)
                row = {"step": step, "metrics": metrics}
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("Validation | " + format_metrics(metrics), flush=True)
                checkpoint["best_metrics"] = best_metrics
                if float(metrics["valid_miou"]) > best_value:
                    best_value = float(metrics["valid_miou"])
                    best_metrics = dict(metrics)
                    checkpoint["best_metrics"] = best_metrics
                    atomic_torch_save(checkpoint, output_dir / "best_val_miou.pth")
                    print(f"New best at step={step}: {100*best_value:.4f}", flush=True)
                atomic_torch_save(checkpoint, output_dir / "latest_resume.pth")
            if step >= args.optimizer_steps:
                break
        epoch += 1

    final_payload = {
        "protocol": "P3_DAMAGEFORMER_ABLATION_V1", "variant": args.variant,
        "variant_spec": spec.__dict__, "completed_step": step,
        "best_step": None, "best_valid_miou": best_value,
        "best_metrics": best_metrics, "parent_checkpoint": parent_report,
        "parent_kind": args.parent_kind,
        "train_images": len(train_set), "val_images": len(val_set),
        "test_accessed": False,
    }
    if history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        if rows:
            final_payload["best_step"] = max(rows, key=lambda row: row["metrics"]["valid_miou"])["step"]
    write_json(output_dir / "training_summary.json", final_payload)
    print(f"Training complete | best valid_mIoU={100*best_value:.4f}", flush=True)


if __name__ == "__main__":
    main()
