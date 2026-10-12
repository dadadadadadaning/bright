#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import segmentation_loss
from .core import (atomic_torch_save, load_frozen_sources, parse_batch, seed_everything,
                   seed_worker, verify_checkpoint_sources, write_json)
from .evaluation import evaluate, format_result
from .losses import readout_losses
from .model import DamageFormerRegionEvidenceReadout
from .variants import VARIANTS, get_variant


def read_list(path: str) -> List[str]:
    rows = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows: raise RuntimeError(f"empty list: {path}")
    return rows


def make_model(args, device):
    model = DamageFormerRegionEvidenceReadout(args.variant).to(device)
    sources = load_frozen_sources(model, args.a1_ckpt, args.stage1_ckpt,
                                  args.minimum_stage1_step, args.expected_stage1_sha256)
    model.configure_readout_training()
    return model, sources


def checkpoint(model, optimizer, scheduler, step, best, sources, args):
    return {"protocol": "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1", "variant": args.variant,
            "step": int(step), "readout_model": model.readout.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_metrics": best, "sources": sources, "contract": model.checkpoint_contract(),
            "test_accessed": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    p.add_argument("--dataset_path", required=True); p.add_argument("--train_list", required=True)
    p.add_argument("--val_list", required=True); p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--stage1_ckpt", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default=""); p.add_argument("--expected_stage1_sha256", default="")
    p.add_argument("--minimum_stage1_step", type=int, default=3000)
    p.add_argument("--optimizer_steps", type=int, default=2000); p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=20); p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=2); p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--accumulation_steps", type=int, default=8); p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    p.add_argument("--ignore_index", type=int, default=255); p.add_argument("--val_max_batches", type=int, default=0)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    if any("test" in Path(x).name.lower() for x in (args.train_list, args.val_list)):
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / "training_summary.json").exists() and not args.resume:
        raise RuntimeError(f"completed experiment exists: {out}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = seed_everything(args.seed)
    model, sources = make_model(args, device)
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.optimizer_steps))
    step, best = 0, {}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("variant") != args.variant: raise RuntimeError("resume variant mismatch")
        verify_checkpoint_sources(payload, sources)
        model.readout.load_state_dict(payload["readout_model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        step, best = int(payload["step"]), dict(payload.get("best_metrics", {}))

    train_set = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.train_list), args.crop_size, type="train")
    val_set = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.val_list), args.crop_size, type="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=seed_worker,
                              generator=generator, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                            persistent_workers=args.num_workers > 0)
    dam_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and args.amp_dtype == "fp16")
    print(json.dumps({"protocol": "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1",
                      "variant": get_variant(args.variant).to_dict(), "sources": sources,
                      "trainable_parameters": sum(x.numel() for x in params),
                      "step0_full_validation": args.val_max_batches == 0,
                      "test_accessed": False}, indent=2), flush=True)

    if step == 0:
        initial = evaluate(model, val_loader, device, args.ignore_index, args.val_max_batches)
        print("Step0 identity | " + format_result(initial), flush=True)
        if abs(initial["gain_valid_miou_pp"]) > 1e-7 or initial["metrics"]["changed_pixel_fraction"] > 0:
            raise RuntimeError("identity initialization failed")
        best = initial["metrics"]
        write_json(out / "step0_identity.json", {**initial, "sources": sources, "test_accessed": False})
        atomic_torch_save(checkpoint(model, optimizer, scheduler, 0, best, sources, args), out / "best_val_miou.pth")

    micro, running, started = 0, {}, time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < args.optimizer_steps:
        for batch in train_loader:
            model.set_readout_training()
            pre, post, _, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                _, logits, aux = model(pre, post, return_aux=True)
                seg, ce, lv = segmentation_loss(logits, damage, dam_ce, True, 1.0, 1.0, args.ignore_index)
                extras = readout_losses(args.variant, aux, damage, ignore_index=args.ignore_index)
                total = seg + extras["auxiliary"]
            scaler.scale(total / args.accumulation_steps).backward(); micro += 1
            for key, value in {"total": total, "seg": seg, "ce": ce, "lovasz": lv, **extras}.items():
                running[key] = running.get(key, 0.0) + float(value.detach())
            if micro % args.accumulation_steps: continue
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step += 1
            if step % args.log_every == 0:
                denom = args.log_every * args.accumulation_steps
                row = {"step": step, **{k: v / denom for k, v in running.items()},
                       "lr": optimizer.param_groups[0]["lr"],
                       "steps_per_second": step / max(1e-6, time.time() - started)}
                print(json.dumps(row), flush=True)
                with (out / "history.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
                running.clear()
            if step % args.eval_every == 0 or step == args.optimizer_steps:
                result = evaluate(model, val_loader, device, args.ignore_index, args.val_max_batches)
                print(f"Validation step={step} | {format_result(result)}", flush=True)
                write_json(out / f"validation_step{step}.json", result)
                if result["metrics"]["valid_miou"] > float(best.get("valid_miou", -1)):
                    best = result["metrics"]
                    atomic_torch_save(checkpoint(model, optimizer, scheduler, step, best, sources, args), out / "best_val_miou.pth")
                    print(f"New best step={step}", flush=True)
            atomic_torch_save(checkpoint(model, optimizer, scheduler, step, best, sources, args), out / "latest.pth")
            if step >= args.optimizer_steps: break
    write_json(out / "training_summary.json", {"protocol": "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1",
               "variant": args.variant, "completed_step": step, "best_metrics": best,
               "sources": sources, "test_accessed": False})


if __name__ == "__main__": main()
