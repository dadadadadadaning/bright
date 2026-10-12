#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
for value in (str(ROOT), str(ROOT / "bda_benchmark"), str(ROOT / "bda_benchmark/script/standard_ML")):
    if value not in sys.path:
        sys.path.insert(0, value)

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    get_variant, occurrence_loss, parse_batch, read_list, sha256_file, write_json,
)
from bda_benchmark.model.DamageFormer_BuildingPrototypeMemory_V1 import DamageFormerBuildingPrototypeMemoryV1


def save_checkpoint(path: Path, model, optimizer, step, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": "SPI_BUILDING_PROTOTYPE_MEMORY_V1", "variant": args.variant,
        "step": step, "head": model.head.state_dict(),
        "projection": model.projection.state_dict(), "gain": model.gain.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "parent_ckpt": str(Path(args.parent_ckpt).resolve()),
        "parent_sha256": sha256_file(args.parent_ckpt),
        "memory_path": str(Path(args.memory_path).resolve()),
        "memory_sha256": sha256_file(args.memory_path),
        "train_list_sha256": sha256_file(args.train_list),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser("Train a low-capacity building prototype verifier on the complete training set")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--memory_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--optimizer_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--nonharm_weight", type=float, default=0.1)
    parser.add_argument("--false_positive_weight", type=float, default=0.25)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    spec = get_variant(args.variant)
    if not spec["train_head"]:
        raise RuntimeError("MM0 is evaluation-only and must not be trained")
    if "test" in Path(args.train_list).name.lower():
        raise RuntimeError("this experiment forbids test-list access")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingPrototypeMemoryV1(args.variant).to(device)
    parent_report = model.load_parent(args.parent_ckpt)
    memory_report = model.load_memory(args.memory_path)
    model.freeze_parent()
    parameters = list(model.projection.parameters()) + list(model.head.parameters()) + [model.gain]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["variant"] != args.variant:
            raise RuntimeError("resume variant mismatch")
        model.head.load_state_dict(saved["head"])
        model.projection.load_state_dict(saved["projection"])
        model.gain.data.copy_(saved["gain"].to(device))
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
    train_names = read_list(args.train_list)
    # Some crops contain no positive affected token and are deliberately skipped.
    # Fivefold headroom prevents the iterator from ending before optimizer_steps.
    sample_count = args.optimizer_steps * args.batch_size * args.accumulation_steps * 5
    train_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, train_names, args.crop_size, max_iters=sample_count,
        type="train", use_damage_aware_sampler=True,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "SPI_BUILDING_PROTOTYPE_MEMORY_V1", "variant": args.variant,
        "complete_train_used": True, "train_split_created": False,
        "validation_accessed_during_training": False,
        "validation_updates_memory_or_model": False, "test_accessed": False,
        "parent": parent_report, "memory": memory_report, "arguments": vars(args),
    }
    write_json(output_dir / "manifest.json", manifest)
    model.train(); optimizer.zero_grad(set_to_none=True)
    step = start_step
    accumulated = 0
    for batch in train_loader:
        if step >= args.optimizer_steps:
            break
        pre, post, location, damage, names = parse_batch(batch, device)
        _, _, aux = model(pre, post, names=names, return_aux=True)
        loss, stats = occurrence_loss(
            aux["adjusted_logits_feature"], aux["parent_logits_feature"], aux["delta"],
            damage, location, args.false_positive_weight, args.nonharm_weight,
        )
        if stats["positive_tokens"] == 0 or stats["negative_tokens"] == 0:
            continue
        (loss / args.accumulation_steps).backward()
        accumulated += 1
        if accumulated < args.accumulation_steps:
            continue
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        accumulated = 0; step += 1
        if step % args.log_every == 0:
            row = {"step": step, "total_loss": float(loss.item()), **stats,
                   "gain": float(torch.tanh(model.gain).item()),
                   "delta_abs_mean": float(aux["delta"].detach().abs().mean().item())}
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(output_dir / "latest_resume.pth", model, optimizer, step, args)
    summary = {"protocol": manifest["protocol"], "variant": args.variant, "completed_step": step,
               "validation_accessed_during_training": False, "test_accessed": False}
    if step < args.optimizer_steps:
        raise RuntimeError(
            f"training data iterator ended at step {step}/{args.optimizer_steps}; "
            "too many batches lacked both positive and negative building tokens"
        )
    save_checkpoint(output_dir / "final.pth", model, optimizer, step, args)
    write_json(output_dir / "training_summary.json", summary)
    print(f"Training complete: step={step}; validation was not accessed")


if __name__ == "__main__":
    main()
