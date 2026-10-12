from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_SafeAO_V2 import DamageFormerSafeAOV2, VARIANTS
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset, parse_batch, seed_worker, segmentation_loss, set_random_seed,
)
from bda_benchmark.experiments.spi_model_centric_v1.train import anchor_kl


PROTOCOL = "SPI_SAFE_AFFECTED_MODEL_TRAIN_ONLY_v2"


def balanced_occurrence_loss(logits, damage, building):
    logits = F.interpolate(logits, damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0].float()
    valid = (damage != 255) & (building > 0)
    positive = valid & (damage >= 2)
    negative = valid & (damage == 1)
    loss = F.binary_cross_entropy_with_logits(logits, positive.float(), reduction="none")
    positive_loss = (loss * positive).sum() / positive.sum().clamp_min(1)
    negative_loss = (loss * negative).sum() / negative.sum().clamp_min(1)
    return 0.5 * (positive_loss + negative_loss), int(positive.sum()), int(negative.sum())


def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train shallow class-safe AO V2")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-occurrence", type=float, default=0.50)
    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--anchor-confidence", type=float, default=0.85)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.variant == "a0_parent":
        raise ValueError("a0_parent is evaluation-only")
    device = torch.device(args.device)
    generator = set_random_seed(args.seed, 0, False)
    model = DamageFormerSafeAOV2(args.variant)
    model.load_parent_checkpoint(args.base_checkpoint)
    model.freeze_parent()
    model.to(device)
    model.set_plugin_training()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)
    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker, generator=generator,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "safe_ao_last.pth"
    manifest = {
        "protocol": PROTOCOL, "variant": args.variant, "args": vars(args),
        "train_images": len(dataset), "meta_calibration_accessed": False,
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Safe AO V2 {args.variant} | frozen P3 | trainable={sum(p.numel() for p in parameters):,} "
        f"steps={args.optimizer_steps} EBS={args.batch_size * args.accumulation_steps}", flush=True,
    )
    optimizer.zero_grad(set_to_none=True)
    step = micro = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, location, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                _, output, aux = model(pre, post, return_aux=True)
                semantic, ce, lovasz = segmentation_loss(output, damage, ce_fn, True, 1.0, 1.0, 255)
                occurrence, positive, negative = balanced_occurrence_loss(
                    aux["occurrence_logits"], damage, location
                )
                parent = F.interpolate(
                    aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False
                )
                anchor, safe = anchor_kl(output, parent.detach(), damage, args.anchor_confidence)
                loss = semantic + args.lambda_occurrence * occurrence + args.lambda_anchor * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / args.accumulation_steps).backward()
            micro += 1
            if micro < args.accumulation_steps:
                continue
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            model.project_safe_parameters()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if step == 1 or step % args.log_every == 0:
                confidence = aux.get("match_confidence", torch.ones_like(aux["affected_delta"]))
                row = {
                    "optimizer_step": step, "variant": args.variant,
                    "loss": float(loss.detach()), "ce": float(ce.detach()),
                    "lovasz": float(lovasz.detach()), "occurrence_loss": float(occurrence.detach()),
                    "anchor": float(anchor.detach()), "positive_pixels": positive,
                    "negative_pixels": negative, "safe_pixels": safe,
                    "gradient_norm_before_clip": float(gradient),
                    "affected_gain": float(model.affected_residual.gain.detach()),
                    "delta_mean": float(aux["affected_delta"].detach().float().mean()),
                    "delta_max": float(aux["affected_delta"].detach().float().max()),
                    "confidence_mean": float(confidence.detach().float().mean()),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save(
                    {**manifest, "optimizer_step": step, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict()}, checkpoint
                )
            if step >= args.optimizer_steps:
                break
    print(f"Saved: {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
