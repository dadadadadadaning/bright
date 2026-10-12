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

from bda_benchmark.model.DamageFormer_ModelCentric_V1 import (
    DamageFormerModelCentricV1,
    VARIANTS,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset,
    parse_batch,
    seed_worker,
    segmentation_loss,
    set_random_seed,
)


PROTOCOL = "SPI_MODEL_CENTRIC_TRAIN_ONLY_v1"


def atomic_save(payload, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def hierarchical_auxiliary_loss(aux, damage: torch.Tensor) -> torch.Tensor:
    size = damage.shape[-2:]
    building_logits = F.interpolate(aux["building_logits"], size, mode="bilinear", align_corners=False)[:, 0]
    affected_logits = F.interpolate(aux["affected_logits"], size, mode="bilinear", align_corners=False)[:, 0]
    destroyed_logits = F.interpolate(aux["destroyed_logits"], size, mode="bilinear", align_corners=False)[:, 0]
    valid = damage != 255
    building = (damage > 0) & valid
    affected = (damage >= 2) & valid
    destroyed = (damage == 3) & valid
    return (
        masked_bce(building_logits, building, valid)
        + masked_bce(affected_logits, affected, building)
        + masked_bce(destroyed_logits, destroyed, affected)
    )


def anchor_kl(output: torch.Tensor, parent: torch.Tensor, target: torch.Tensor, threshold: float):
    with torch.no_grad():
        probability = parent.float().softmax(1)
        confidence, prediction = probability.max(1)
        safe = (target != 255) & (prediction == target) & (confidence >= threshold)
    divergence = F.kl_div(F.log_softmax(output.float(), 1), probability, reduction="none").sum(1)
    return (divergence * safe).sum() / safe.sum().clamp_min(1), int(safe.sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train isolated model-centric DamageFormer plugins")
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-hierarchy", type=float, default=0.25)
    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--anchor-confidence", type=float, default=0.85)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    if args.variant == "p0_parent":
        raise ValueError("p0_parent is an evaluation control and has no trainable plugins")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    generator = set_random_seed(args.seed, 0, args.deterministic)
    model = DamageFormerModelCentricV1(args.variant)
    model.load_parent_checkpoint(args.base_checkpoint)
    model.freeze_parent()
    model.to(device)
    model.set_plugin_training()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("selected variant has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)

    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "plugin_last.pth"
    manifest = {
        "protocol": PROTOCOL,
        "variant": args.variant,
        "args": vars(args),
        "train_images": len(dataset),
        "meta_calibration_accessed": False,
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "test_accessed": False,
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Model-centric {args.variant} | frozen parent | trainable={sum(p.numel() for p in trainable):,} "
        f"| steps={args.optimizer_steps} | EBS={args.batch_size * args.accumulation_steps}",
        flush=True,
    )
    print("Meta-calibration/confirm/Val450/test accessed: NEVER / NEVER / NEVER / NEVER", flush=True)

    optimizer.zero_grad(set_to_none=True)
    step = micro = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, _, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                _, output, aux = model(pre, post, return_aux=True)
                semantic, ce, lovasz = segmentation_loss(output, damage, ce_fn, True, 1.0, 1.0, 255)
                parent = F.interpolate(
                    aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False
                )
                anchor, safe = anchor_kl(output, parent.detach(), damage, args.anchor_confidence)
                hierarchy = output.sum() * 0.0
                if model.variant_config.hierarchy:
                    hierarchy = hierarchical_auxiliary_loss(aux, damage)
                loss = semantic + args.lambda_hierarchy * hierarchy + args.lambda_anchor * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            (loss / args.accumulation_steps).backward()
            micro += 1
            if micro < args.accumulation_steps:
                continue
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if step == 1 or step % args.log_every == 0:
                row = {
                    "optimizer_step": step,
                    "variant": args.variant,
                    "loss": float(loss.detach()),
                    "ce": float(ce.detach()),
                    "lovasz": float(lovasz.detach()),
                    "hierarchy_aux": float(hierarchy.detach()),
                    "anchor": float(anchor.detach()),
                    "safe_pixels": safe,
                    "gradient_norm_before_clip": float(gradient),
                    "elapsed_seconds": time.perf_counter() - started,
                    **model.diagnostics(),
                }
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save(
                    {
                        **manifest,
                        "optimizer_step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "diagnostics": model.diagnostics(),
                    },
                    checkpoint,
                )
            if step >= args.optimizer_steps:
                break
    print(f"Saved: {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
