from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import checkpoint_state, file_sha256
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset,
    parse_batch,
    seed_worker,
    segmentation_loss,
    set_random_seed,
)

from .model import SARFeatureAdapter, forward_frozen_p3_with_sar_adapter


PROTOCOL = "SPI_SAR_ADAPTER_TRAIN_ONLY_v1"
TRAIN_VARIANTS = ("fsa", "rcm")


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def safe_anchor_kl(output, base_output, target, confidence_threshold: float):
    with torch.no_grad():
        base_probability = F.softmax(base_output.float(), dim=1)
        confidence, prediction = base_probability.max(dim=1)
        safe = (target != 255) & (prediction == target) & (confidence >= confidence_threshold)
    per_pixel = F.kl_div(
        F.log_softmax(output.float(), dim=1),
        base_probability,
        reduction="none",
    ).sum(dim=1)
    safe_float = safe.to(per_pixel.dtype)
    return (per_pixel * safe_float).sum() / safe_float.sum().clamp_min(1.0), safe.sum()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train one isolated SAR feature adapter")
    parser.add_argument("--variant", choices=TRAIN_VARIANTS, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--anchor-confidence", type=float, default=0.80)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
        torch.backends.cudnn.allow_tf32 = not args.deterministic
    generator = set_random_seed(args.seed, 0, args.deterministic)

    base_sha = file_sha256(args.base_checkpoint)
    saved_base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(saved_base), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)

    adapter = SARFeatureAdapter(args.variant).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "adapter_last.pth"
    optimizer_step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("variant") != args.variant or saved.get("base_checkpoint_sha256") != base_sha:
            raise RuntimeError("existing adapter checkpoint provenance mismatch")
        optimizer_step = int(saved["optimizer_step"])
        adapter.load_state_dict(saved["adapter_state"], strict=True)
        if optimizer_step < args.optimizer_steps:
            optimizer.load_state_dict(saved["optimizer_state"])
        else:
            print(f"{args.variant}: already complete at step={optimizer_step}", flush=True)
            return 0

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
    if not len(loader):
        raise RuntimeError("empty training loader")
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)
    amp_enabled = device.type == "cuda"
    manifest = {
        "protocol": PROTOCOL,
        "variant": args.variant,
        "base_checkpoint_sha256": base_sha,
        "args": vars(args),
        "train_images": len(dataset),
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("=" * 100, flush=True)
    print(
        f"SAR adapter | {args.variant} | frozen P3 | "
        f"trainable={sum(p.numel() for p in adapter.parameters()):,}",
        flush=True,
    )
    print(
        f"steps={args.optimizer_steps} EBS={args.batch_size * args.accumulation_steps}",
        flush=True,
    )
    print("validation/official instances/test accessed: NEVER / NEVER / NEVER", flush=True)
    print("=" * 100, flush=True)

    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.perf_counter()
    while optimizer_step < args.optimizer_steps:
        for batch in loader:
            pre, post, _, damage, _ = parse_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                _, output, auxiliary = forward_frozen_p3_with_sar_adapter(
                    p3, adapter, pre, post
                )
                semantic, ce, lovasz = segmentation_loss(
                    output, damage, ce_fn, True, 1.0, 1.0, 255
                )
                anchor, safe_pixels = safe_anchor_kl(
                    output,
                    auxiliary["base_output_damage"],
                    damage,
                    args.anchor_confidence,
                )
                loss = semantic + args.lambda_anchor * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step={optimizer_step}")
            (loss / args.accumulation_steps).backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            optimizer_step += 1

            if optimizer_step == 1 or optimizer_step % args.log_every == 0:
                diagnostics = adapter.diagnostics()
                row = {
                    "optimizer_step": optimizer_step,
                    "variant": args.variant,
                    "loss": float(loss.detach()),
                    "semantic_loss": float(semantic.detach()),
                    "ce": float(ce.detach()),
                    "lovasz": float(lovasz.detach()),
                    "anchor": float(anchor.detach()),
                    "safe_pixels": int(safe_pixels.detach()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                    **{
                        key: float(value) if isinstance(value, torch.Tensor) else value
                        for key, value in diagnostics.items()
                    },
                }
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if optimizer_step % args.save_every == 0 or optimizer_step == args.optimizer_steps:
                atomic_save(
                    {
                        **manifest,
                        "optimizer_step": optimizer_step,
                        "adapter_state": adapter.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
            if optimizer_step >= args.optimizer_steps:
                break
    print(f"Saved: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
