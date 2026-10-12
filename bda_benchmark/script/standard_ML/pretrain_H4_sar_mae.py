# -*- coding: utf-8 -*-
"""Train-split-only masked reconstruction pretraining for the H4 SAR encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H4 import (  # noqa: E402
    SARMaskedAutoencoder,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import seed_worker  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = (reconstruction.float() - target.float()).abs()
    return (error * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def load_h3_sar_initialization(
    model: SARMaskedAutoencoder,
    checkpoint_path: str,
) -> Dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if str(checkpoint.get("variant", "")).lower() != "h3":
        raise RuntimeError(
            "SAR-MAE initialization requires a validated H3 checkpoint"
        )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("H3 checkpoint does not contain a model state")
    prefix = "sar_encoder."
    sar_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not sar_state:
        raise RuntimeError("H3 checkpoint contains no sar_encoder parameters")
    model.sar_encoder.load_state_dict(sar_state, strict=True)
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "sha256": file_sha256(checkpoint_path),
        "variant": checkpoint.get("variant"),
        "step": int(checkpoint.get("step", -1)),
        "sar_state_keys": len(sar_state),
    }


def freeze_shallow_sar_encoder(model: SARMaskedAutoencoder) -> None:
    """Keep H3 low-level scattering statistics fixed during masked adaptation."""

    frozen_prefixes = (
        "sar_encoder.stem",
        "sar_encoder.stages.0",
        "sar_encoder.stages.1",
        "sar_encoder.downsamples.0",
        "sar_encoder.downsamples.1",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(frozen_prefixes):
            parameter.requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser("Pretrain H4 native SAR encoder with MIM")
    parser.add_argument("--source_h3_checkpoint", required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.60)
    parser.add_argument("--mask_patch_size", type=int, default=16)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask_ratio must be between zero and one")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SARMaskedAutoencoder().to(device)
    source_provenance = load_h3_sar_initialization(
        model,
        args.source_h3_checkpoint,
    )
    freeze_shallow_sar_encoder(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and args.amp == "fp16"
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and args.amp == "fp16"
        )

    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("protocol") != "H4_SAR_MAE_resume_v1":
            raise RuntimeError("unexpected SAR-MAE resume protocol")
        saved_source = checkpoint.get("source_h3", {})
        if not isinstance(saved_source, Mapping) or (
            saved_source.get("sha256") != source_provenance["sha256"]
        ):
            raise RuntimeError(
                "SAR-MAE resume source differs from the current H3 checkpoint"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint.get("step", 0))

    total_crops = (
        args.optimizer_steps
        * args.train_batch_size
        * args.accumulation_steps
    )
    dataset = HeterogeneousRGBSARDataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        max_iters=total_crops,
        split="train",
        variant="h3",
        sar_stats_path=args.sar_stats_path,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    print("=" * 80, flush=True)
    print("Experiment       : H4-SAR-MAE-TrainOnly", flush=True)
    print(f"H3 initialization: {args.source_h3_checkpoint}", flush=True)
    print("Input            : native one-channel SAR", flush=True)
    print("Frozen SAR       : stem + stages 1-2 + connecting downsamples", flush=True)
    print(f"Mask ratio/patch : {args.mask_ratio}/{args.mask_patch_size}", flush=True)
    print(
        f"Steps/EBS/crops  : {args.optimizer_steps}/"
        f"{args.train_batch_size*args.accumulation_steps}/{total_crops}",
        flush=True,
    )
    print(f"Output           : {output_dir}", flush=True)
    print("=" * 80, flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    micro_step = 0
    running_loss = 0.0
    window_start = time.time()
    while step < args.optimizer_steps:
        try:
            _, sar, _, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            _, sar, _, _, _ = next(iterator)
        sar = sar.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(
                sar,
                mask_ratio=args.mask_ratio,
                patch_size=args.mask_patch_size,
            )
            loss = reconstruction_loss(
                outputs["reconstruction"], sar, outputs["mask"]
            )
            scaled_loss = loss / args.accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SAR-MAE loss at step {step + 1}")
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        running_loss += float(loss.detach().item())
        if micro_step % args.accumulation_steps:
            continue

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            denominator = args.log_every * args.accumulation_steps
            elapsed = max(time.time() - window_start, 1e-6)
            print(
                json.dumps(
                    {
                        "step": step,
                        "masked_l1_loss": running_loss / denominator,
                        "optimizer_steps_per_second": args.log_every / elapsed,
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            running_loss = 0.0
            window_start = time.time()

        if step % args.save_every == 0 or step == args.optimizer_steps:
            resume_payload: Dict[str, object] = {
                "protocol": "H4_SAR_MAE_resume_v1",
                "step": step,
                "model": model.state_dict(),
                "sar_encoder": model.sar_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "source_h3": source_provenance,
                "args": vars(args),
            }
            atomic_save(resume_payload, output_dir / "latest_resume.pth")

    encoder_payload = {
        "protocol": "H4_SAR_MAE_encoder_v1",
        "step": step,
        "sar_encoder": model.sar_encoder.state_dict(),
        "source_h3": source_provenance,
        "args": vars(args),
    }
    atomic_save(encoder_payload, output_dir / "sar_encoder_final.pth")
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "protocol": "H4_SAR_MAE_train_only_v1",
                "completed_step": step,
                "encoder_checkpoint": str(output_dir / "sar_encoder_final.pth"),
                "source_h3": source_provenance,
                "shallow_sar_frozen": True,
                "test_accessed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SAR-MAE pretraining complete at step {step}", flush=True)


if __name__ == "__main__":
    main()
