# -*- coding: utf-8 -*-
"""Train-only RGB/SAR joint feature pretraining for H5-C1 adapters."""
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
from bda_benchmark.model.DamageFormer_RGBSAR_H5_BC import (  # noqa: E402
    JointRGBSARFeaturePretrainer,
    build_h5_bc_damageformer,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H5_BC import (  # noqa: E402
    file_sha256,
    load_h4_a4_source,
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


def text_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_pretraining_parameters(
    pretrainer: JointRGBSARFeaturePretrainer,
) -> None:
    for parameter in pretrainer.model.parameters():
        parameter.requires_grad_(False)
    for parameter in pretrainer.model.cross_modal_adapters.parameters():
        parameter.requires_grad_(True)
    for collection in (
        pretrainer.projectors,
        pretrainer.rgb_to_sar,
        pretrainer.sar_to_rgb,
    ):
        for parameter in collection.parameters():
            parameter.requires_grad_(True)


def keep_fixed_encoders_in_eval(
    pretrainer: JointRGBSARFeaturePretrainer,
) -> None:
    pretrainer.model.rgb_encoder.eval()
    pretrainer.model.sar_encoder.eval()
    pretrainer.model.localization_decoder.eval()
    pretrainer.model.localization_head.eval()


def resume_payload(
    pretrainer: JointRGBSARFeaturePretrainer,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    source: Mapping[str, object],
    args,
) -> Dict[str, object]:
    return {
        "protocol": "H5_C_joint_resume_v1",
        "step": int(step),
        "cross_modal_adapters": (
            pretrainer.model.cross_modal_adapters.state_dict()
        ),
        "projectors": pretrainer.projectors.state_dict(),
        "rgb_to_sar": pretrainer.rgb_to_sar.state_dict(),
        "sar_to_rgb": pretrainer.sar_to_rgb.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "source_a4": dict(source),
        "args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Train H5-C joint RGB/SAR feature-space pretraining"
    )
    parser.add_argument("--source_a4_checkpoint", required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.60)
    parser.add_argument("--changed_margin", type=float, default=0.20)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask_ratio must be between zero and one")
    if not -1.0 < args.changed_margin < 1.0:
        parser.error("--changed_margin must be between -1 and 1")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_h5_bc_damageformer("c1", pretrained_rgb=False)
    source = load_h4_a4_source(
        model,
        args.source_a4_checkpoint,
        require_exact=False,
    )
    pretrainer = JointRGBSARFeaturePretrainer(model).to(device)
    configure_pretraining_parameters(pretrainer)
    trainable = [
        parameter for parameter in pretrainer.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled and args.amp == "fp16",
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and args.amp == "fp16"
        )

    step = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("protocol") != "H5_C_joint_resume_v1":
            raise RuntimeError("unexpected H5-C pretraining resume protocol")
        saved_source = checkpoint.get("source_a4", {})
        if (
            not isinstance(saved_source, Mapping)
            or saved_source.get("sha256") != source["sha256"]
        ):
            raise RuntimeError("H5-C resume uses a different A4 source")
        pretrainer.model.cross_modal_adapters.load_state_dict(
            checkpoint["cross_modal_adapters"],
            strict=True,
        )
        pretrainer.projectors.load_state_dict(
            checkpoint["projectors"],
            strict=True,
        )
        pretrainer.rgb_to_sar.load_state_dict(
            checkpoint["rgb_to_sar"],
            strict=True,
        )
        pretrainer.sar_to_rgb.load_state_dict(
            checkpoint["sar_to_rgb"],
            strict=True,
        )
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

    contract = {
        "name": "H5-C1_train_only_joint_feature_pretraining",
        "independent_rgb_sar_encoders": True,
        "pretrained_modules": "stage3_stage4_symmetric_adapters",
        "unchanged_definition": "background_or_intact",
        "changed_definition": "damaged_or_destroyed_grouped",
        "severity_labels_used_as_targets": False,
        "masked_prediction": "intact_building_feature_space_only",
        "pixel_reconstruction": False,
        "loss_weights": {
            "unchanged_alignment": 1.0,
            "building_feature_prediction": 0.5,
            "changed_unique": 0.5,
            "variance": 0.05,
        },
        "supervised_finetuning_reuses_pretraining_losses": False,
    }
    print("=" * 80, flush=True)
    print("Experiment       : H5-C1-Joint-RGBSAR-Feature-Pretraining", flush=True)
    print(f"A4 initialization: {args.source_a4_checkpoint}", flush=True)
    print("Encoders         : RGB/SAR independent and frozen", flush=True)
    print("Trainable        : symmetric stage-3/4 adapters + disposable heads", flush=True)
    print("Alignment        : background + intact only", flush=True)
    print("Changed regions  : damaged/destroyed grouped as modality-unique", flush=True)
    print("Masked prediction: intact-building feature space; no pixel MAE", flush=True)
    print(
        f"Steps/EBS/crops  : {args.optimizer_steps}/"
        f"{args.train_batch_size*args.accumulation_steps}/{total_crops}",
        flush=True,
    )
    print(f"Output           : {output_dir}", flush=True)
    print("=" * 80, flush=True)

    pretrainer.train()
    keep_fixed_encoders_in_eval(pretrainer)
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    micro_step = 0
    running: Dict[str, float] = {}
    window_start = time.time()
    while step < args.optimizer_steps:
        try:
            rgb, sar, _, damage, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            rgb, sar, _, damage, _ = next(iterator)
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            losses = pretrainer(
                rgb,
                sar,
                damage,
                mask_ratio=args.mask_ratio,
                changed_margin=args.changed_margin,
            )
            loss = losses["joint_pretrain_loss"]
            scaled_loss = loss / args.accumulation_steps
        if not torch.isfinite(loss):
            values = {
                key: float(value.detach().float().item())
                for key, value in losses.items()
            }
            raise FloatingPointError(
                f"non-finite H5-C loss at step {step + 1}: {values}"
            )
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().item())
        if micro_step % args.accumulation_steps:
            continue

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable, 5.0)
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
                        **{
                            key: value / denominator
                            for key, value in running.items()
                        },
                        "optimizer_steps_per_second": (
                            args.log_every / elapsed
                        ),
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
            running = {}
            window_start = time.time()

        if step % args.save_every == 0 or step == args.optimizer_steps:
            atomic_save(
                resume_payload(
                    pretrainer,
                    optimizer,
                    scaler,
                    step,
                    source,
                    args,
                ),
                output_dir / "latest_resume.pth",
            )

    final_path = output_dir / "joint_adapters_final.pth"
    atomic_save(
        {
            "protocol": "H5_C_joint_adapters_v1",
            "step": int(step),
            "cross_modal_adapters": pretrainer.trainable_adapter_state(),
            "source_a4": source,
            "pretraining_contract": contract,
            "train_list_sha256": text_sha256(args.train_data_list_path),
            "sar_stats_path": str(Path(args.sar_stats_path).resolve()),
            "sar_stats_sha256": file_sha256(args.sar_stats_path),
            "args": vars(args),
        },
        final_path,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "protocol": "H5_C_joint_pretraining_summary_v1",
                "completed_step": step,
                "adapter_checkpoint": str(final_path),
                "source_a4": source,
                "pretraining_contract": contract,
                "test_accessed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"H5-C joint pretraining complete at step {step}", flush=True)


if __name__ == "__main__":
    main()
