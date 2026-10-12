#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from variant_common import (
    VariantCacheDataset,
    bootstrap,
    collate_variant,
    parse_steps,
)


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser("Train RGID with a GT instance sidecar")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--gt-sidecar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-steps", default="100,300,500")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    bootstrap(args.experiment_root)
    from rgid.config import RGIDConfig
    from rgid.model import RGIDHead
    from rgid.training import train_rgid_head

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    train_cache = Path(args.train_cache).expanduser().resolve()
    gt_sidecar = Path(args.gt_sidecar).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if any(output_dir.glob("rgid_step_*.pth")):
        raise FileExistsError(f"Output already has checkpoints: {output_dir}")
    dataset = VariantCacheDataset(train_cache, gt_sidecar)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        collate_fn=collate_variant,
    )
    config = RGIDConfig()
    head = RGIDHead(
        config.in_channels, config.hidden_channels, config.num_shared_blocks
    )
    result = train_rgid_head(
        head=head,
        train_loader=loader,
        config=config,
        output_dir=output_dir,
        max_optimizer_steps=args.steps,
        accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_steps=parse_steps(args.checkpoint_steps, args.steps),
        device=args.device,
        use_bf16=True,
        gradient_clip_norm=5.0,
        log_jsonl_path=output_dir / "train_log.jsonl",
        checkpoint_extra={
            "train_cache": os.path.abspath(train_cache),
            "gt_sidecar_dir": os.path.abspath(gt_sidecar),
            "gt_variant": "damage_class_cc",
            "seed": int(args.seed),
        },
    )
    print(result["last_loss"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
