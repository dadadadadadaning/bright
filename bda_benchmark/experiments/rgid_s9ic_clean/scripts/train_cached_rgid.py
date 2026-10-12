#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from rgid.cache_dataset import RGIDCacheDataset, collate_cached_samples
from rgid.config import RGIDConfig
from rgid.model import RGIDHead
from rgid.training import train_rgid_head


def _checkpoint_steps(value: str, final_step: int) -> Tuple[int, ...]:
    try:
        steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "checkpoint steps must be comma-separated integers"
        ) from exc
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("checkpoint steps must be positive")
    return tuple(step for step in steps if step <= final_step)


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train RGIDHead from exported feature cache")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--checkpoint-steps", default="100,300,500")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.accumulation <= 0:
        raise ValueError("steps, batch-size and accumulation must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    generator = _seed_everything(args.seed)
    train_cache = Path(args.train_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RGIDCacheDataset(train_cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_cached_samples,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    config = RGIDConfig()
    head = RGIDHead(
        config.in_channels,
        config.hidden_channels,
        config.num_shared_blocks,
    )
    checkpoints = _checkpoint_steps(args.checkpoint_steps, args.steps)
    result = train_rgid_head(
        head=head,
        train_loader=loader,
        config=config,
        output_dir=output_dir,
        max_optimizer_steps=args.steps,
        accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_steps=checkpoints,
        device=device,
        use_bf16=True,
        gradient_clip_norm=5.0,
        log_jsonl_path=output_dir / "train_log.jsonl",
        checkpoint_extra={
            "train_cache": os.path.abspath(train_cache),
            "seed": int(args.seed),
        },
    )
    summary = {
        "optimizer_steps": result["optimizer_steps"],
        "checkpoint": str(output_dir / f"rgid_step_{args.steps}.pth"),
        "train_cache": str(train_cache),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
